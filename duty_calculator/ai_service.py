"""
duty_calculator/ai_service.py — AI features for the duty calculator
=======================================================================
Adapted from ICEDutyAI's backend/ai_service.py, which called Claude
Haiku directly via the `anthropic` SDK. RegulAI already has a working
Gemini integration (rag_search.call_gemini), so this reuses that
instead of adding a second LLM provider + API key:

1. Product classification to CTH (HS Code)
2. Plain-English summary of a duty calculation

Same prompts, output shape, and fallback behavior as the original —
only the model-calling plumbing changed.
"""
from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any, Optional

from rag_search import call_gemini

logger = logging.getLogger("regulai.duty_ai")


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM response before JSON parsing."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def classify_product(description: str, country: str = "ALL") -> Dict[str, Any]:
    """
    Use Gemini to classify a product description into an Indian Customs
    Tariff Heading (CTH). Returns top 3 CTH suggestions with confidence.
    """
    prompt = f"""You are an expert Indian Customs Tariff classifier. Given a product description,
identify the most likely 8-digit CTH (Customs Tariff Heading) codes under the Indian Harmonized System.

Product Description: "{description}"
Country of Origin: {country}

Return a JSON object with exactly this structure (no markdown, no explanation, just JSON):
{{
    "suggestions": [
        {{
            "cth": "XXXXXXXX",
            "description": "Official tariff description",
            "confidence": "high",
            "reason": "Brief explanation of why this CTH matches"
        }},
        {{
            "cth": "XXXXXXXX",
            "description": "Official tariff description",
            "confidence": "medium",
            "reason": "Brief explanation"
        }},
        {{
            "cth": "XXXXXXXX",
            "description": "Official tariff description",
            "confidence": "low",
            "reason": "Brief explanation"
        }}
    ]
}}

Rules:
- Return exactly 3 suggestions ordered by confidence (high, medium, low)
- CTH must be exactly 8 digits (no dots or spaces)
- Use actual Indian customs tariff headings from the ITC-HS schedule
- confidence must be one of: "high", "medium", "low"
- Be specific about the tariff classification reasoning
"""

    try:
        result_raw = call_gemini(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a precise, JSON-only Indian customs tariff classification assistant.",
            max_tokens=1024,
            temperature=0.2,
            thinking_budget=0,
        )
        text = _strip_markdown_fences(result_raw["text"])
        result = json.loads(text)

        if "suggestions" not in result or not isinstance(result["suggestions"], list):
            raise ValueError("Invalid response structure")

        for s in result["suggestions"]:
            s["cth"] = re.sub(r'[^0-9]', '', str(s.get("cth", "")))[:8]
            if len(s["cth"]) < 8:
                s["cth"] = s["cth"].ljust(8, '0')
            s["confidence"] = s.get("confidence", "low").lower()
            if s["confidence"] not in ("high", "medium", "low"):
                s["confidence"] = "medium"

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini classification response: {e}")
        return _fallback_classification(description)
    except Exception as e:
        logger.error(f"Gemini classification error: {e}")
        return _fallback_classification(description)


def generate_summary(
    cth: str,
    description: str,
    country: str,
    cif_value: float,
    breakup: Dict[str, Any],
    fta_info: Optional[Dict[str, Any]] = None,
    source: str = "icegate_live",
) -> Dict[str, Any]:
    """Generate a plain-English AI summary of the duty calculation."""
    fta_text = ""
    if fta_info and fta_info.get("fta_applicable"):
        fta_names = ", ".join(f.get("name", "") for f in fta_info.get("ftas", []))
        fta_text = f"FTA applicable: {fta_names}"
    else:
        fta_text = "No FTA benefit available for this origin country."

    prompt = f"""You are an expert Indian customs duty advisor. Summarize the following duty calculation in plain English.

Product: {description} (CTH: {cth})
Country of Origin: {country}
CIF Value: ₹{cif_value:,.2f}

Duty Calculation:
- Assessable Value: ₹{breakup['assessable_value']:,.2f}
- BCD ({breakup['bcd_rate']}%): ₹{breakup['bcd_amount']:,.2f}
- AIDC ({breakup['aidc_rate']}%): ₹{breakup['aidc_amount']:,.2f}
- Health Cess ({breakup['chcess_rate']}%): ₹{breakup['chcess_amount']:,.2f}
- EAIDC ({breakup['eaidc_rate']}%): ₹{breakup['eaidc_amount']:,.2f}
- SWC ({breakup['swc_rate']}%): ₹{breakup['swc_amount']:,.2f}
- IGST ({breakup['igst_rate']}%): ₹{breakup['igst_amount']:,.2f}
- Comp Cess ({breakup['cc_rate']}%): ₹{breakup['cc_amount']:,.2f}
- Total Duty: ₹{breakup['total_duty']:,.2f}
- Effective Rate: {breakup['effective_rate']}%
- Total Landed Cost: ₹{breakup['total_landed_cost']:,.2f}

FTA Status: {fta_text}
Data Source: {source}

Return a JSON object (no markdown, no explanation, just JSON):
{{
    "headline": "One sentence summarizing total duty as a headline, e.g. 'Total import duty of ₹X.XX (Y% effective rate) applies on this ₹Z shipment of product'",
    "key_points": [
        "Key point 1 about the major duty components",
        "Key point 2 about what makes up the biggest portion",
        "Key point 3 about the total landed cost"
    ],
    "warnings": [
        "Any warnings about rates, compliance, or documentation needed"
    ],
    "advice": "One actionable piece of advice for the importer",
    "fta_hint": "Brief note about FTA status and potential savings",
    "data_source_note": "Note about data freshness and accuracy"
}}
"""

    try:
        result_raw = call_gemini(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a precise, JSON-only Indian customs duty advisor.",
            max_tokens=1024,
            temperature=0.3,
            thinking_budget=0,
        )
        text = _strip_markdown_fences(result_raw["text"])
        return json.loads(text)

    except Exception as e:
        logger.error(f"Gemini summary error: {e}")
        return _fallback_summary(breakup, source, fta_info)


def _fallback_classification(description: str) -> Dict[str, Any]:
    """Fallback when Gemini is unavailable or returns something unparseable."""
    return {
        "suggestions": [
            {
                "cth": "00000000",
                "description": f"AI classification unavailable. Please enter the 8-digit CTH code manually for: {description}",
                "confidence": "low",
                "reason": "The AI classifier could not be reached or returned an unparseable response.",
            }
        ],
        "ai_available": False,
        "note": "AI classification is temporarily unavailable. Please enter the CTH code manually.",
    }


def _fallback_summary(
    breakup: Dict[str, Any],
    source: str,
    fta_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fallback summary when Gemini is unavailable."""
    total = breakup.get("total_duty", 0)
    rate = breakup.get("effective_rate", 0)
    landed = breakup.get("total_landed_cost", 0)

    fta_hint = "No FTA information available."
    if fta_info:
        if fta_info.get("fta_applicable"):
            fta_names = ", ".join(f.get("name", "") for f in fta_info.get("ftas", []))
            fta_hint = f"FTA benefits may be available under: {fta_names}. Check with customs broker."
        else:
            fta_hint = "No FTA benefit available for this country of origin."

    source_note = (
        "Rates fetched live from ICEGATE." if source == "icegate_live"
        else "Using estimated standard rates. ICEGATE was unavailable."
    )

    return {
        "headline": f"Total import duty of ₹{total:,.2f} ({rate}% effective rate) applies on this shipment",
        "key_points": [
            f"Total customs duty payable is ₹{total:,.2f}",
            f"The effective duty rate is {rate}% of assessable value",
            f"Total landed cost including all duties: ₹{landed:,.2f}",
        ],
        "warnings": [
            "Always verify rates with your customs broker before import",
            "Rates may change based on notifications issued by CBIC",
        ],
        "advice": "Consult a licensed customs broker for documentation requirements and the latest applicable rates.",
        "fta_hint": fta_hint,
        "data_source_note": source_note,
    }