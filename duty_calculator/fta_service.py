"""
duty_calculator/fta_service.py — FTA (Free Trade Agreement) Eligibility Checker
=================================================================================
Ported unchanged from ICEDutyAI's backend/fta_service.py. Static
reference data + a lookup function, no I/O, no changes needed.

Checks whether a given CTH code and country of origin qualifies for
preferential duty rates under India's active FTAs.
"""
from __future__ import annotations

from typing import Dict, Any, List

# India's active FTAs with member countries
FTA_AGREEMENTS = {
    "ASEAN-India FTA (AIFTA)": {
        "countries": ["SGP", "MYS", "THA", "IDN", "PHL", "VNM", "MMR", "KHM", "LAO", "BRN"],
        "description": "ASEAN-India Free Trade Area — covers trade in goods with preferential tariffs",
        "note": "Certificate of Origin (CO) Form AI required. Preferential BCD rates may apply.",
    },
    "India-Japan CEPA": {
        "countries": ["JPN"],
        "description": "Comprehensive Economic Partnership Agreement between India and Japan",
        "note": "Certificate of Origin required. Phased tariff elimination on many products.",
    },
    "India-Korea CEPA": {
        "countries": ["KOR"],
        "description": "Comprehensive Economic Partnership Agreement between India and South Korea",
        "note": "Certificate of Origin (CO Form KR) required. Check product-specific concessions.",
    },
    "India-UAE CEPA": {
        "countries": ["ARE"],
        "description": "India-UAE Comprehensive Economic Partnership Agreement (effective May 2022)",
        "note": "Certificate of Origin required. Significant tariff concessions on many items.",
    },
    "India-Australia ECTA": {
        "countries": ["AUS"],
        "description": "India-Australia Economic Cooperation and Trade Agreement (effective Dec 2022)",
        "note": "Certificate of Origin required. Phased tariff reductions on covered products.",
    },
    "SAFTA": {
        "countries": ["PAK", "BGD", "LKA", "NPL", "BTN", "MDV"],
        "description": "South Asian Free Trade Area — preferential rates among SAARC nations",
        "note": "SAFTA Certificate of Origin required. Sensitive lists may apply to some products.",
    },
    "India-Mauritius CECPA": {
        "countries": ["MUS"],
        "description": "India-Mauritius Comprehensive Economic Cooperation and Partnership Agreement",
        "note": "Certificate of Origin required. Covers select tariff lines with preferential rates.",
    },
    "India-Singapore CECA": {
        "countries": ["SGP"],
        "description": "India-Singapore Comprehensive Economic Cooperation Agreement (bilateral, in addition to AIFTA)",
        "note": "Bilateral agreement. Importer can choose between CECA and AIFTA rates — whichever is more beneficial.",
    },
    "India-Malaysia CECA": {
        "countries": ["MYS"],
        "description": "India-Malaysia Comprehensive Economic Cooperation Agreement (bilateral)",
        "note": "Bilateral agreement. Check which agreement offers better concession.",
    },
}


def check_fta_eligibility(cth: str, country: str) -> Dict[str, Any]:
    """
    Check FTA eligibility for a given CTH code and country of origin.

    Args:
        cth: 8-digit CTH code
        country: ISO 3-letter country code

    Returns:
        Dict with fta_applicable, ftas list, and notes
    """
    applicable_ftas: List[Dict[str, str]] = []

    for fta_name, fta_info in FTA_AGREEMENTS.items():
        if country.upper() in fta_info["countries"]:
            applicable_ftas.append({
                "name": fta_name,
                "description": fta_info["description"],
                "note": fta_info["note"],
            })

    if applicable_ftas:
        return {
            "fta_applicable": True,
            "cth": cth,
            "country": country,
            "ftas": applicable_ftas,
            "note": (
                f"{len(applicable_ftas)} FTA(s) applicable. "
                "Preferential duty rates may be available with valid Certificate of Origin. "
                "Consult your customs broker for the most beneficial option."
            ),
        }
    else:
        return {
            "fta_applicable": False,
            "cth": cth,
            "country": country,
            "ftas": [],
            "note": (
                "No FTA benefit available for this country of origin. "
                "Standard MFN (Most Favoured Nation) rates will apply."
            ),
        }