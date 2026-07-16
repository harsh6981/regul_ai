"""
duty_calculator/scraper.py — ICEGATE Duty Structure Scraper (sync port)
==========================================================================
Adapted from ICEDutyAI's backend/scraper.py. The original used httpx's
AsyncClient because it ran under FastAPI; RegulAI's routes are plain
sync Flask views, so this uses a `requests.Session` instead. The
multi-step navigation, HTML parsing, and fallback logic are unchanged.

The ICEGATE site requires a multi-step navigation flow:
1. GET the Trade Guide page (establish session/cookies)
2. POST to Tariff-head-details with CTH code
3. Navigate to/POST the Structure-of-Duty page

TTL cache (500 items, 12-hour TTL) avoids hitting ICEGATE repeatedly
for the same CTH+country pair within a work session.
"""
from __future__ import annotations

import re
import logging
from typing import Dict, Any
from cachetools import TTLCache
import requests
import urllib3

# ICEGATE serves over a self-signed/old TLS config in some environments;
# the original scraper disabled verification (verify=False) for this
# reason. Suppress the resulting InsecureRequestWarning noise.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("regulai.duty_scraper")

# In-memory TTL cache: 500 items, 12-hour TTL
_cache = TTLCache(maxsize=500, ttl=43200)

ICEGATE_BASE = "https://www.old.icegate.gov.in/Webappl"
ICEGATE_TRADE_GUIDE = f"{ICEGATE_BASE}/Trade-Guide-on-Imports"
ICEGATE_TARIFF_DETAILS = f"{ICEGATE_BASE}/Tariff-head-details"
ICEGATE_DUTY_STRUCTURE = f"{ICEGATE_BASE}/Structure-of-Duty-for-selected-Tariff"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Standard fallback rates when ICEGATE is unreachable
FALLBACK_RATES = {
    "bcd_tariff": 10.0,
    "bcd_effective": 10.0,
    "bcd_notification": "N/A",
    "aidc_tariff": 0.0,
    "aidc_effective": 0.0,
    "chcess": 0.0,
    "eaidc": 0.0,
    "swc": 10.0,
    "igst": 18.0,
    "igst_notification": "N/A",
    "cc": 0.0,
}


def _parse_rate(text: str) -> float:
    """Extract a numeric rate from text like '7.5%' or '7.50' or 'Nil'."""
    if not text:
        return 0.0
    text = text.strip().replace('%', '').replace(',', '')
    if text.lower() in ('nil', 'n/a', '-', '', 'free'):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_icegate_html(html: str) -> Dict[str, Any]:
    """
    Parse the ICEGATE duty structure HTML table.

    Table columns:
    1. Customs Duty (component name)
    2. Rate of Duty (Tariff)%
    3. Spec Duty
    4. Unit
    5. Notification -SlNo (dropdown)
    6. Rate of Duty (Effective) %
    7. Spec Duty
    8. Unit
    9. Duty Amount
    """
    result: Dict[str, Any] = {"description": "", "rates": {}}

    desc_match = re.search(
        r'DESCRIPTION\s+FOR\s+CTH\s*:?\s*(.*?)(?:</?(?:table|div|tr))',
        html, re.IGNORECASE | re.DOTALL
    )
    if desc_match:
        desc = desc_match.group(1)
        desc = re.sub(r'<[^>]+>', ' ', desc).strip()
        desc = re.sub(r'\s+', ' ', desc)
        if len(desc) > 5:
            result["description"] = desc

    if not result["description"]:
        for pattern in [
            r'DESCRIPTION\s*(?:FOR\s*CTH)?\s*:?\s*</[^>]+>\s*(.*?)</(?:td|div|tr)',
            r'DESCRIPTION[^<]*</(?:b|strong|td)>\s*([^<]+)',
        ]:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                desc = re.sub(r'<[^>]+>', ' ', match.group(1)).strip()
                desc = re.sub(r'\s+', ' ', desc)
                if len(desc) > 5:
                    result["description"] = desc
                    break

    bcd_tariff = 0.0
    bcd_effective = 0.0
    bcd_notification = "N/A"
    aidc_tariff = 0.0
    aidc_effective = 0.0
    chcess = 0.0
    eaidc = 0.0
    swc = 10.0
    igst = 18.0
    igst_notification = "N/A"
    cc = 0.0
    total_duty_pct = 0.0

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)

    duty_amounts: Dict[str, float] = {}

    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
        clean_cells = []
        for c in cells:
            text = re.sub(r'<[^>]+>', ' ', c).strip()
            text = re.sub(r'\s+', ' ', text)
            clean_cells.append(text)

        if not clean_cells:
            continue

        first_cell = clean_cells[0].upper()

        if 'BASIC CUSTOMS DUTY' in first_cell or first_cell.strip() == 'BCD':
            if len(clean_cells) > 1:
                bcd_tariff = _parse_rate(clean_cells[1])
            if len(clean_cells) > 5:
                bcd_effective = _parse_rate(clean_cells[5])
            elif bcd_tariff > 0:
                bcd_effective = bcd_tariff
            if len(clean_cells) > 4:
                notif_text = clean_cells[4]
                notif_match = re.search(r'(\d{3}/\d{4}[-\w]*)', notif_text)
                if notif_match:
                    bcd_notification = notif_match.group(1)
            if len(clean_cells) > 8:
                duty_amounts['bcd'] = _parse_rate(clean_cells[8])

        elif ('CUSTOMS AIDC' in first_cell or first_cell.strip() == 'AIDC') and 'EAIDC' not in first_cell:
            if len(clean_cells) > 1:
                aidc_tariff = _parse_rate(clean_cells[1])
            if len(clean_cells) > 5:
                aidc_effective = _parse_rate(clean_cells[5])
            elif aidc_tariff > 0:
                aidc_effective = aidc_tariff
            if len(clean_cells) > 8:
                duty_amounts['aidc'] = _parse_rate(clean_cells[8])

        elif 'HEALTH' in first_cell or 'CHCESS' in first_cell:
            if len(clean_cells) > 5:
                chcess = _parse_rate(clean_cells[5])
            elif len(clean_cells) > 1:
                chcess = _parse_rate(clean_cells[1])

        elif 'EXCISE AIDC' in first_cell or 'EAIDC' in first_cell:
            if len(clean_cells) > 5:
                eaidc = _parse_rate(clean_cells[5])
            elif len(clean_cells) > 1:
                eaidc = _parse_rate(clean_cells[1])

        elif 'SOCIAL WELFARE' in first_cell or 'SWC' in first_cell or 'SWS' in first_cell:
            if len(clean_cells) > 5:
                val = _parse_rate(clean_cells[5])
                if val > 0:
                    swc = val
            elif len(clean_cells) > 1:
                val = _parse_rate(clean_cells[1])
                if val > 0:
                    swc = val

        elif 'IGST' in first_cell:
            if len(clean_cells) > 1:
                val = _parse_rate(clean_cells[1])
                if val > 0:
                    igst = val
            if len(clean_cells) > 5:
                val = _parse_rate(clean_cells[5])
                if val > 0:
                    igst = val
            if len(clean_cells) > 4:
                notif_text = clean_cells[4]
                notif_match = re.search(r'(\d{3}/\d{4}[-\w]*)', notif_text)
                if notif_match:
                    igst_notification = notif_match.group(1)

        elif 'COMPENSATION' in first_cell or ('COMP' in first_cell and 'CESS' in first_cell):
            if len(clean_cells) > 5:
                cc = _parse_rate(clean_cells[5])
            elif len(clean_cells) > 1:
                cc = _parse_rate(clean_cells[1])
            if len(clean_cells) > 8:
                duty_amounts['cc'] = _parse_rate(clean_cells[8])

    # If we have duty amounts for AV=100000, back-calculate effective AV-based
    # rates. ICEGATE's sample duty amount is always computed for AV=100000.
    icegate_av = 100000
    if duty_amounts.get('aidc', 0) > 0:
        derived_aidc = round(duty_amounts['aidc'] / icegate_av * 100, 4)
        if derived_aidc != aidc_effective and derived_aidc < aidc_effective:
            # ICEGATE's AIDC is applied on BCD, not AV — store the
            # AV-equivalent rate instead, since our calculator's cascade
            # applies every rate directly on AV.
            logger.info(f"AIDC: ICEGATE shows {aidc_effective}% but amount implies {derived_aidc}% on AV")
            aidc_effective = derived_aidc
            aidc_tariff = derived_aidc

    result["rates"] = {
        "bcd_tariff": bcd_tariff,
        "bcd_effective": bcd_effective,
        "bcd_notification": bcd_notification,
        "aidc_tariff": aidc_tariff,
        "aidc_effective": aidc_effective,
        "chcess": chcess,
        "eaidc": eaidc,
        "swc": swc,
        "igst": igst,
        "igst_notification": igst_notification,
        "cc": cc,
        "total_duty_pct": total_duty_pct,
    }

    return result


def fetch_duty_structure(cth: str, country: str = "ALL") -> Dict[str, Any]:
    """
    Fetch duty structure from ICEGATE for a given CTH code and country.
    Sync version (requests) of the original async httpx implementation.

    Args:
        cth: 8-digit CTH (HS Code) e.g. "39076190"
        country: ISO country code or "ALL"

    Returns:
        Dict with cth, country, description, source, and rates
    """
    cache_key = f"{cth}:{country}"

    if cache_key in _cache:
        logger.info(f"Cache hit for {cache_key}")
        return _cache[cache_key]

    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)

            # Step 1: GET the trade guide page to establish session/cookies
            resp1 = session.get(ICEGATE_TRADE_GUIDE, timeout=20, verify=False)
            logger.info(f"Step 1 (Trade Guide): status={resp1.status_code}")

            # Step 2: POST to Tariff-head-details with CTH code
            form_data = {
                "cth": cth,
                "item": "",
                "country": "",
                "submitbutton": "Search",
            }
            resp2 = session.post(
                ICEGATE_TARIFF_DETAILS,
                data=form_data,
                headers={"Referer": ICEGATE_TRADE_GUIDE, "Content-Type": "application/x-www-form-urlencoded"},
                timeout=20,
                verify=False,
            )
            logger.info(f"Step 2 (Tariff Details): status={resp2.status_code}")

            if resp2.status_code != 200:
                logger.warning(f"ICEGATE tariff details returned {resp2.status_code}")
                return _build_fallback(cth, country)

            # Step 3: Navigate to Structure of Duty page
            duty_form = {
                "tariffno": cth,
                "country": country if country != "ALL" else "",
            }
            headers3 = {"Referer": ICEGATE_TARIFF_DETAILS, "Content-Type": "application/x-www-form-urlencoded"}
            resp3 = session.post(ICEGATE_DUTY_STRUCTURE, data=duty_form, headers=headers3, timeout=20, verify=False)
            logger.info(f"Step 3 (Duty Structure): status={resp3.status_code}")

            if resp3.status_code != 200:
                resp3 = session.get(
                    f"{ICEGATE_DUTY_STRUCTURE}?tariffno={cth}&country={country}",
                    headers=headers3,
                    timeout=20,
                    verify=False,
                )
                logger.info(f"Step 3 alt (GET): status={resp3.status_code}")

            html = resp3.text

            if not html or len(html) < 200:
                logger.warning(f"ICEGATE returned empty/short response for {cth}")
                return _build_fallback(cth, country)

            if any(msg in html.lower() for msg in [
                'access denied', 'waf', 'firewall', 'blocked',
                'no data found', 'no record', 'invalid tariff'
            ]):
                logger.warning(f"ICEGATE blocked or no data for {cth}")
                return _build_fallback(cth, country, note="ICEGATE returned no data or access was restricted")

            parsed = _parse_icegate_html(html)

            result = {
                "cth": cth,
                "country": country,
                "description": parsed.get("description", ""),
                "source": "icegate_live",
                "rates": parsed.get("rates", FALLBACK_RATES.copy()),
            }

            rates = result["rates"]
            has_valid_data = (
                rates.get("bcd_tariff", 0) > 0 or
                rates.get("bcd_effective", 0) > 0 or
                rates.get("igst", 0) > 0
            )

            if not has_valid_data:
                logger.warning(f"No valid rates parsed for {cth}, falling back")
                return _build_fallback(cth, country, description=parsed.get("description", ""))

            _cache[cache_key] = result
            logger.info(
                f"Fetched rates for {cth}: BCD={rates.get('bcd_effective')}%, "
                f"AIDC={rates.get('aidc_effective')}%, IGST={rates.get('igst')}%, SWC={rates.get('swc')}%"
            )

            return result

    except requests.exceptions.Timeout:
        logger.error(f"ICEGATE timeout for {cth}")
        return _build_fallback(cth, country)
    except Exception as e:
        logger.error(f"ICEGATE error for {cth}: {str(e)}")
        return _build_fallback(cth, country)


def _build_fallback(cth: str, country: str, description: str = "", note: str = "") -> Dict[str, Any]:
    """Build a fallback response with standard estimated rates."""
    return {
        "cth": cth,
        "country": country,
        "description": description or f"Product under CTH {cth}",
        "source": "fallback_estimated",
        "note": note or "ICEGATE was unreachable. Using standard estimated rates.",
        "rates": FALLBACK_RATES.copy(),
    }


# Country code to name mapping (22 major trading partners)
COUNTRIES = {
    "ALL": "All Countries (Default)",
    "CHN": "China",
    "USA": "United States",
    "ARE": "UAE",
    "SAU": "Saudi Arabia",
    "DEU": "Germany",
    "KOR": "South Korea",
    "JPN": "Japan",
    "SGP": "Singapore",
    "MYS": "Malaysia",
    "THA": "Thailand",
    "IDN": "Indonesia",
    "VNM": "Vietnam",
    "TWN": "Taiwan",
    "GBR": "United Kingdom",
    "ITA": "Italy",
    "FRA": "France",
    "AUS": "Australia",
    "BGD": "Bangladesh",
    "LKA": "Sri Lanka",
    "PAK": "Pakistan",
    "NPL": "Nepal",
    "MUS": "Mauritius",
}