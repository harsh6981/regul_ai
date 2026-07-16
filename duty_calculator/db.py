"""
duty_calculator/db.py — MongoDB data-access layer for duty HS codes
======================================================================
Replaces ICEDutyAI's original `db/hs_codes_db.py` (SQLite). Same shape
of methods, but reads/writes the `duty_hs_codes`, `trade_defense`, and
`duty_icegate_cache` collections in RegulAI's existing `regulai` Mongo
database (populated by migrate_sqlite_to_mongo.py) rather than a
bundled SQLite file — so the calculator lives in the same datastore as
the rest of RegulAI instead of a second, parallel database engine.

Uses the MONGO_URI env var already required by the rest of RegulAI
(app.py, auth.py, mongo_pipeline.py, etc.) — no new connection info
needed.
"""
from __future__ import annotations

import os
import re
import datetime
from typing import Optional, List, Dict, Any
from pymongo import MongoClient


_client = None
_db = None


def _get_db():
    """Lazy singleton, mirrors the pattern used elsewhere in RegulAI (auth.py, app.py)."""
    global _client, _db
    if _db is None:
        mongo_uri = os.getenv("MONGO_URI")
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5_000)
        _db = _client["regulai"]
    return _db


class DutyHSCodesDB:
    """Mongo-backed access layer for duty calculator reference data."""

    def get_hs_code_info(self, hs_code: str) -> Optional[Dict[str, Any]]:
        """Get CUSTADA baseline rates for an HS code (used as a fallback/cross-check
        alongside the live ICEGATE scrape)."""
        hs_8digit = re.sub(r"\D", "", hs_code or "").ljust(8, "0")[:8]
        doc = _get_db()["duty_hs_codes"].find_one({"hs_8digit": hs_8digit}, {"_id": 0})
        return doc

    def get_chapter_hs_codes(self, chapter: int) -> List[Dict[str, Any]]:
        """Get all HS codes in a chapter."""
        cursor = _get_db()["duty_hs_codes"].find({"chapter": chapter}, {"_id": 0}).sort("hs_8digit", 1)
        return list(cursor)

    def search_hs_codes(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search HS codes by description (case-insensitive substring match)."""
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        cursor = (
            _get_db()["duty_hs_codes"]
            .find({"description": pattern}, {"_id": 0})
            .sort("hs_8digit", 1)
            .limit(limit)
        )
        return list(cursor)

    def get_custada_rates(self, hs_code: str) -> Optional[Dict[str, float]]:
        """Get CUSTADA rates breakdown in the shape the calculator expects."""
        info = self.get_hs_code_info(hs_code)
        if not info:
            return None
        return {
            "hs_code": info["hs_8digit"],
            "bcd": info.get("custada_bcd", 0.0),
            "aidc": info.get("custada_aidc", 0.0),
            "igst": info.get("custada_igst", 18.0),
            "swc": info.get("custada_swc", 10.0),
            "cc": info.get("custada_cc", 0.0),
            "edition": info.get("custada_edition"),
        }

    def get_trade_defense_measures(self, hs_code: str, origin_country: str) -> List[Dict[str, Any]]:
        """
        Get applicable ADD/CVD/Safeguard measures for an HS code and country.

        Returns measures that:
        - Have HS code in range (hs_code_start <= hs_code <= hs_code_end)
        - Match origin country or are universal ("ALL")
        - Are currently active (effective_date <= today <= expiry_date)
        """
        hs_8digit = re.sub(r"\D", "", hs_code or "").ljust(8, "0")[:8]
        today = datetime.date.today().isoformat()

        cursor = _get_db()["trade_defense"].find(
            {
                "hs_code_start": {"$lte": hs_8digit},
                "hs_code_end": {"$gte": hs_8digit},
                "origin_country": {"$in": [origin_country.upper(), "ALL"]},
                "effective_date": {"$lte": today},
                "expiry_date": {"$gte": today},
            },
            {"_id": 0},
        ).sort("duty_rate_percent", -1)

        return list(cursor)

    def save_icegate_rate(self, cth: str, country_code: str, rates: Dict[str, float]):
        """Persist a fetched ICEGATE rate so it survives app restarts (the
        in-memory TTLCache in scraper.py handles the hot path; this is the
        cold-start / audit-trail copy)."""
        _get_db()["duty_icegate_cache"].update_one(
            {"cth": cth, "country_code": country_code},
            {"$set": {
                "cth": cth,
                "country_code": country_code,
                **rates,
                "cached_at": datetime.datetime.utcnow().isoformat(),
            }},
            upsert=True,
        )

    def get_icegate_rate(self, cth: str, country_code: str) -> Optional[Dict[str, float]]:
        """Get a persisted ICEGATE rate (cold-start cache, independent of the
        in-process TTLCache in scraper.py)."""
        doc = _get_db()["duty_icegate_cache"].find_one(
            {"cth": cth, "country_code": country_code}, {"_id": 0}
        )
        return doc

    def save_calculation(
        self, hs_code: str, origin_country: str, cif_value: float,
        rates_source: str, total_duty: float, effective_rate: float,
        user_email: Optional[str] = None,
    ):
        """Save calculation for audit trail."""
        _get_db()["duty_calculations"].insert_one({
            "hs_code": hs_code,
            "origin_country": origin_country,
            "cif_value": cif_value,
            "rates_source": rates_source,
            "total_duty": total_duty,
            "effective_rate": effective_rate,
            "user_email": user_email,
            "calculation_timestamp": datetime.datetime.utcnow().isoformat(),
        })

    def get_trade_defense_stats(self) -> Dict[str, Any]:
        col = _get_db()["trade_defense"]
        total_measures = col.count_documents({})
        countries = len(col.distinct("origin_country"))

        pipeline = [
            {"$match": {"duty_rate_percent": {"$gt": 0}}},
            {"$group": {
                "_id": None,
                "avg_rate": {"$avg": "$duty_rate_percent"},
                "max_rate": {"$max": "$duty_rate_percent"},
                "min_rate": {"$min": "$duty_rate_percent"},
            }},
        ]
        agg = list(col.aggregate(pipeline))
        rates = agg[0] if agg else {"avg_rate": 0, "max_rate": 0, "min_rate": 0}

        return {
            "total_measures": total_measures,
            "countries_covered": countries,
            "avg_duty_rate": round(rates.get("avg_rate") or 0, 2),
            "max_duty_rate": rates.get("max_rate") or 0,
            "min_duty_rate": rates.get("min_rate") or 0,
        }

    def get_db_stats(self) -> Dict[str, Any]:
        db = _get_db()
        return {
            "hs_codes": db["duty_hs_codes"].count_documents({}),
            "chapters": len(db["duty_hs_codes"].distinct("chapter")),
            "cached_rates": db["duty_icegate_cache"].count_documents({}),
            "calculations": db["duty_calculations"].count_documents({}),
            "trade_defense_measures": db["trade_defense"].count_documents({}),
        }


# Global instance, mirrors get_db() singleton pattern from the original module
_instance: Optional[DutyHSCodesDB] = None


def get_duty_db() -> DutyHSCodesDB:
    global _instance
    if _instance is None:
        _instance = DutyHSCodesDB()
    return _instance
    