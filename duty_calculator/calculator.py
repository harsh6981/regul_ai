"""
duty_calculator/calculator.py — India Customs Duty Cascade Engine
===================================================================
Ported from ICEDutyAI's backend/calculator.py. Pure Python, no
side effects, fully deterministic — no changes needed for RegulAI's
Flask/Mongo stack since this module never touches I/O.

Cascade:
    Landing Charges = CIF x 1%
    Assessable Value (AV) = CIF + Landing
    BCD, AIDC, CHCESS, EAIDC = each computed on AV
    Customs Subtotal = BCD + AIDC + CHCESS + EAIDC
    SWC = Customs Subtotal x 10%   (NOT on AV)
    IGST Base = AV + Customs Subtotal + SWC
    IGST = IGST Base x IGST rate
    CC = IGST Base x CC rate
    Total Duty = BCD + AIDC + CHCESS + EAIDC + SWC + IGST + CC
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DutyRates:
    """Duty rates as percentages (e.g., 7.5 means 7.5%)"""
    bcd: float = 0.0          # Basic Customs Duty %
    aidc: float = 0.0         # Additional Infrastructure Development Cess %
    chcess: float = 0.0       # Custom Health Cess %
    eaidc: float = 0.0        # Excise AIDC %
    swc: float = 10.0         # Social Welfare Surcharge % (standard 10%)
    igst: float = 18.0        # IGST %
    cc: float = 0.0           # Compensation Cess %


@dataclass
class DutyBreakup:
    """Complete duty computation result with every intermediate value."""
    cif_value: float
    quantity: Optional[float]

    landing_charges: float
    assessable_value: float

    bcd_rate: float
    bcd_amount: float
    aidc_rate: float
    aidc_amount: float
    chcess_rate: float
    chcess_amount: float
    eaidc_rate: float
    eaidc_amount: float

    customs_subtotal: float
    swc_rate: float
    swc_amount: float

    igst_base: float
    igst_rate: float
    igst_amount: float

    cc_rate: float
    cc_amount: float

    total_duty: float
    effective_rate: float
    total_landed_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


def round_duty(amount: float) -> float:
    """Round duty amount to 2 decimal places."""
    return round(amount, 2)


def calculate_duty(
    cif_value: float,
    rates: DutyRates,
    quantity: Optional[float] = None,
) -> DutyBreakup:
    """Run the complete Indian customs duty cascade for a CIF value + rate set."""

    landing_charges = round_duty(cif_value * 0.01)
    assessable_value = round_duty(cif_value + landing_charges)

    bcd_amount = round_duty(assessable_value * rates.bcd / 100)
    aidc_amount = round_duty(assessable_value * rates.aidc / 100)
    chcess_amount = round_duty(assessable_value * rates.chcess / 100)
    eaidc_amount = round_duty(assessable_value * rates.eaidc / 100)

    customs_subtotal = round_duty(bcd_amount + aidc_amount + chcess_amount + eaidc_amount)

    # SWC is on customs duties only, NOT on assessable value
    swc_amount = round_duty(customs_subtotal * rates.swc / 100)

    igst_base = round_duty(assessable_value + customs_subtotal + swc_amount)
    igst_amount = round_duty(igst_base * rates.igst / 100)
    cc_amount = round_duty(igst_base * rates.cc / 100)

    total_duty = round_duty(
        bcd_amount + aidc_amount + chcess_amount + eaidc_amount +
        swc_amount + igst_amount + cc_amount
    )

    effective_rate = round(
        (total_duty / assessable_value * 100) if assessable_value > 0 else 0.0, 2
    )

    total_landed_cost = round_duty(assessable_value + total_duty)

    return DutyBreakup(
        cif_value=cif_value,
        quantity=quantity,
        landing_charges=landing_charges,
        assessable_value=assessable_value,
        bcd_rate=rates.bcd,
        bcd_amount=bcd_amount,
        aidc_rate=rates.aidc,
        aidc_amount=aidc_amount,
        chcess_rate=rates.chcess,
        chcess_amount=chcess_amount,
        eaidc_rate=rates.eaidc,
        eaidc_amount=eaidc_amount,
        customs_subtotal=customs_subtotal,
        swc_rate=rates.swc,
        swc_amount=swc_amount,
        igst_base=igst_base,
        igst_rate=rates.igst,
        igst_amount=igst_amount,
        cc_rate=rates.cc,
        cc_amount=cc_amount,
        total_duty=total_duty,
        effective_rate=effective_rate,
        total_landed_cost=total_landed_cost,
    )


def format_inr(amount: float) -> str:
    """Format amount in Indian number system (lakhs/crores)."""
    if amount >= 10_000_000:
        return f"₹{amount / 10_000_000:.2f} Cr"
    elif amount >= 100_000:
        return f"₹{amount / 100_000:.2f} L"
    elif amount >= 1_000:
        return f"₹{amount / 1_000:.2f}K"
    else:
        return f"₹{amount:.2f}"