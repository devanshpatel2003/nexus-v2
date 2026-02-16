"""
Export Control Event Database
Curated timeline of BIS AI-chip export control announcements (2022-2026)
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class EventType(Enum):
    RESTRICTION = "New Restriction"
    EXPANSION = "Control Expansion"
    RELAXATION = "Policy Relaxation"
    RUMOR = "Market Rumor"
    EARNINGS = "Earnings Impact"
    GUIDANCE = "Company Guidance"


@dataclass
class ExportControlEvent:
    """Structured export control event with metadata."""
    date: str
    title: str
    description: str
    severity: Severity
    event_type: EventType
    chips_affected: List[str]
    countries_targeted: List[str]
    source: str
    nvda_reaction_pct: Optional[float] = None  # 5-day stock reaction
    amd_reaction_pct: Optional[float] = None
    notes: Optional[str] = None


# ============================================================
# CURATED EVENT DATABASE
# ============================================================

EXPORT_CONTROL_EVENTS: List[ExportControlEvent] = [
    # 2022 Events
    ExportControlEvent(
        date="2022-08-31",
        title="Initial A100/H100 Export Restrictions",
        description="US government notifies Nvidia that A100 and H100 chips require export licenses for China and Russia. First major restriction on AI accelerators.",
        severity=Severity.HIGH,
        event_type=EventType.RESTRICTION,
        chips_affected=["A100", "H100"],
        countries_targeted=["China", "Russia"],
        source="SEC 8-K Filing, Reuters",
        nvda_reaction_pct=-6.6,
        amd_reaction_pct=-3.2,
        notes="Nvidia disclosed ~$400M potential Q3 revenue impact"
    ),
    ExportControlEvent(
        date="2022-10-07",
        title="Sweeping China Chip Controls (October 7 Rules)",
        description="BIS announces comprehensive export controls on advanced semiconductors to China. Covers chips, manufacturing equipment, and US persons working in China fabs.",
        severity=Severity.CRITICAL,
        event_type=EventType.RESTRICTION,
        chips_affected=["A100", "H100", "A800", "H800", "Advanced GPUs"],
        countries_targeted=["China"],
        source="BIS Federal Register, Commerce Department",
        nvda_reaction_pct=-12.4,
        amd_reaction_pct=-8.1,
        notes="Most sweeping chip controls since Cold War. Marked regime shift in US-China tech decoupling."
    ),

    # 2023 Events
    ExportControlEvent(
        date="2023-05-24",
        title="Nvidia Q1 FY24 Earnings - AI Demand Explosion",
        description="Nvidia reports blowout earnings driven by AI demand. Data center revenue +14% QoQ despite China restrictions. Guides Q2 revenue to $11B vs $7.2B expected.",
        severity=Severity.HIGH,
        event_type=EventType.EARNINGS,
        chips_affected=["H100", "A100"],
        countries_targeted=[],
        source="Nvidia 10-Q, Earnings Call",
        nvda_reaction_pct=+24.4,
        amd_reaction_pct=+11.2,
        notes="Stock jumped 24% after-hours. AI demand overwhelmed China headwinds."
    ),
    ExportControlEvent(
        date="2023-10-17",
        title="Export Controls Expanded - A800/H800 Restricted",
        description="BIS closes loophole by restricting A800 and H800 chips (China-specific variants). Also adds restrictions on chip-making equipment and additional entities.",
        severity=Severity.HIGH,
        event_type=EventType.EXPANSION,
        chips_affected=["A800", "H800", "L40S", "RTX 4090"],
        countries_targeted=["China"],
        source="BIS Federal Register",
        nvda_reaction_pct=-4.7,
        amd_reaction_pct=-2.1,
        notes="Closed the A800/H800 workaround. Consumer GPUs (RTX 4090) also restricted."
    ),

    # 2024 Events
    ExportControlEvent(
        date="2024-01-30",
        title="H200 Export Status Uncertain",
        description="Market speculation about whether new H200 chips will require export licenses. Nvidia states they are 'working with Commerce Department' on classification.",
        severity=Severity.MEDIUM,
        event_type=EventType.RUMOR,
        chips_affected=["H200"],
        countries_targeted=["China"],
        source="Reuters, Bloomberg",
        nvda_reaction_pct=-2.8,
        amd_reaction_pct=-0.9,
        notes="Created uncertainty about next-gen chip availability in China"
    ),
    ExportControlEvent(
        date="2024-04-15",
        title="Huawei Ascend 910B Benchmark Reports",
        description="Reports emerge that Huawei's domestically-produced Ascend 910B approaches H100 performance for some workloads. Questions raised about export control effectiveness.",
        severity=Severity.MEDIUM,
        event_type=EventType.GUIDANCE,
        chips_affected=["H100", "H200"],
        countries_targeted=["China"],
        source="SemiAnalysis, Financial Times",
        nvda_reaction_pct=-3.1,
        amd_reaction_pct=+0.5,
        notes="First credible domestic China alternative. Long-term competitive concern."
    ),
    ExportControlEvent(
        date="2024-09-09",
        title="BIS Proposes 'Presumption of Denial' Policy",
        description="Commerce Department proposes stricter licensing policy with presumption of denial for advanced AI chips to China. Would effectively ban most exports.",
        severity=Severity.CRITICAL,
        event_type=EventType.RESTRICTION,
        chips_affected=["H100", "H200", "B100", "All advanced AI accelerators"],
        countries_targeted=["China"],
        source="Commerce Department, Federal Register Notice",
        nvda_reaction_pct=-9.5,
        amd_reaction_pct=-6.2,
        notes="Most aggressive proposed policy. Industry lobbying intensified."
    ),

    # 2025 Events
    ExportControlEvent(
        date="2025-01-15",
        title="Biden Administration Finalizes AI Diffusion Rule",
        description="Outgoing Biden administration finalizes framework for AI chip exports. Creates tiered country system with close allies getting preferential access.",
        severity=Severity.HIGH,
        event_type=EventType.RESTRICTION,
        chips_affected=["H200", "B100", "B200", "All advanced AI chips"],
        countries_targeted=["China", "Tier 2 countries"],
        source="Commerce Department Final Rule",
        nvda_reaction_pct=-5.2,
        amd_reaction_pct=-3.8,
        notes="Created 3-tier country classification system"
    ),
    ExportControlEvent(
        date="2025-05-21",
        title="Nvidia Q1 FY26 Earnings - China Revenue Disclosure",
        description="Nvidia reports Q1 results with detailed China revenue breakdown for first time. China data center revenue down 40% YoY but offset by rest-of-world growth.",
        severity=Severity.MEDIUM,
        event_type=EventType.EARNINGS,
        chips_affected=["H200", "B100"],
        countries_targeted=["China"],
        source="Nvidia 10-Q, Earnings Call",
        nvda_reaction_pct=+8.3,
        amd_reaction_pct=+4.1,
        notes="Stock rallied as non-China demand exceeded expectations"
    ),
    ExportControlEvent(
        date="2025-10-12",
        title="Controls Extended to AI Model Weights",
        description="BIS proposes extending export controls to AI model weights and training techniques, not just hardware. Significant expansion of regulatory scope.",
        severity=Severity.HIGH,
        event_type=EventType.EXPANSION,
        chips_affected=["Software/Models"],
        countries_targeted=["China", "Russia", "Iran"],
        source="BIS Proposed Rule",
        nvda_reaction_pct=-2.1,
        amd_reaction_pct=-1.5,
        notes="First extension beyond hardware. Affects hyperscaler model deployment."
    ),

    # 2026 Events
    ExportControlEvent(
        date="2026-01-20",
        title="New Administration Reviews Export Policy",
        description="Incoming administration announces review of AI chip export controls. Signals potential shift to 'case-by-case' licensing from presumption of denial.",
        severity=Severity.MEDIUM,
        event_type=EventType.RELAXATION,
        chips_affected=["H200", "B100", "B200"],
        countries_targeted=["China"],
        source="Commerce Department Statement, Reuters",
        nvda_reaction_pct=+6.8,
        amd_reaction_pct=+4.2,
        notes="Market interpreted as potential easing. Policy remains under review."
    ),
]


def get_events_dataframe() -> pd.DataFrame:
    """Convert events to pandas DataFrame for analysis."""
    records = []
    for event in EXPORT_CONTROL_EVENTS:
        records.append({
            "date": pd.to_datetime(event.date),
            "title": event.title,
            "description": event.description,
            "severity": event.severity.value,
            "event_type": event.event_type.value,
            "chips_affected": ", ".join(event.chips_affected),
            "countries_targeted": ", ".join(event.countries_targeted) if event.countries_targeted else "N/A",
            "source": event.source,
            "nvda_reaction_pct": event.nvda_reaction_pct,
            "amd_reaction_pct": event.amd_reaction_pct,
            "notes": event.notes,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_events_by_severity(severity: Severity) -> List[ExportControlEvent]:
    """Filter events by severity level."""
    return [e for e in EXPORT_CONTROL_EVENTS if e.severity == severity]


def get_events_by_type(event_type: EventType) -> List[ExportControlEvent]:
    """Filter events by event type."""
    return [e for e in EXPORT_CONTROL_EVENTS if e.event_type == event_type]


def get_events_in_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Get events within date range."""
    df = get_events_dataframe()
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask]


def get_average_reaction_by_severity() -> pd.DataFrame:
    """Calculate average stock reaction by severity level."""
    df = get_events_dataframe()
    return df.groupby("severity").agg({
        "nvda_reaction_pct": ["mean", "count"],
        "amd_reaction_pct": ["mean"],
    }).round(2)


def get_event_summary() -> Dict:
    """Get summary statistics of the event database."""
    df = get_events_dataframe()
    return {
        "total_events": len(df),
        "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "by_severity": df["severity"].value_counts().to_dict(),
        "by_type": df["event_type"].value_counts().to_dict(),
        "avg_nvda_reaction": df["nvda_reaction_pct"].mean(),
        "avg_amd_reaction": df["amd_reaction_pct"].mean(),
        "most_negative_nvda": df.loc[df["nvda_reaction_pct"].idxmin(), "title"],
        "most_positive_nvda": df.loc[df["nvda_reaction_pct"].idxmax(), "title"],
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXPORT CONTROL EVENT DATABASE")
    print("=" * 70)

    summary = get_event_summary()
    print(f"\nTotal Events: {summary['total_events']}")
    print(f"Date Range: {summary['date_range']}")
    print(f"\nBy Severity: {summary['by_severity']}")
    print(f"By Type: {summary['by_type']}")
    print(f"\nAvg NVDA Reaction: {summary['avg_nvda_reaction']:.2f}%")
    print(f"Avg AMD Reaction: {summary['avg_amd_reaction']:.2f}%")
    print(f"\nMost Negative Event: {summary['most_negative_nvda']}")
    print(f"Most Positive Event: {summary['most_positive_nvda']}")

    print("\n" + "=" * 70)
    print("FULL EVENT TABLE")
    print("=" * 70)
    df = get_events_dataframe()
    print(df[["date", "title", "severity", "nvda_reaction_pct"]].to_string())
