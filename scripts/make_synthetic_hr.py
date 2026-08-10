"""Generate the single synthetic file in this project.

WHY THIS FILE EXISTS
--------------------
The RBAC demonstration needs a restricted dataset that can be *combined* with a
permitted one, so that a derived figure (revenue per employee, by department)
would leak a restricted value. Apple's 10-K states only a single company-wide
employee count, which is too coarse to show that.

Executive compensation is NOT synthetic — that comes from the real DEF 14A proxy
statement. Only the departmental headcount split is fabricated.

The department totals are scaled to approximate the company-wide figure Apple
actually reports, so the numbers are plausible, but they are invented and must
never be cited as Apple data.

Run:  python scripts/make_synthetic_hr.py
"""

from pathlib import Path

import pandas as pd

OUT = Path("data/raw/_synthetic/headcount_by_department.xlsx")

BANNER = "SYNTHETIC DATA - FABRICATED FOR RBAC DEMONSTRATION - NOT APPLE DATA"

ROWS = [
    ("Retail",                  2024,  62000),
    ("Operations",              2024,  28500),
    ("Research and Development", 2024, 31000),
    ("Sales and Marketing",     2024,  21500),
    ("Corporate Functions",     2024,  11000),
    ("General and Administrative", 2024, 10000),
    ("Retail",                  2025,  62500),
    ("Operations",              2025,  28500),
    ("Research and Development", 2025, 32500),
    ("Sales and Marketing",     2025,  21500),
    ("Corporate Functions",     2025,  11000),
    ("General and Administrative", 2025, 10000),
]

# The real company-wide counts Apple reports in its 10-K (p.8 of each filing).
# The fabricated departmental splits are scaled to land on these totals so the
# synthetic data stays consistent with the real filings it sits beside.
REAL_TOTALS = {2023: 161_000, 2024: 164_000, 2025: 166_000}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ROWS, columns=["Department", "FiscalYear", "Headcount"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        # The banner occupies row 1 so that anyone opening the file sees the
        # disclaimer before any number. The header row follows at row 2.
        pd.DataFrame([[BANNER]]).to_excel(
            writer, sheet_name="Headcount", index=False, header=False)
        df.to_excel(writer, sheet_name="Headcount", index=False, startrow=1)

    totals = df.groupby("FiscalYear")["Headcount"].sum().to_dict()
    print(f"Wrote {OUT}")
    print(f"  {len(df)} rows, totals by fiscal year: {totals}")

    # Fail loudly rather than ship synthetic data that contradicts the filings.
    for fy, total in totals.items():
        expected = REAL_TOTALS.get(fy)
        if expected is not None and total != expected:
            raise SystemExit(
                f"FY{fy} synthetic total {total:,} does not match the "
                f"{expected:,} Apple reports in its 10-K")
    print("  totals reconcile with the figures stated in the 10-K filings")


if __name__ == "__main__":
    main()
