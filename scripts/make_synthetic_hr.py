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
    ("Retail",                  2025,  63500),
    ("Operations",              2025,  29000),
    ("Research and Development", 2025, 33500),
    ("Sales and Marketing",     2025,  22000),
    ("Corporate Functions",     2025,  11500),
    ("General and Administrative", 2025, 10500),
]


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


if __name__ == "__main__":
    main()
