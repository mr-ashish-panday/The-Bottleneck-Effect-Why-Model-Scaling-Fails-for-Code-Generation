import csv

with open('MSA_Worldwide_US_Franchise_STRICT_PRE_FRANCHISE_CORRECTED_FINAL_CALL_TEAM_READY_2026-06-19.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for i in [1, 4, 5]:
    r = rows[i]
    print(f"=== Row {i+2}: {r['Company']} ===")
    print(f"Why It Fits: {r['Why It Fits Client Offer']}")
    print(f"Why Now: {r['Why Now Signal']}")
    print(f"Verification: {r['Verification Status']}")
    print()
