"""
Extended audit: check for duplicate companies, generic decision makers,
corporate phones (800/888/877/866), and other Cory-objectionable patterns.
"""
import csv, os, re, glob
from collections import Counter

DIR = os.path.dirname(os.path.abspath(__file__))
csvs = sorted(glob.glob(os.path.join(DIR, "*.csv")))

TOLL_FREE = re.compile(r'(800|888|877|866|855|844|833)[\-\.\s\)\(]')
GENERIC_DM = [
    "team", "routing", "route", "office", "center", "department",
    "contact route", "general inquiry", "support", "corporate"
]

all_companies = []

for fpath in csvs:
    fname = os.path.basename(fpath)
    if "CORY_FINAL_CORRECTION" in fname or "audit" in fname or "check_msa" in fname or "extract_pdf" in fname:
        continue
    with open(fpath, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    file_companies = []
    toll_free_rows = []
    generic_dm_rows = []
    no_website_rows = []
    
    for i, row in enumerate(rows, start=2):
        if not any(v.strip() for v in row.values()):
            continue
        
        company = row.get("Company", "").strip()
        dm = row.get("Decision Maker", "").strip()
        phone = row.get("Suggested Phone", "").strip()
        website = row.get("Website", "").strip()
        
        file_companies.append(company)
        all_companies.append((fname, company))
        
        # Toll-free phone check
        if TOLL_FREE.search(phone):
            toll_free_rows.append(f"  Row {i}: {company} -> {phone}")
        
        # Generic DM check
        dm_lower = dm.lower()
        if any(g in dm_lower for g in GENERIC_DM):
            generic_dm_rows.append(f"  Row {i}: {company} -> '{dm}'")
        
        # No website
        if not website or website in ["N/A", "n/a"]:
            no_website_rows.append(f"  Row {i}: {company}")
    
    # Check for duplicates within file
    dup_within = [c for c, count in Counter(file_companies).items() if count > 1]
    
    print(f"\n{'='*70}")
    print(f"FILE: {fname} ({len(file_companies)} rows)")
    
    if dup_within:
        print(f"  DUPLICATE COMPANIES WITHIN FILE: {dup_within}")
    
    if toll_free_rows:
        print(f"  TOLL-FREE PHONES ({len(toll_free_rows)}):")
        for t in toll_free_rows:
            print(t)
    
    if generic_dm_rows:
        print(f"  GENERIC/TEAM DM NAMES ({len(generic_dm_rows)}):")
        for g in generic_dm_rows:
            print(g)
    
    if no_website_rows:
        print(f"  MISSING WEBSITE ({len(no_website_rows)}):")
        for n in no_website_rows:
            print(n)
    
    if not dup_within and not toll_free_rows and not generic_dm_rows and not no_website_rows:
        print("  OK - Clean on extended checks.")

# Cross-file duplicate check
all_names = [c[1] for c in all_companies]
cross_dups = [c for c, count in Counter(all_names).items() if count > 1]
print(f"\n{'='*70}")
print("CROSS-FILE DUPLICATE CHECK:")
if cross_dups:
    for d in cross_dups:
        files = [c[0] for c in all_companies if c[1] == d]
        print(f"  '{d}' appears in: {files}")
else:
    print("  No cross-file duplicates found.")

print(f"\n{'='*70}")
print("EXTENDED AUDIT COMPLETE")
