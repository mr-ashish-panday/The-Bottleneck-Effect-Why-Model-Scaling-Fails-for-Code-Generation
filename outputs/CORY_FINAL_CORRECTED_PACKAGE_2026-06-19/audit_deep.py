import csv, os, re, glob

DIR = os.path.dirname(os.path.abspath(__file__))
csvs = sorted(glob.glob(os.path.join(DIR, "*.csv")))

# Match any 10-digit US phone in various formats
PHONE_RE = re.compile(r'[\d\(\+][\d\s\-\.\(\)]{7,}')

for fpath in csvs:
    fname = os.path.basename(fpath)
    if fname.startswith("CORY_FINAL_CORRECTION") or fname.endswith("audit_phone.py") or fname == "audit_deep.py":
        continue
    with open(fpath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    
    total = 0
    issues = []
    
    for i, row in enumerate(rows, start=2):
        if not any(v.strip() for v in row.values()):
            continue
        total += 1
        
        phone = row.get("Suggested Phone", "").strip()
        dm = row.get("Decision Maker", "").strip()
        company = row.get("Company", "").strip()
        title = row.get("Title", "").strip()
        why_fits = row.get("Why It Fits Client Offer", "").strip()
        verification = row.get("Verification Status", "").strip()
        location = row.get("Location", "").strip()
        geo_fit = row.get("Geography Fit", "").strip()
        
        row_issues = []
        
        # 1. Phone check
        if not phone:
            row_issues.append("CRITICAL: No Suggested Phone")
        elif not PHONE_RE.search(phone):
            row_issues.append(f"CRITICAL: No numeric phone digits found: '{phone}'")
        
        # 2. Decision Maker check
        if not dm or dm.lower() in ["n/a", "unknown", "tbd"]:
            row_issues.append("WARNING: Missing/generic Decision Maker name")
        
        # 3. Empty "Why It Fits" 
        if not why_fits:
            row_issues.append("WARNING: Empty 'Why It Fits Client Offer'")
            
        # 4. Empty Verification Status
        if not verification:
            row_issues.append("WARNING: Empty Verification Status")

        # MSA special: check for "already franchised" or "mature franchise" language
        if "MSA" in fname:
            combined = (why_fits + " " + verification + " " + row.get("Why Now Signal", "")).lower()
            if "mature franchise" in combined and "not a mature" not in combined:
                row_issues.append("RED: MSA row mentions mature franchise without negation")
            if any(w in combined for w in ["already franchised", "established franchise system", "franchise network operating"]):
                row_issues.append("RED: MSA row signals already-franchised company")
        
        # Absolute Vending special: corporate or existing vending
        if "Absolute_Vending" in fname:
            combined = (why_fits + " " + verification + " " + row.get("Why Now Signal", "")).lower()
            if "corporate" in combined and "non-corporate" not in combined:
                row_issues.append("RED: Vending row signals corporate location")
            if "already has vending" in combined or "existing vending machine" in combined:
                row_issues.append("RED: Vending row signals existing vending machine")
        
        if row_issues:
            issues.append((i, company, row_issues))
    
    print(f"\n{'='*70}")
    print(f"FILE: {fname}")
    print(f"Total data rows: {total}")
    print(f"Has 'Suggested Phone' column: {'Suggested Phone' in fields}")
    print(f"Issues found: {len(issues)}")
    for rownum, company, problems in issues:
        for p in problems:
            print(f"  Row {rownum} [{company}]: {p}")
    if not issues:
        print("  OK - No issues detected.")

print(f"\n{'='*70}")
print("SUMMARY COMPLETE")
