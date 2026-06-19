import csv, os, re, glob

DIR = os.path.dirname(os.path.abspath(__file__))
csvs = sorted(glob.glob(os.path.join(DIR, "*.csv")))

PHONE_RE = re.compile(r'\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4}')

for fpath in csvs:
    fname = os.path.basename(fpath)
    if fname.startswith("CORY_FINAL_CORRECTION") or fname == "audit_phone.py":
        continue
    with open(fpath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total = 0
    missing_phone = []
    weak_phone = []
    missing_dm = []
    
    for i, row in enumerate(rows, start=2):
        # skip blank rows
        if not any(v.strip() for v in row.values()):
            continue
        total += 1
        
        phone = row.get("Suggested Phone", "").strip()
        dm = row.get("Decision Maker", "").strip()
        company = row.get("Company", "").strip()
        
        if not phone:
            missing_phone.append(f"  Row {i}: {company}")
        elif not PHONE_RE.search(phone):
            weak_phone.append(f"  Row {i}: {company} -> '{phone}'")
        
        if not dm or dm.lower() in ["n/a", "unknown", ""]:
            missing_dm.append(f"  Row {i}: {company}")
    
    print(f"\n{'='*70}")
    print(f"FILE: {fname}")
    print(f"Total data rows: {total}")
    print(f"Missing Suggested Phone: {len(missing_phone)}")
    if missing_phone:
        for m in missing_phone:
            print(m)
    print(f"No numeric phone in Suggested Phone: {len(weak_phone)}")
    if weak_phone:
        for w in weak_phone:
            print(w)
    print(f"Missing Decision Maker: {len(missing_dm)}")
    if missing_dm:
        for d in missing_dm:
            print(d)
