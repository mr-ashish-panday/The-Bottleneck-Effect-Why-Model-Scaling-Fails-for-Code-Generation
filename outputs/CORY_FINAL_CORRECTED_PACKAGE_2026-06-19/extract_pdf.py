import fitz
doc = fitz.open(r"C:\Users\Ashish\all\Downloads\Pilot AI Lead Data Sourcing.pdf")
for i in range(len(doc)):
    print(f"--- PAGE {i+1} ---")
    print(doc[i].get_text())
