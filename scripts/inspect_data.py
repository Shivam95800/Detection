import os
import pypdf
import pandas as pd

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"

print("--- 1. README.txt ---")
with open(os.path.join(data_dir, "README.txt"), "r", encoding="utf-8") as f:
    print(f.read())

print("\n--- 2. email-thread.txt ---")
with open(os.path.join(data_dir, "email-thread.txt"), "r", encoding="utf-8") as f:
    print(f.read())

print("\n--- 3. ops-policy.pdf ---")
reader = pypdf.PdfReader(os.path.join(data_dir, "ops-policy.pdf"))
pdf_text = ""
for i, page in enumerate(reader.pages):
    pdf_text += f"\n--- Page {i+1} ---\n" + page.extract_text()
print(pdf_text)

# Save extracted pdf text for reference
with open(r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data\ops-policy.txt", "w", encoding="utf-8") as f:
    f.write(pdf_text)

print("\n--- 4. CSV Schemas and Shapes ---")
for csv_name in ["products.csv", "partners.csv", "sample_submission.csv", "test_unlabelled.csv", "train.csv"]:
    p = os.path.join(data_dir, csv_name)
    df = pd.read_csv(p, nrows=5)
    full_df = pd.read_csv(p)
    print(f"\n>> {csv_name}: Shape = {full_df.shape}")
    print(f"Columns: {list(full_df.columns)}")
    print("Sample row:")
    print(full_df.head(1).to_dict(orient="records"))
