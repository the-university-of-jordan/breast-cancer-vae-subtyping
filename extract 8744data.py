import pandas as pd
import os

# --- 1. SETUP ---
base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples'
input_path = os.path.join(base_dir, 'X_Merged_With_Labels.csv')

# --- 2. LOAD ---
print("Loading the 19,131-sample file...")
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
print("Columns available:", [c for c in df.columns if c.lower() in
      ('sampleid', 'sample_type', 'pam50call_rnaseq')])

# --- 3. LABELING RULE (your exact logic) ---
def assign_label(row):
    sid = str(row['sampleID'])
    if sid.startswith('GTEX'):
        return 'Normal_GTEX'
    elif sid.startswith('TARGET'):
        return 'pediatrics'
    elif row['sample_type'] == 'Metastatic':          # NEW — matches your original intent
        return 'Metastatic_Excluded'
    elif row['sample_type'] == 'Solid Tissue Normal':
        return 'Normal_TCGA'
    elif pd.notna(row['PAM50Call_RNAseq']) and str(row['PAM50Call_RNAseq']).strip() != '':
        return str(row['PAM50Call_RNAseq']).strip()
    else:
        return 'Unknown_Tumor'

df['Final_Label'] = df.apply(assign_label, axis=1)

print("\nLabel counts across all samples:")
print(df['Final_Label'].value_counts())

# UPDATED exclusion list — now also drops Metastatic_Excluded
df_8749 = df[~df['Final_Label'].isin(['Unknown_Tumor', 'pediatrics', 'Metastatic_Excluded'])].copy()
print(f"\nRemaining: {len(df_8749)} samples (expect 8,744 now, not 8,749)")

out_path = os.path.join(base_dir, 'X_norm_8749_real_patients.csv')
df_8749.to_csv(out_path, index=False)
print(f"✅ Saved: {out_path}")