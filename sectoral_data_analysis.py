import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = "sector_data/"

SECTOR_FILES = {
    "nifty_it.csv": "IT",
    "nifty_bank.csv": "Bank",
    "nifty_finserv.csv": "Financial Services",
    "nifty_pharma.csv": "Pharma",
    "nifty_fmcg.csv": "FMCG",
    "nifty_energy.csv": "Energy",
    "nifty_infra.csv": "Infrastructure",
    "nifty_metal.csv": "Metal",
    "nifty_realty.csv": "Realty",
}


# ============================================================
# FUNCTION: Load CSV → detect price → clean → monthly resample
# ============================================================

def load_and_resample(filepath):
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Auto-detect correct price column
    possible_cols = ['price', 'close', 'adj close', 'last']
    price_col = None

    for col in df.columns:
        if any(key in col for key in possible_cols):
            price_col = col
            break

    if price_col is None:
        raise ValueError(f"No price column found in {filepath}. Columns: {df.columns}")

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Remove commas and convert to float
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Sort by date
    df = df.sort_values("date")

    # Monthly resample — take last close each month
    df_monthly = df.resample("M", on="date").last()

    return df_monthly[[price_col]].rename(columns={price_col: os.path.basename(filepath)})


# ============================================================
# LOAD ALL SECTORS & MERGE
# ============================================================

merged = pd.DataFrame()

for file, sector_name in SECTOR_FILES.items():
    filepath = os.path.join(DATA_FOLDER, file)

    if not os.path.exists(filepath):
        print(f"WARNING: File missing → {filepath}")
        continue

    df_monthly = load_and_resample(filepath)
    df_monthly.columns = [sector_name]

    if merged.empty:
        merged = df_monthly
    else:
        merged = merged.join(df_monthly, how="outer")

# Remove rows where all values are NaN
merged.dropna(how="all", inplace=True)

print("\nMerged Dataset Preview:")
print(merged.head())



# ============================================================
# PLOT 1 — RAW SECTOR GROWTH (2000–2024)
# ============================================================

plt.figure(figsize=(14, 8))
for col in merged.columns:
    plt.plot(merged.index, merged[col], label=col)

plt.title("Sectoral Growth (2000–2024)", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Index Value (Linear)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("sector_growth_2000_2024.png", dpi=300)
plt.show()



# ============================================================
# PLOT 2 — NORMALIZED GROWTH FROM 2008 (2008 = 100)
# ============================================================

# Select only data from 2012 onwards
post_2008 = merged.loc["2012-01-01":].copy()

# Find the base (first row of 2012 period)
base = post_2008.iloc[0]

# Normalize: (value / base) * 100
normalized_2008 = (post_2008 / base) * 100

plt.figure(figsize=(14, 8))
for col in normalized_2008.columns:
    plt.plot(normalized_2008.index, normalized_2008[col], label=col)

plt.title("Sectoral Growth Relative to 2008 (Base = 100)", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Index (2008 = 100)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("sector_growth_2008_base100.png", dpi=300)
plt.show()
