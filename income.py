import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (semicolon separated)
df = pd.read_csv("wid_income.csv", sep=";")
df.columns = df.columns.str.lower().str.strip()

# Convert types
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["income_share"] = pd.to_numeric(df["income_share"], errors="coerce")

# Filter to only 1920+
df = df[df["year"] >= 1920]

# Function to extract a continuous series
def get_series(percentile_code):
    s = df[df["percentile"] == percentile_code][["year", "income_share"]]
    s = s.dropna()                 # keep only valid points (1920,1930,...)
    s = s.sort_values("year")      # ensure correct order
    return s

# Extract all three groups
top10 = get_series("p90p100")
top1 = get_series("p99p100")
bottom50 = get_series("p0p50")

# Plot
plt.figure(figsize=(14, 8))

plt.plot(top10["year"], top10["income_share"] * 100, linewidth=2, label="Top 10% income share")
plt.plot(top1["year"], top1["income_share"] * 100, linewidth=2, label="Top 1% income share")
plt.plot(bottom50["year"], bottom50["income_share"] * 100, linewidth=2, label="Bottom 50% income share")

plt.title("Income Inequality in India (1920–Present)", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Share of National Income (%)", fontsize=14)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("inequality_1920_present.png", dpi=300)
plt.show()