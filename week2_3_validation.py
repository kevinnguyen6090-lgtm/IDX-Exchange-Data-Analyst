
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Configuration and data loading
PLOT_DIR = Path("eda_plots")
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
]

sold_files = sorted(Path(".").glob("CRMLSSold202[4-6][0-9][0-9].csv"))
if not sold_files:
    raise FileNotFoundError("No original monthly Sold CSV files were found.")

# Original, unfiltered Sold files are used only for property-type distribution.
sold_combined = pd.concat(
    [pd.read_csv(path, low_memory=False) for path in sold_files],
    ignore_index=True,
)
sold = pd.read_csv("CRMLSSold2024_2026.csv", low_memory=False)
listing = pd.read_csv("CRMLSListing2024_2026.csv", low_memory=False)

datasets = {"Sold": sold, "Listing": listing}

for name, df in datasets.items():
    print(f"\n{name} dataset shape: {df.shape}")
    print(f"\n{name} first five rows:\n{df.head()}")
    print(f"\n{name} columns:\n{df.columns.tolist()}")
    print(f"\n{name} data types:\n{df.dtypes}")



# Missing values and validation
missing_percentages = {}
for name, df in datasets.items():
    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_percentages[name] = missing_pct

    print(f"\n{name} missing counts:\n{missing_counts}")
    print(f"\n{name} missing percentages:\n{missing_pct}")
    print(f"\n{name} columns over 90% missing:\n{missing_pct[missing_pct > 90]}")
    print(f"\n{name} unique property types: {df['PropertyType'].unique()}")
    print(f"{name} duplicate rows: {df.duplicated().sum()}")
    print(f"{name} duplicate Listing IDs: {df['ListingId'].duplicated().sum()}")

missing_report = pd.DataFrame({
    "Sold_Missing_Percent": missing_percentages["Sold"],
    "Listing_Missing_Percent": missing_percentages["Listing"],
})
missing_report.to_csv("missing_value_report.csv")
print("\nMissing value report saved successfully!")

for name, df in datasets.items():
    date_column = "CloseDate" if name == "Sold" else "ListingContractDate"
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    print(f"\n{name} {date_column} range: {df[date_column].min()} to {df[date_column].max()}")

# Numeric distributions, outliers, and saved plots
# Display Sold price distribution
plt.hist(sold[sold["ClosePrice"] <= 10000000]["ClosePrice"].dropna(), bins=50)
plt.title("Distribution of Sold Prices")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.show()

# Display Listing price distribution
plt.hist(listing[listing["ListPrice"] <= 10000000]["ListPrice"].dropna(), bins=50)
plt.title("Distribution of Listing Prices")
plt.xlabel("List Price")
plt.ylabel("Frequency")
plt.show()

for name, df in datasets.items():
    print(f"\n{name} numeric summary:\n{df[NUMERIC_FIELDS].describe()}")

    price_field = "ClosePrice" if name == "Sold" else "ListPrice"
    print(f"\n{name} 10 highest {price_field} values:")
    print(df.nlargest(10, price_field)[[price_field, "LivingArea", "DaysOnMarket"]])
    print(f"\n{name} 10 lowest {price_field} values:")
    print(df.nsmallest(10, price_field)[[price_field, "LivingArea", "DaysOnMarket"]])
    print(f"{name} prices under $10,000: {(df[price_field] < 10_000).sum()}")
    print(f"{name} prices over $10 million: {(df[price_field] > 10_000_000).sum()}")

    for column in NUMERIC_FIELDS:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=50)
        ax.set(title=f"{name} - {column} Distribution", xlabel=column, ylabel="Frequency")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{name}_{column}_histogram.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(values)
        ax.set(title=f"{name} - {column} Boxplot", ylabel=column)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{name}_{column}_boxplot.png")
        plt.close(fig)

print("\nEDA plots saved locally.")

# Market analysis and data-quality checks
print("\nAverage Sold price:", sold["ClosePrice"].mean())
print("Median Sold price:", sold["ClosePrice"].median())

valid_sales = sold.loc[
    sold["ClosePrice"].notna()
    & sold["ListPrice"].notna()
    & (sold["ListPrice"] > 0)
]
for label, condition in {
    "above": valid_sales["ClosePrice"] > valid_sales["ListPrice"],
    "below": valid_sales["ClosePrice"] < valid_sales["ListPrice"],
    "at": valid_sales["ClosePrice"] == valid_sales["ListPrice"],
}.items():
    print(f"Homes sold {label} list price: {condition.mean() * 100:.2f}%")

print("\nSold Days on Market summary:\n", sold["DaysOnMarket"].describe())
print("Median Days on Market:", sold["DaysOnMarket"].median())
print("Average Days on Market:", sold["DaysOnMarket"].mean())

sold["ListingContractDate"] = pd.to_datetime(
    sold["ListingContractDate"], errors="coerce"
)
invalid_dates = sold["CloseDate"] < sold["ListingContractDate"]
print("\nHomes with closing dates before listing dates:", invalid_dates.sum())

county_prices = (
    sold.groupby("CountyOrParish")["ClosePrice"]
    .median()
    .sort_values(ascending=False)
)
print("\nTop 10 counties by median Sold price:\n", county_prices.head(10))

property_counts = sold_combined["PropertyType"].value_counts()
property_percentages = (
    sold_combined["PropertyType"].value_counts(normalize=True) * 100
)
print("\nOriginal Sold property types:\n", property_counts)
print("\nOriginal Sold property-type percentages:\n", property_percentages.round(2))

print("\nCore numeric market analysis fields:\n", NUMERIC_FIELDS)
print("\nOther fields (including additional market fields and metadata):\n",
      [col for col in sold.columns if col not in NUMERIC_FIELDS])

# FRED 30-year mortgage rate enrichment
fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(fred_url)
mortgage["observation_date"] = pd.to_datetime(mortgage["observation_date"])
mortgage["year_month"] = mortgage["observation_date"].dt.to_period("M")
monthly_mortgage = (
    mortgage.groupby("year_month")["MORTGAGE30US"].mean().reset_index()
)
print("\nLatest monthly mortgage rate averages:\n", monthly_mortgage.tail(12))

sold["year_month"] = sold["CloseDate"].dt.to_period("M")
listing["year_month"] = listing["ListingContractDate"].dt.to_period("M")

sold_enriched = sold.merge(monthly_mortgage, on="year_month", how="left", validate="many_to_one")
listing_enriched = listing.merge(monthly_mortgage, on="year_month", how="left", validate="many_to_one")

for name, df in {"Sold": sold_enriched, "Listing": listing_enriched}.items():
    missing_rates = df["MORTGAGE30US"].isna().sum()
    print(f"\n{name} enriched shape: {df.shape}")
    print(f"{name} missing mortgage rates: {missing_rates}")
    if missing_rates:
        raise ValueError(f"{name} has {missing_rates} missing mortgage rates; review dates before export.")

sold_enriched.to_csv("CRMLSSold2024_2026_enriched.csv", index=False)
listing_enriched.to_csv("CRMLSListing2024_2026_enriched.csv", index=False)
print("\nEnriched datasets saved successfully!")
print("Data validation and exploratory analysis completed successfully.")