import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

os.makedirs("eda_plots", exist_ok=True)

# Load the original sold files before residential filtering
sold_files=glob.glob("CRMLSSold202[4-6][0-9][0-9].csv")
sold_combined=pd.concat(
    [pd.read_csv(file, low_memory=False) for file in sold_files],
    ignore_index=True
)

# Load the final residential datasets from week 1
sold = pd.read_csv("CRMLSSold2024_2026.csv")
listing = pd.read_csv("CRMLSListing2024_2026.csv")

# Check the size of each dataset
print("Sold dataset shape:", sold.shape)
print("Listing dataset:", listing.shape)

# Show the first 5 rows
print("\nFirst 5 rows of Sold data:")
print(sold.head())

print("\nFirst 5 rows of Listing data:")
print(listing.head())

# Check column names
print("\nSold columns:")
print(sold.columns.tolist())

print("\nListing columns:")
print(listing.columns.tolist())

# Check missing values
print("\nMissing values in Sold data:")
print(sold.isnull().sum().sort_values(ascending=False))

print("\nMissing values in Listing data:")
print(listing.isnull().sum().sort_values(ascending=False))

# Calculate missing vlaue percentages
sold_missing_percent = (sold.isnull().sum() / len(sold)) * 100
listing_missing_percent = (listing.isnull().sum() / len(listing)) * 100

print("\nSold missing percentages:")
print(sold_missing_percent.sort_values(ascending=False))

print("\nListing missing percentages:")
print(listing_missing_percent.sort_values(ascending=False))

# Find columns with more than 90% missing values
sold_over_90 = sold_missing_percent[sold_missing_percent > 90]
listing_over_90 = listing_missing_percent[listing_missing_percent > 90]

print("\nSold columns over 90% missing: ")
print(sold_over_90)

print("\nListing columns over 90% missing:")
print(listing_over_90)

# Summary statistics for important numeric columns
numeric_columns = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "DaysOnMarket",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "YearBuilt"
]

print("\nSold numeric summary:")
print(sold[numeric_columns].describe())

print("\nListing numeric summary:")
print(listing[numeric_columns].describe())

# Show all columns in the Terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Check extreme values
print("\nTop 10 highest Sold prices:")
print(sold.nlargest(10, "ClosePrice")[["ClosePrice", "ListPrice", "LivingArea"]])

print("\nTop 10 highest Listing prices:")
print(listing.nlargest(10, "ListPrice")[["ListPrice", "LivingArea", "DaysOnMarket"]])

# Check lowest values
print("\n10 lowest Sold prices:")
print(sold.nsmallest(10, "ClosePrice")[["ClosePrice", "ListPrice", "LivingArea"]])

print("\n10 lowest living prices:")
print(listing.nsmallest(10, "ListPrice")[["ListPrice", "LivingArea", "DaysOnMarket"]])

# Count suspicious price values
sold_low_price = sold[sold["ClosePrice"] < 10000]
sold_high_price = sold[sold["ClosePrice"] > 10000000]

listing_low_price = listing[listing["ListPrice"] < 10000]
listing_high_price = listing[listing["ListPrice"] > 10000000]

print("\nSold prices under $10,000:", len(sold_low_price))
print("Sold prices over $10 million:", len(sold_high_price))

print("\nListing prices under $10,000:", len(listing_low_price))
print("Listing prices over $10 million:", len(listing_high_price))

# Save missing value report
missing_report = pd.DataFrame( {
    "Sold_Missing_Percent": sold_missing_percent,
    "Listing_Missing_Percent": listing_missing_percent
})

missing_report.to_csv("missing_value_report.csv")

print("\nMissing value report saved successfully!")

# Check unique property types
print("\nUnique property types in Sold data:")
print(sold["PropertyType"].unique())

print("\nUnique property types in Listing data:")
print(listing["PropertyType"].unique())

# Check for duplicate records
print("\nDuplicate rows in Sold data:", sold.duplicated().sum())
print("Duplicate rows in Listing data:", listing.duplicated().sum())

# Check duplicate Listing IDs
print("\nDuplicate Listing IDs in Sold data:", sold["ListingId"].duplicated().sum())
print("Duplicate Listing IDs in Listing Data:", listing["ListingId"].duplicated().sum())

# Check date ranges
print("\nSold date range:")
print(sold["CloseDate"].min(), "to", sold["CloseDate"].max())

print("\nListing date range:")
print(listing["ListingContractDate"].min(), "to", listing["ListingContractDate"].max())

# Create histogram of Sold prices
plt.hist(sold[sold["ClosePrice"] <=10000000]["ClosePrice"].dropna(), bins=50)
plt.title("Distribution of Sold Prices")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.show()

# Create histogram of Listing prices
plt.hist(listing[listing["ListPrice"] <=10000000]["ListPrice"].dropna(), bins=50)
plt.title("Distribution of Listing Prices")
plt.xlabel("List Price")
plt.ylabel("Frequency")
plt.show()

# Numeric fields required for Weeks 2-3 EDA
numeric_fields = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt"
]

# Numeric summaries for Sold and Listing datasets
print("\nSold numeric summaries:")
print(sold[numeric_fields].describe())

print("\nLisitng numeric summaries:")
print(listing[numeric_fields].describe())

# Generate histograms for all numeric fields
for column in numeric_fields:
    for name, df in [("Sold", sold), ("Listing", listing)]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=50)
        plt.title(f"{name} - {column} Distrbution")
        plt.xlabel(column)
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(f"eda_plots/{name}_{column}_histogram.png")
        plt.close()

# Generate boxplots for all numeric fields
for column in numeric_fields:
    for name, df in [("Sold", sold), ("Listing", listing)]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        plt.figure(figsize=(8, 5))
        plt.boxplot(values)
        plt.title(f"{name} - {column} Boxplot")
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(f"eda_plots/{name}_{column}_boxplot.png")
        plt.close()

# Average and median sold prices
print("\nAverage Sold price:")
print(sold["ClosePrice"].mean())

print("\nMedian Sold Price:")
print(sold["ClosePrice"].median())

# Percentage of homes sold above, below, and at list price
valid_sales = sold[
    (sold["ClosePrice"].notna()) &
    (sold["ListPrice"].notna()) &
    (sold["ListPrice"] > 0)
]

above_list = (valid_sales["ClosePrice"] > valid_sales["ListPrice"]).mean() * 100
below_list = (valid_sales["ClosePrice"] < valid_sales["ListPrice"]).mean() * 100
at_list = (valid_sales["ClosePrice"] == valid_sales["ListPrice"]).mean() * 100

print("\nHomes sold above list price:", round(above_list, 2), "%")
print("Homes sold below list price:", round(below_list, 2), "%")
print("Homes sold at list price:", round(at_list, 2), "%")

# Days on market analysis
print("\nDays on Market Summary")
print(sold["DaysOnMarket"].describe())

print("\nMedian Days on Market:")
print(sold["DaysOnMarket"].median())

print("\nAverage Days on Market:")
print(sold["DaysOnMarket"].mean())

# Check for date inconsistencies
sold["CloseDate"] = pd.to_datetime(sold["CloseDate"], errors="coerce")
sold["ListingContractDate"] = pd.to_datetime(
    sold["ListingContractDate"], errors="coerce"
)

invalid_dates= sold[
    sold["CloseDate"] < sold["ListingContractDate"]
]

print("\nHomes with closing dates before listing dates:", len(invalid_dates))

# Finding counties with the highest median sold prices
county_prices= (
    sold.groupby("CountyOrParish")["ClosePrice"]
    .median()
    .sort_values(ascending=False)
)

print("\nTop 10 counties by median sold price:")
print(county_prices.head(10))

# Property type distribution before filtering
property_counts = sold_combined["PropertyType"].value_counts()
property_percentages = sold_combined["PropertyType"].value_counts(normalize=True) *100

print("\nProperty types found:")
print(property_counts)

print("\nProperty type percentages:")
print(property_percentages.round(2))

# Seperate market analysis fields from meta data
market_fields=[
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt"
]

metadata_fields= [
    col for col in sold.columns if col not in market_fields
]

print("\nMarket analysis fields:")
print(market_fields)

print("\nOther fields (including metadata):")
print(metadata_fields)

# Mortgage rate enrichment
# Download historical 30-year fixed mortgage rates from FRED

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

mortgage = pd.read_csv(url)

print("\nMortgage rate data:")
print(mortgage.head())
print("\nMortgage rate columns:")
print(mortgage.columns.tolist())

# Convert mortgage dates to datetime
mortgage["observation_date"] = pd.to_datetime(
    mortgage["observation_date"]
)

# Create a year-month column
mortgage["year_month"] = mortgage["observation_date"].dt.to_period("M")

# Calculate average mortgage rate for each month
monthly_mortgage = (
    mortgage.groupby("year_month")["MORTGAGE30US"]
    .mean()
    .reset_index()
)

print("\nMonthly mortgage rate averages:")
print(monthly_mortgage.tail(12))

# Create year-month columns for matching mortgage rates
sold["year_month"] = sold["CloseDate"].dt.to_period("M")

listing["ListingContractDate"] = pd.to_datetime(
    listing["ListingContractDate"], errors="coerce"
)

listing["year_month"] = listing["ListingContractDate"].dt.to_period("M")

print("\nSold year-month:")
print(sold["year_month"].head())

print("\nListing year-month:")
print(listing["year_month"].head())

# Merge monthly mortgage rates into both datasets
sold_enriched = sold.merge(
    monthly_mortgage,
    on="year_month",
    how="left",
    validate="many_to_one"
)

listing_enriched = listing.merge(
    monthly_mortgage,
    on="year_month",
    how="left",
    validate="many_to_one"
)

print("\nSold enriched shape:", sold_enriched.shape)
print("Listing enriched shape:", listing_enriched.shape)

print("\nSold mortgage rates missing:",
      sold_enriched["MORTGAGE30US"].isna().sum())

print("Listing mortgage rates missing:",
      listing_enriched["MORTGAGE30US"].isna().sum())

# Save datasets with monthly mortgage rates
sold_enriched.to_csv(
    "CRMLSSold2024_2026_enriched.csv",
    index=False
)

listing_enriched.to_csv(
    "CRMLSListing2024_2026_enriched.csv",
    index=False
)

print("\nEnriched datasets saved successfully!")

print("\nData validation and exploratory analysis completed successfully.")