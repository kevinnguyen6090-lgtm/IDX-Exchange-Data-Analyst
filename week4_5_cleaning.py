import pandas as pd

# Load the enriched datasets from weeks 2-3
sold = pd.read_csv("CRMLSSold2024_2026_enriched.csv", low_memory=False)
listing = pd.read_csv("CRMLSListing2024_2026_enriched.csv", low_memory=False)

# Checking the dataset sizes
print("Sold dataset:", sold.shape)
print("Listing dataset:", listing.shape)

# Convert date columns to datetime
date_columns = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

for name, df in [("Sold", sold), ("Listing", listing)]:
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"\n{name} date column types:")
    print(df[[col for col in date_columns if col in df.columns]].dtypes)

# Identify potentially redundant columns
for name, df in [("Sold", sold), ("Listing", listing)]:
    duplicate_columns = [
        col for col in df.columns
        if col.endswith(".1") or col.endswith(".2")        
    ]

    print(f"\n{name} potentially duplicate coumns:")
    print(duplicate_columns)

# Compare potentially dupplicate listing columns
for col in listing.columns:
    if col.endswith(".1"):
        original = col[:-2]

        if original in listing.columns:
            same = listing[col].equals(listing[original])
            print(f"{col} vs {original}: Identical ={same}")

# Remove identical duplicate columns
columns_to_remove = []

for col in listing.columns:
    if col.endswith(".1"):
        original = col[:-2]

        if original in listing.columns:
            if listing[col].equals(listing[original]):
                columns_to_remove.append(col)

listing = listing.drop(columns=columns_to_remove)

print("\nRemoved duplicate columns:", columns_to_remove)
print("Listing dataset after removal:", listing.shape)

# Investigate remaining duplicate columns
remaining_columns = [
    "CloseDate.1",
    "BuyerOfficeName.1",
    "UnparsedAddress.1",
]

for col in remaining_columns:
    original = col[:-2]

    different = ~listing[col].eq(listing[original]).fillna(False)
    both_missing = listing[col].isna() & listing[original].isna()
    different = different & ~both_missing

    print(f"\n{col} vs {original}")
    print("Different rows:", different.sum())
    print("Original missing:", listing[original].isna().sum())
    print("Duplicate missing:", listing[col].isna().sum())

# Remove redundant closedate column
listing = listing.drop(columns=["CloseDate.1"])

print("\nRemoved redundant column: CloseDate.1")
print("Listing dataset after cleanup:", listing.shape)

# Check missing values in important columns
important_columns = [
    "ClosePrice",
    "ListPrice",
    "LivingArea",
    "DaysOnMarket",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "Latitude",
    "Longitude"
]

for name, df in [("Sold", sold), ("Listing", listing)]:
    print(f"\n{name} missing values:")

    for col in important_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            percentage = df[col].isna().mean() * 100
            print(f"{col}: {missing} ({percentage:.2f}%)")

# Cleaning numeric types and flag invalid values
numeric_columns = [
    "ClosePrice", "ListPrice", "LivingArea",
    "DaysOnMarket", "BedroomsTotal",
    "BathroomsTotalInteger", "Latitude", "Longitude"
]

for name, df in [("Sold", sold), ("Listing", listing)]:
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["invalid_close_price_flag"] = (
        df["ClosePrice"].notna() & (df["ClosePrice"] <= 0)
    )
    df["invalid_living_area_flag"] = (
        df["LivingArea"].notna() & (df["LivingArea"] <= 0)
    )
    df["negative_dom_flag"] = df["DaysOnMarket"] < 0
    df["negative_bedrooms_flag"] = df["BedroomsTotal"] < 0
    df["negative_bathrooms_flag"] = df["BathroomsTotalInteger"] < 0

    print(f"\n{name} invalid numeric values:")
    for col in [
        "invalid_close_price_flag",
        "invalid_living_area_flag",
        "negative_dom_flag",
        "negative_bedrooms_flag",
        "negative_bathrooms_flag"
    ]:
        print(f"{col}: {df[col].sum()}")

    print(f"{name} numeric types:")
    print(df[numeric_columns].dtypes)

    