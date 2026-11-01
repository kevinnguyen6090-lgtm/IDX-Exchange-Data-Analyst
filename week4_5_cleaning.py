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

# Check date consistency
for name, df in [("Sold", sold), ("Listing", listing)]:

    # Flag properties listed after their closing date
    df["listing_after_close_flag"] = (
        df["ListingContractDate"] > df["CloseDate"]
    )

    # Flag purchase contracts dated after closing
    df["purchase_after_close_flag"] = (
        df["PurchaseContractDate"] > df["CloseDate"]
    )

    # Flag listings dated after the purchase contract
    df["negative_timeline_flag"] = (
        df["ListingContractDate"] > df["PurchaseContractDate"]
    )

    print(f"\n{name} date consistency checks:")
    print("Listing after close:", df["listing_after_close_flag"].sum())
    print("Purchase after close:", df["purchase_after_close_flag"].sum())
    print("Listing after purchase:", df["negative_timeline_flag"].sum())

# Geographic data validation
for name, df in [("Sold", sold), ("Listing", listing)]:

    # Flag missing coordinates
    df["missing_coordinates_flag"] = (
        df["Latitude"].isna() | df["Longitude"].isna()
    )

    # Flag coordinates equal to zero
    df["zero_coordinates_flag"] = (
        (df["Latitude"] == 0) | (df["Longitude"] == 0)
    )

    # Flag positive longitude
    df["positive_longitude_flag"] = df["Longitude"] > 0

    # Flag coordinates outside California's approximate boundaries
    df["invalid_location_flag"] = (
        df["Latitude"].notna() &
        df["Longitude"].notna() &
        (
            ~df["Latitude"].between(32, 42) |
            ~df["Longitude"].between(-125, -114)
        )
    )

    print(f"\n{name} geographic validation:")
    print("Missing coordinates:", df["missing_coordinates_flag"].sum())
    print("Zero coordinates:", df["zero_coordinates_flag"].sum())
    print("Positive longitude:", df["positive_longitude_flag"].sum())
    print("Outside California boundaries:", df["invalid_location_flag"].sum())

# Export cleaned datasets
sold.to_csv("CRMLSSold2024_2026_cleaned.csv", index=False)
listing.to_csv("CRMLSListing2024_2026_cleaned.csv", index=False)

print("\nCleaned datasets saved successfully.")
print("Final Sold dataset:", sold.shape)
print("Final Listing dataset:", listing.shape)

# Verifying exported datasets
sold_check = pd.read_csv("CRMLSSold2024_2026_cleaned.csv", low_memory=False)
listing_check = pd.read_csv("CRMLSListing2024_2026_cleaned.csv", low_memory=False)

print("\nExport verification:")
print("Sold matches:", sold_check.shape == sold.shape)
print("Listing matches:", listing_check.shape == listing.shape)

# Before and after cleaning summary

print("\n=== DATA CLEANING SUMMARY ===")

print("\nSold Dataset:")
print("Original rows: 494521")
print("Final rows:", len(sold))
print("Original columns: 86")
print("Final columns:", len(sold.columns))

print("\nListing Dataset:")
print("Original rows: 650390")
print("Final rows:", len(listing))
print("Original columns: 86")
print("Final columns:", len(listing.columns))

print("\nCleaning process completed.")

