import pandas as pd
import numpy as np

# Loading cleaned datasets from Weeks 4-5
sold = pd.read_csv("CRMLSSold2024_2026_cleaned.csv", low_memory=False)
listing = pd.read_csv("CRMLSListing2024_2026_cleaned.csv", low_memory=False)

# Checking dataset sizes
print("Sold dataset:", sold.shape)
print("Listing dataset:", listing.shape)

# Checking required price columns
price_columns = ["ClosePrice", "OriginalListPrice", "ListPrice"]

for name, df in [("Sold", sold), ("Listing", listing)]:
    print(f"\n{name} price columns:")

    for col in price_columns:
        print(f"{col}: {col in df.columns}")

# Calculate Price Ratio
for name, df in [("Sold", sold), ("Listing", listing)]:

    # Convert price columns to numeric
    df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
    df["OriginalListPrice"] = pd.to_numeric(
        df["OriginalListPrice"], errors="coerce"
    )

    # Calculate price ratio only when both prices are positive
    df["PriceRatio"] = np.where(
        (df["ClosePrice"] > 0) & (df["OriginalListPrice"] > 0),
        df["ClosePrice"] / df["OriginalListPrice"].replace(0, np.nan),
        np.nan
    )

    print(f"\n{name} Price Ratio sample:")
    print(df[["ClosePrice", "OriginalListPrice", "PriceRatio"]].head())

# Calculate Price Per Square Foot
for name, df in [("Sold", sold), ("Listing", listing)]:

    df["LivingArea"] = pd.to_numeric(
        df["LivingArea"], errors="coerce"
    )

    df["PricePerSqFt"] = np.where(
        (df["ClosePrice"] > 0) & (df["LivingArea"] > 0),
        df["ClosePrice"] / df["LivingArea"].replace(0, np.nan),
        np.nan
    )

    print(f"\n{name} Price Per Sq Ft sample:")
    print(df[["ClosePrice", "LivingArea", "PricePerSqFt"]].head())

# Days on Market
for name, df in [("Sold", sold), ("Listing", listing)]:
    df["DOM"] = pd.to_numeric(df["DaysOnMarket"], errors="coerce")
    df.loc[df["DOM"] < 0, "DOM"] = np.nan

    print(f"\n{name} Days on Market sample:")
    print(df[["DaysOnMarket", "DOM"]].head())


# Year, Month, and YrMo
date_columns = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate"
]

for name, df in [("Sold", sold), ("Listing", listing)]:
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Year"] = df["CloseDate"].dt.year.astype("Int64")
    df["Month"] = df["CloseDate"].dt.month.astype("Int64")
    df["YrMo"] = df["CloseDate"].dt.strftime("%Y-%m")

    print(f"\n{name} Year / Month / YrMo sample:")
    print(df[["CloseDate", "Year", "Month", "YrMo"]].head())

# Listing to Contract Days
for name, df in [("Sold", sold), ("Listing", listing)]:
    df["ListingToContractDays"] = (
        df["PurchaseContractDate"] - df["ListingContractDate"]
    ).dt.days

    # Invalid negative timelines are treated as missing
    df.loc[
        df["ListingToContractDays"] < 0,
        "ListingToContractDays"
    ] = np.nan

    print(f"\n{name} Listing to Contract sample:")
    print(df[
        ["ListingContractDate", "PurchaseContractDate",
         "ListingToContractDays"]
    ].head())

# Contract to Close Days
for name, df in [("Sold", sold), ("Listing", listing)]:
    df["ContractToCloseDays"] = (
        df["CloseDate"] - df["PurchaseContractDate"]
    ).dt.days

    df.loc[
        df["ContractToCloseDays"] < 0,
        "ContractToCloseDays"
    ] = np.nan

    print(f"\n{name} Contract to Close sample:")
    print(df[
        ["PurchaseContractDate", "CloseDate",
         "ContractToCloseDays"]
    ].head())

# Close to Original List Ratio
for name, df in [("Sold", sold), ("Listing", listing)]:
    df["CloseToOriginalListRatio"] = df["PriceRatio"]

    print(f"\n{name} Close to Original List Ratio sample:")
    print(df[
        ["ClosePrice", "OriginalListPrice",
         "CloseToOriginalListRatio"]
    ].head())

# Segment Analysis
segment_columns = [
    "PropertyType",
    "PropertySubType",
    "CountyOrParish",
    "MLSAreaMajor",
    "ListOfficeName",
    "BuyerOfficeName"
]

for name, df in [("Sold", sold), ("Listing", listing)]:
    for col in segment_columns:
        if col not in df.columns:
            print(f"{name}: Missing segment column {col}")
            continue

        summary = (
            df.groupby(col, dropna=False)
            .agg(
                PropertyCount=("PriceRatio", "size"),
                MedianPriceRatio=("PriceRatio", "median"),
                MedianPricePerSqFt=("PricePerSqFt", "median"),
                AverageDaysOnMarket=("DOM", "mean"),
                AverageListingToContractDays=(
                    "ListingToContractDays", "mean"
                ),
                AverageContractToCloseDays=(
                    "ContractToCloseDays", "mean"
                )
            )
            .reset_index()
        )

        print(f"\n{name} summary by {col}:")
        print(summary.head(10))

        summary.to_csv(
            f"week6_{name.lower()}_by_{col}.csv",
            index=False
        )

# Show the engineered metrics
metric_columns = [
    "PriceRatio",
    "PricePerSqFt",
    "DOM",
    "Year",
    "Month",
    "YrMo",
    "CloseToOriginalListRatio",
    "ListingToContractDays",
    "ContractToCloseDays"
]

for name, df in [("Sold", sold), ("Listing", listing)]:
    print(f"\n{name} engineered metrics sample:")
    print(df[metric_columns].head(10).to_string(index=False))

    print(f"\n{name} metric completeness:")
    print(df[metric_columns].notna().sum())

# Export engineered datasets
sold.to_csv(
    "CRMLSSold2024_2026_week6_features.csv",
    index=False
)

listing.to_csv(
    "CRMLSListing2024_2026_week6_features.csv",
    index=False
)

print("\nWeek 6 feature engineering completed successfully!")
print("Final Sold dataset:", sold.shape)
print("Final Listing dataset:", listing.shape)


# School District Mapping
import geopandas as gpd

districts = gpd.read_file(
    "school_district_data/DistrictAreas2425.shp"
)

print("\nSchool district data loaded!")
print("District shape:", districts.shape)
print("District columns:", districts.columns.tolist())
print(districts[["DistrictNa", "DistrictTy"]].head())

print("\nSold coordinate columns:")
print([c for c in sold.columns if "lat" in c.lower() or "lon" in c.lower()])

print("\nListing coordinate columns:")
print([c for c in listing.columns if "lat" in c.lower() or "lon" in c.lower()])


# Match property coordinates to California school districts

districts = districts.to_crs("EPSG:4326")

def match_school_districts(df, label):
    latitude = pd.to_numeric(df["Latitude"], errors="coerce")
    longitude = pd.to_numeric(df["Longitude"], errors="coerce")

    valid = (
        latitude.between(32, 42.1)
        & longitude.between(-125, -114)
    )

    points = gpd.GeoDataFrame(
        {"row_id": df.index[valid]},
        geometry=gpd.points_from_xy(
            longitude[valid], latitude[valid]
        ),
        crs="EPSG:4326"
    )

    matches = gpd.sjoin(
        points,
        districts[["DistrictNa", "DistrictTy", "geometry"]],
        how="left",
        predicate="intersects"
    )

    # Keep separate district types because boundaries may overlap
    matches = matches.dropna(subset=["DistrictNa"])
    matches["DistrictTy"] = matches["DistrictTy"].fillna("Unknown")

    grouped = (
        matches.groupby(["row_id", "DistrictTy"])["DistrictNa"]
        .agg(lambda names: "; ".join(sorted(set(names))))
        .unstack()
    )

    result = df.copy()

    for district_type in grouped.columns:
        column = "SchoolDistrict_" + str(district_type).replace(" ", "_")
        result[column] = result.index.map(grouped[district_type])

    print(f"\n{label} school district matching complete!")
    print("Properties with valid coordinates:", valid.sum())
    print("Properties with district matches:", matches["row_id"].nunique())
    print("District columns:", [c for c in result if c.startswith("SchoolDistrict_")])

    return result

sold = match_school_districts(sold, "Sold")
listing = match_school_districts(listing, "Listing")

sold.to_csv("CRMLSSold2024_2026_week6_with_districts.csv", index=False)
listing.to_csv("CRMLSListing2024_2026_week6_with_districts.csv", index=False)

print("\nWeek 6 school district mapping completed successfully!")


# Verify school district results

for name, df in [("Sold", sold), ("Listing", listing)]:
    district_cols = [
        col for col in df.columns
        if col.startswith("SchoolDistrict_")
    ]

    print(f"\n{name} district verification:")
    print("Total properties:", len(df))
    print("District columns:", district_cols)

    if district_cols:
        matched = df[district_cols].notna().any(axis=1)
        print("Matched properties:", matched.sum())
        print("Match rate:", round(matched.mean() * 100, 2), "%")
        print(df[district_cols].head(10).to_string(index=False))
    else:
        print("WARNING: No school district columns created.")


# Export final Week 6 datasets

sold_output = "CRMLSSold2024_2026_week6_with_districts.csv"
listing_output = "CRMLSListing2024_2026_week6_with_districts.csv"

sold.to_csv(sold_output, index=False)
listing.to_csv(listing_output, index=False)

print("\nFinal datasets exported successfully!")


# Verify exported files

from pathlib import Path

for filename in [sold_output, listing_output]:
    file_path = Path(filename)

    if file_path.exists():
        print(f"Verified: {filename}")
        print(
            "File size (MB):",
            round(file_path.stat().st_size / 1048576, 2)
        )
    else:
        print(f"WARNING: Missing {filename}")

