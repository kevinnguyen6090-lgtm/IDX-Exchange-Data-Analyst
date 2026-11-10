
import pandas as pd
import numpy as np
from pathlib import Path

print("IDX Exchange Week 7: Outlier Detection and Data Quality")

sold = pd.read_csv(
    "CRMLSSold2024_2026_week6_with_districts.csv",
    low_memory=False
)

listing = pd.read_csv(
    "CRMLSListing2024_2026_week6_with_districts.csv",
    low_memory=False
)

print("\nDatasets loaded successfully!")
print("Sold dataset:", sold.shape)
print("Listing dataset:", listing.shape)

numeric_fields = ["ClosePrice", "LivingArea", "DaysOnMarket"]

optional_fields = [
    "PricePerSqFt",
    "PriceRatio",
    "CloseToOriginalListRatio"
]


def flag_outliers(df, dataset_name):
    df = df.copy()
    iqr_results = []

    print(f"\nProcessing {dataset_name} dataset")

    for column in numeric_fields:
        if column not in df.columns:
            raise KeyError(f"Missing required column: {column}")

        values = pd.to_numeric(df[column], errors="coerce")
        df[column] = values

        if column in ["ClosePrice", "LivingArea"]:
            invalid = values.notna() & (values <= 0)
        else:
            invalid = values.notna() & (values < 0)

        df[f"{column}_Invalid"] = invalid

        valid_values = values[values.notna() & ~invalid]

        if valid_values.empty:
            q1 = np.nan
            q3 = np.nan
            iqr = np.nan
            lower = np.nan
            upper = np.nan
            p01 = np.nan
            p99 = np.nan
            outlier = pd.Series(False, index=df.index)
        else:
            q1 = valid_values.quantile(0.25)
            q3 = valid_values.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            p01 = valid_values.quantile(0.01)
            p99 = valid_values.quantile(0.99)

            outlier = (
                values.notna()
                & ~invalid
                & ((values < lower) | (values > upper))
            )

        df[f"{column}_Outlier"] = outlier
        df[f"{column}_Flag"] = invalid | outlier

        iqr_results.append({
            "Dataset": dataset_name,
            "Field": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "LowerBound": lower,
            "UpperBound": upper,
            "P01": p01,
            "P99": p99,
            "InvalidCount": int(invalid.sum()),
            "OutlierCount": int(outlier.sum()),
            "MissingCount": int(values.isna().sum())
        })

        print(f"\n{column}")
        print("Q1:", round(q1, 2))
        print("Q3:", round(q3, 2))
        print("IQR:", round(iqr, 2))
        print("Lower bound:", round(lower, 2))
        print("Upper bound:", round(upper, 2))
        print("1st percentile:", round(p01, 2))
        print("99th percentile:", round(p99, 2))
        print("Invalid values:", int(invalid.sum()))
        print("IQR outliers:", int(outlier.sum()))
        print("Missing values:", int(values.isna().sum()))

    flag_columns = [
        f"{column}_Flag"
        for column in numeric_fields
    ]

    df["AnyOutlierOrInvalid"] = df[flag_columns].any(axis=1)

    df["MissingClosePrice"] = df["ClosePrice"].isna()
    df["MissingLivingArea"] = df["LivingArea"].isna()
    df["MissingDaysOnMarket"] = df["DaysOnMarket"].isna()

    print("\nTotal records:", len(df))
    print("Records flagged:", int(df["AnyOutlierOrInvalid"].sum()))

    return df, pd.DataFrame(iqr_results)


sold_flagged, sold_iqr = flag_outliers(sold, "Sold")
listing_flagged, listing_iqr = flag_outliers(listing, "Listing")

iqr_summary = pd.concat(
    [sold_iqr, listing_iqr],
    ignore_index=True
)


def create_clean_dataset(df, dataset_name):
    clean = df.loc[~df["AnyOutlierOrInvalid"]].copy()

    if dataset_name == "Sold":
        clean = clean.loc[clean["ClosePrice"].notna()].copy()

    print(f"\n{dataset_name} filtering results:")
    print("Original rows:", len(df))
    print("Clean rows:", len(clean))
    print("Rows excluded:", len(df) - len(clean))

    return clean


sold_clean = create_clean_dataset(sold_flagged, "Sold")
listing_clean = create_clean_dataset(listing_flagged, "Listing")


def compare_before_after(original, clean, dataset_name):
    results = []

    for column in numeric_fields + optional_fields:
        if column not in original.columns:
            continue

        before = pd.to_numeric(original[column], errors="coerce")
        after = pd.to_numeric(clean[column], errors="coerce")

        median_before = before.median()
        median_after = after.median()

        results.append({
            "Dataset": dataset_name,
            "Field": column,
            "OriginalRows": len(original),
            "CleanRows": len(clean),
            "RowsExcluded": len(original) - len(clean),
            "RowsExcludedPct": round(
                (len(original) - len(clean)) / len(original) * 100,
                2
            ) if len(original) else 0,
            "OriginalNonMissing": int(before.notna().sum()),
            "CleanNonMissing": int(after.notna().sum()),
            "MedianBefore": median_before,
            "MedianAfter": median_after,
            "MedianChange": median_after - median_before
        })

    return pd.DataFrame(results)


sold_comparison = compare_before_after(
    sold_flagged, sold_clean, "Sold"
)

listing_comparison = compare_before_after(
    listing_flagged, listing_clean, "Listing"
)

comparison = pd.concat(
    [sold_comparison, listing_comparison],
    ignore_index=True
)

print("\nBefore and after filtering:")
print(comparison.to_string(index=False))

output_files = {
    "CRMLSSold2024_2026_week7_flagged.csv": sold_flagged,
    "CRMLSSold2024_2026_week7_clean.csv": sold_clean,
    "CRMLSListing2024_2026_week7_flagged.csv": listing_flagged,
    "CRMLSListing2024_2026_week7_clean.csv": listing_clean,
    "week7_iqr_summary.csv": iqr_summary,
    "week7_before_after_comparison.csv": comparison
}

print("\nExporting Week 7 files...")

for filename, dataframe in output_files.items():
    dataframe.to_csv(filename, index=False)
    print("Saved:", filename)

report_lines = [
    "IDX Exchange Week 7: Outlier Detection and Data Quality",
    "",
    "Methodology:",
    "The Interquartile Range (IQR) method was applied separately",
    "to the sold and listing datasets for ClosePrice,",
    "LivingArea, and DaysOnMarket.",
    "",
    "Lower bound = Q1 - 1.5 * IQR",
    "Upper bound = Q3 + 1.5 * IQR",
    "",
    "Records outside these bounds were flagged as outliers.",
    "Invalid values were identified using business rules.",
    "All original records were preserved in full flagged datasets.",
    "Separate clean datasets were created for analysis.",
    "",
    "Missing ClosePrice values were not automatically",
    "classified as outliers because unsold listings may",
    "not have a closing price.",
    "",
    "Flagged records may represent legitimate transactions.",
    "Percentile thresholds should be reviewed before",
    "making business decisions.",
    ""
]

for dataset_name, original, clean in [
    ("Sold", sold_flagged, sold_clean),
    ("Listing", listing_flagged, listing_clean)
]:
    removed = len(original) - len(clean)
    removed_pct = (
        removed / len(original) * 100
        if len(original) else 0
    )

    report_lines.extend([
        f"{dataset_name.upper()} DATASET",
        f"Original records: {len(original):,}",
        f"Clean records: {len(clean):,}",
        f"Excluded records: {removed:,}",
        f"Excluded percentage: {removed_pct:.2f}%",
        ""
    ])

    for column in numeric_fields:
        before = pd.to_numeric(
            original[column], errors="coerce"
        ).median()

        after = pd.to_numeric(
            clean[column], errors="coerce"
        ).median()

        report_lines.extend([
            f"{column}:",
            f"Median before: {before:,.2f}",
            f"Median after: {after:,.2f}",
            ""
        ])

report_lines.extend([
    "Conclusion:",
    "The clean datasets exclude records flagged by the",
    "IQR method or the specified invalid-value rules.",
    "The full flagged datasets preserve all original rows",
    "for future review and analysis.",
    "The comparison shows how filtering affects",
    "dataset sizes and median values."
])

report_path = Path("week7_data_quality_report.txt")

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nWritten report saved:", report_path)

print("\nVerifying exported files...")

for filename in list(output_files) + [str(report_path)]:
    path = Path(filename)

    if path.exists():
        print(
            f"Verified: {filename} "
            f"({path.stat().st_size / 1048576:.2f} MB)"
        )
    else:
        raise FileNotFoundError(
            f"Expected output missing: {filename}"
        )

print("\nIDX Exchange Week 7 completed successfully!")
print("Sold original:", len(sold_flagged))
print("Sold clean:", len(sold_clean))
print("Listing original:", len(listing_flagged))
print("Listing clean:", len(listing_clean))

