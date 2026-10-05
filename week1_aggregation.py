import pandas as pd
import glob
import os

#Find all monthly sold and listing CSV files
sold_files = glob.glob("CRMLSSold202[4-6][0-9][0-9].csv")
listing_files = glob.glob("CRMLSListing202[4-6][0-9][0-9].csv")

# Check how many files were found
print("Sold files found:", len(sold_files))
print("Listing files found:", len(listing_files))

# Read and combine all sold CSV files
sold_dataframes = [pd.read_csv(file, low_memory=False) for file in sold_files]
sold_combined = pd.concat(sold_dataframes, ignore_index=True)

# Read and combine all listing CSV files
listing_dataframes= [pd.read_csv(file, low_memory=False) for file in listing_files]
listing_combined = pd.concat(listing_dataframes, ignore_index=True)

#Show row counts after concatenation
print("Combined sold rows:", len(sold_combined))
print("Combined listing rows:", len(listing_combined))

#Filter both datasets to Residental properties only
sold_residential = sold_combined[sold_combined["PropertyType"] == "Residential"]
listing_residential = listing_combined[listing_combined["PropertyType"] == "Residential"]


# Show row counts before and after filtering
print("Sold rows before filter:", len(sold_combined))
print("Sold rows after Residential filter:", len(sold_residential))

print("Listing rows before filter:", len(listing_combined))
print("Listing rows after Residential filter:", len(listing_residential))

#Save final Residential datasets
sold_residential.to_csv("CRMLSSold2024_2026.csv", index=False)
listing_residential.to_csv("CRMLSListing2024_2026.csv", index=False)

print("\nData aggregation and Residential property filtering completed successfully.")
