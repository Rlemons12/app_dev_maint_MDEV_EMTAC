import os
import pandas as pd
import glob
from pathlib import Path
import re


def merge_excel_workbooks(folder_path, output_file):
    """
    Merge multiple Excel workbooks with specific sheets into a single consolidated sheet.

    Parameters:
    - folder_path: Path to the folder containing Excel workbooks
    - output_file: Path to save the consolidated Excel file
    """
    # Create an empty list to store all dataframes
    all_data = []

    # Get all Excel files in the folder
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    excel_files.extend(glob.glob(os.path.join(folder_path, "*.xls")))

    # Filter out temporary Excel files
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]

    # Skip consolidated report files that might be in the folder
    excel_files = [f for f in excel_files if not 'consolidated_report' in os.path.basename(f).lower()]

    print(f"Found {len(excel_files)} Excel files to process.")

    # Process each file
    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        # Extract the first 5 letters as the area name
        area = file_name[:5]

        print(f"Processing file: {file_name}, Area: {area}")

        try:
            # Load the Excel file
            excel = pd.ExcelFile(file_path)

            # Get all sheet names in the workbook
            all_sheets = excel.sheet_names
            print(f"  Available sheets: {all_sheets}")

            # List to match our target day sheets - using lowercase comparison and removing spaces
            normalized_sheets = {s.lower().replace(" ", ""): s for s in all_sheets}

            # These are the sheets we want to find (with variations in casing and spacing)
            target_days = ["7days", "15days", "30days", "45days", "60days"]

            # Track sheets processed in this file
            sheets_processed = []

            # First try to process the standard day sheets
            for day_value in target_days:
                # Look for the day sheet with various formats
                possible_variations = [
                    day_value,  # e.g., "7days"
                    day_value[:1] + " " + day_value[1:],  # e.g., "7 days"
                    day_value[:2] + " " + day_value[2:],  # e.g., "15 days"
                    day_value[:1] + " Days",  # e.g., "7 Days"
                    day_value[:2] + " Days"  # e.g., "15 Days"
                ]

                sheet_found = False
                for variation in possible_variations:
                    normalized_var = variation.lower().replace(" ", "")

                    if normalized_var in normalized_sheets:
                        actual_sheet_name = normalized_sheets[normalized_var]
                        print(f"  Found sheet for {day_value}: {actual_sheet_name}")

                        try:
                            # Read the sheet into a dataframe
                            df = excel.parse(actual_sheet_name)

                            # Skip empty sheets
                            if df.empty:
                                print(f"  Sheet '{actual_sheet_name}' is empty. Skipping.")
                                continue

                            # Normalize the column names to handle variations
                            column_mapping = {}
                            for col in df.columns:
                                col_str = str(col).lower().strip()
                                if "expect" in col_str or "expec" in col_str:
                                    column_mapping[col] = "Expectations"
                                elif "comment" in col_str:
                                    column_mapping[col] = "Comments"
                                elif "attach" in col_str or "attch" in col_str:
                                    column_mapping[col] = "Attachments"

                            # Rename columns if mappings exist
                            if column_mapping:
                                df = df.rename(columns=column_mapping)

                            # Add area and days columns
                            df['Area'] = area
                            df['Days'] = f"{day_value[:1 if len(day_value) <= 5 else 2]} days"

                            # Ensure all required columns exist
                            for col in ['Expectations', 'Comments', 'Attachments']:
                                if col not in df.columns:
                                    df[col] = ''

                            # If the first row contains headers, drop it
                            if df.iloc[0].astype(str).str.contains('expect|comment|attach', case=False).any():
                                print("  First row appears to be headers, dropping it")
                                df = df.iloc[1:].reset_index(drop=True)

                            # Select only the columns we want and keep all rows
                            df = df[['Area', 'Days', 'Expectations', 'Comments', 'Attachments']]

                            # Append to our list of dataframes
                            all_data.append(df)
                            print(f"  Added {len(df)} rows from sheet '{actual_sheet_name}'")
                            sheets_processed.append(actual_sheet_name)
                            sheet_found = True
                            break
                        except Exception as e:
                            print(f"  Error processing sheet '{actual_sheet_name}': {str(e)}")

                if not sheet_found:
                    print(f"  No valid sheet found for {day_value}")

            # If no day sheets were found, try to find any sheet with instructions
            if not sheets_processed and "Instructions" in all_sheets:
                print("  No day sheets processed. Checking if instructions contain useful data.")
                # Skip processing instructions as they usually don't contain the data we want
                pass

            # As a last resort, process any remaining sheets not already processed
            # that might contain useful data (skipping Instructions)
            remaining_sheets = [s for s in all_sheets if s not in sheets_processed and s != "Instructions"]
            if not sheets_processed and remaining_sheets:
                print(f"  No specific day sheets found. Trying to extract data from other sheets: {remaining_sheets}")

                for sheet_name in remaining_sheets:
                    try:
                        df = excel.parse(sheet_name)

                        if df.empty:
                            continue

                        # Check if this sheet appears to have our expected column structure
                        if len(df.columns) >= 3:
                            # Look for column names that might match what we want
                            column_mapping = {}
                            for col_idx, col in enumerate(df.columns):
                                col_str = str(col).lower().strip()
                                if col_idx == 0 or "expect" in col_str or "expec" in col_str:
                                    column_mapping[col] = "Expectations"
                                elif col_idx == 1 or "comment" in col_str:
                                    column_mapping[col] = "Comments"
                                elif col_idx == 2 or "attach" in col_str or "attch" in col_str:
                                    column_mapping[col] = "Attachments"

                            # Rename columns if mappings exist
                            if column_mapping:
                                df = df.rename(columns=column_mapping)

                            # Add area and days columns
                            df['Area'] = area
                            df['Days'] = sheet_name  # Use sheet name as days value

                            # Ensure all required columns exist
                            for col in ['Expectations', 'Comments', 'Attachments']:
                                if col not in df.columns:
                                    df[col] = ''

                            # Also need to update the second section to keep all rows
                            # If the first row contains headers, drop it
                            if df.iloc[0].astype(str).str.contains('expect|comment|attach', case=False).any():
                                df = df.iloc[1:].reset_index(drop=True)

                            # Select only the columns we want and keep all rows
                            df = df[['Area', 'Days', 'Expectations', 'Comments', 'Attachments']]

                            # Append to our list of dataframes
                            all_data.append(df)
                            print(f"  Added {len(df)} rows from sheet '{sheet_name}'")
                        else:
                            print(f"  Sheet '{sheet_name}' does not appear to have the expected column structure")
                    except Exception as e:
                        print(f"  Error processing sheet '{sheet_name}': {str(e)}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    if not all_data:
        print("No data found in any of the Excel files.")
        return

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"Total rows in consolidated data: {len(combined_df)}")

    # Ensure output file has proper extension
    if not output_file.lower().endswith(('.xlsx', '.xls')):
        output_file = output_file + '.xlsx'
        print(f"Added missing extension to output file. New path: {output_file}")

    # Write to Excel
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        combined_df.to_excel(output_file, sheet_name='Consolidated', index=False)
        print(f"Successfully created consolidated file: {output_file}")
    except Exception as e:
        print(f"Error saving output file: {str(e)}")
        # Try saving to current directory as fallback
        fallback_file = "../../../AppData/Roaming/JetBrains/PyCharm2024.2/scratches/consolidated_report.xlsx"
        print(f"Attempting to save to current directory: {fallback_file}")
        combined_df.to_excel(fallback_file, sheet_name='Consolidated', index=False)
        print(f"Saved to fallback location: {fallback_file}")


if __name__ == "__main__":
    # Prompt user for folder path
    folder_path = input("Enter the folder path containing Excel workbooks: ").strip()

    # Verify folder exists
    while not os.path.isdir(folder_path):
        print(f"The folder '{folder_path}' does not exist or is not accessible.")
        folder_path = input("Please enter a valid folder path (or press Ctrl+C to exit): ").strip()

    # Prompt for output file
    default_output = "consolidated_report.xlsx"
    output_prompt = input(f"Enter the output file path (press Enter for default '{default_output}'): ").strip()
    output_file = output_prompt if output_prompt else default_output

    print(f"Processing Excel files from: {folder_path}")
    print(f"Output will be saved to: {output_file}")

    # Run the merge function
    merge_excel_workbooks(folder_path, output_file)