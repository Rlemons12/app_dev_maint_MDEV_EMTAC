#!/usr/bin/env python3
"""
Simple Excel File Reader Test

This script tests reading your Excel file and shows the structure
before running the full matching process.
"""

import pandas as pd
import os
from datetime import datetime

# Your Excel file path
EXCEL_FILE_PATH = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\DB_LOADSHEETS\PART PICTURE LISTING.xls"


def test_excel_reading():
    """Test reading the Excel file and show its structure."""

    print("=" * 70)
    print("EXCEL FILE READER TEST")
    print("=" * 70)
    print(f"File: {EXCEL_FILE_PATH}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check if file exists
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"❌ ERROR: File not found!")
        print(f"Expected location: {EXCEL_FILE_PATH}")
        print("\nPlease verify:")
        print("1. The file path is correct")
        print("2. The file exists at that location")
        print("3. You have permission to read the file")
        return False

    print(f"✅ File exists")

    # Get file info
    try:
        file_size = os.path.getsize(EXCEL_FILE_PATH)
        mod_time = datetime.fromtimestamp(os.path.getmtime(EXCEL_FILE_PATH))
        print(f"📁 File size: {file_size:,} bytes")
        print(f"📅 Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"⚠️ Warning: Could not get file info: {e}")

    print()

    # Try to read the Excel file
    try:
        print("📖 Reading Excel file...")

        # For .xls files, we need xlrd engine
        df = pd.read_excel(EXCEL_FILE_PATH, engine='xlrd')

        print(f"✅ Successfully loaded Excel file!")
        print(f"📊 Rows: {len(df)}")
        print(f"📊 Columns: {len(df.columns)}")

        print(f"\n📋 Column names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        # Check for our expected columns
        expected_columns = ['ITEMNUM', 'NAMEONLY']
        found_columns = []
        missing_columns = []

        for col in expected_columns:
            if col in df.columns:
                found_columns.append(col)
            else:
                missing_columns.append(col)

        if found_columns:
            print(f"\n✅ Found expected columns: {found_columns}")

        if missing_columns:
            print(f"\n❌ Missing expected columns: {missing_columns}")
            print("Available columns that might match:")
            for col in df.columns:
                if any(word in col.upper() for word in ['ITEM', 'PART', 'NUMBER', 'NUM']):
                    print(f"  - {col} (might be ITEMNUM)")
                elif any(word in col.upper() for word in ['NAME', 'ONLY', 'IMAGE', 'PICTURE']):
                    print(f"  - {col} (might be NAMEONLY)")

        # Show sample data
        print(f"\n📋 Sample data (first 5 rows):")
        print("-" * 80)

        # Show headers
        header_line = ""
        for col in df.columns:
            header_line += f"{col[:15]:15s} | "
        print(header_line)
        print("-" * 80)

        # Show data rows
        for i in range(min(5, len(df))):
            row_line = ""
            for col in df.columns:
                value = str(df.iloc[i][col]) if pd.notna(df.iloc[i][col]) else "NULL"
                row_line += f"{value[:15]:15s} | "
            print(f"{i + 1:2d}: {row_line}")

        # Data quality check
        print(f"\n🔍 Data Quality Check:")

        if 'ITEMNUM' in df.columns:
            itemnum_nulls = df['ITEMNUM'].isna().sum()
            itemnum_unique = df['ITEMNUM'].nunique()
            print(f"  ITEMNUM: {len(df)} total, {itemnum_nulls} nulls, {itemnum_unique} unique")

        if 'NAMEONLY' in df.columns:
            nameonly_nulls = df['NAMEONLY'].isna().sum()
            nameonly_unique = df['NAMEONLY'].nunique()
            print(f"  NAMEONLY: {len(df)} total, {nameonly_nulls} nulls, {nameonly_unique} unique")

        # Show some actual values
        if 'ITEMNUM' in df.columns and 'NAMEONLY' in df.columns:
            print(f"\n📋 Sample ITEMNUM -> NAMEONLY pairs:")
            clean_df = df.dropna(subset=['ITEMNUM', 'NAMEONLY'])
            for i in range(min(3, len(clean_df))):
                itemnum = clean_df.iloc[i]['ITEMNUM']
                nameonly = clean_df.iloc[i]['NAMEONLY']
                print(f"  {i + 1}. '{itemnum}' -> '{nameonly}'")

        print(f"\n✅ Excel file reading test completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ ERROR: Missing required library")
        print(f"Error: {e}")
        print("\nTo fix this, install the required library:")
        print("  pip install xlrd")
        print("  or")
        print("  pip install openpyxl")
        return False

    except Exception as e:
        print(f"❌ ERROR reading Excel file: {e}")
        print(f"Error type: {type(e).__name__}")

        # Try alternative method
        print("\n🔄 Trying alternative reading method...")
        try:
            df_alt = pd.read_excel(EXCEL_FILE_PATH, engine='openpyxl')
            print(f"✅ Alternative method worked! Loaded {len(df_alt)} rows")
            return True
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")

        print("\nTroubleshooting suggestions:")
        print("1. Install required libraries: pip install pandas xlrd openpyxl")
        print("2. Try converting the .xls file to .xlsx format")
        print("3. Check if the file is password protected")
        print("4. Verify the file is not corrupted")

        return False


def main():
    """Main function."""
    success = test_excel_reading()

    print("\n" + "=" * 70)
    if success:
        print("🎉 TEST PASSED - Ready to run the full matching script!")
        print("\nNext steps:")
        print("1. Review the column names and sample data above")
        print("2. If ITEMNUM and NAMEONLY columns look correct, proceed with matching")
        print("3. If column names are different, update the script accordingly")
    else:
        print("❌ TEST FAILED - Fix the issues above before proceeding")
        print("\nCommon solutions:")
        print("1. pip install pandas xlrd openpyxl")
        print("2. Verify file path and permissions")
        print("3. Check if file is accessible and not corrupted")

    print("=" * 70)


if __name__ == "__main__":
    main()