#!/usr/bin/env python3
"""
Read filenames from stdin and convert any .parquet files to CSV format.
Output to stdout.

Usage:
    echo "file1.parquet file2.csv file3.parquet" | python cat_parquet.py
    find . -name "*.parquet" | python cat_parquet.py
    ls *.parquet | python cat_parquet.py
"""

import sys
import pandas as pd
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')


def convert_parquet_to_csv(filename):
    """
    Read a parquet file and return its CSV representation.
    
    Args:
        filename: Path to the parquet file
        
    Returns:
        String containing CSV data or error message
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(filename)
        
        # Convert to CSV string
        csv_string = df.to_csv(index=True)
        
        return csv_string
    except FileNotFoundError:
        return f"# Error: File '{filename}' not found\n"
    except Exception as e:
        return f"# Error reading '{filename}': {str(e)}\n"


def main():
    """
    Main function to process filenames from stdin or command line args.
    """
    # Check if there are command line arguments
    if len(sys.argv) > 1:
        # Use command line arguments as filenames
        filenames = sys.argv[1:]
    else:
        # Read all input from stdin
        input_text = sys.stdin.read()
        
        # Split by whitespace and newlines to get individual filenames
        filenames = input_text.split()
    
    if not filenames:
        print("# No filenames provided", file=sys.stderr)
        print("# Usage: cat_parquet.py [file1.parquet file2.parquet ...]", file=sys.stderr)
        print("#    or: echo 'file1.parquet' | cat_parquet.py", file=sys.stderr)
        print("#    or: find . -name '*.parquet' | cat_parquet.py", file=sys.stderr)
        sys.exit(1)
    
    # Process each filename
    for filename in filenames:
        # Strip any extra whitespace
        filename = filename.strip()
        
        # Skip empty strings
        if not filename:
            continue
            
        # Check if it's a parquet file
        if filename.endswith('.parquet'):
            # Add header comment to identify the file
            print(f"# File: {filename}")
            
            # Convert and print the CSV
            csv_output = convert_parquet_to_csv(filename)
            print(csv_output)
        else:
            # For non-parquet files, just note them
            print(f"# Skipping non-parquet file: {filename}", file=sys.stderr)


if __name__ == "__main__":
    main()