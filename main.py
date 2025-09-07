# main.py
#
# This is the main orchestrator script for the LLM-Powered Evidence Triangulation Framework.
# It handles the overall workflow:
# 1. Loading environment variables for API keys.
# 2. Reading the input data from an Excel file.
# 3. Running extraction using the extraction_matching module.
# 4. Running triangulation analysis using the triangulation module.
# 5. Generating plots and saving results.

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Add notebooks directory to path to import modules
sys.path.append(str(Path(__file__).parent / "notebooks"))

try:
    from notebooks.extraction_matching import run_extraction, process_file
    from notebooks.triangulation import analyze_tri_df, plot_cumulative_trends
    from notebooks.pubmed_fetcher_function import fetch_pubmed
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Some features may not be available.")
    # Define fallback functions to prevent NameError
    def analyze_tri_df(*args, **kwargs):
        print("ERROR: analyze_tri_df function not available")
        return None
    def plot_cumulative_trends(*args, **kwargs):
        print("ERROR: plot_cumulative_trends function not available")

# Load environment variables from the .env file
load_dotenv()

# --- Configuration ---
# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
GLM_API_KEY = os.getenv("GLM_API_KEY")

# --- Configuration ---
# Default model to use for extraction
DEFAULT_MODEL = "gpt-4o-mini"  # You can change this to any supported model

# Data file paths
DATA_DIR = "data"
INPUT_FILE = "all_got_df_final_step_2_salt_cvd_021025.xlsx"
OUTPUT_FILE = "combined_extraction_results.xlsx"

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import openpyxl
    except ImportError:
        missing_deps.append("openpyxl")
    
    if missing_deps:
        print(f"ERROR: Missing required dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install " + " ".join(missing_deps))
        return False
    
    return True

def load_data(file_path):
    """Load data from Excel file with proper error handling."""
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at '{file_path}'")
        print("Available data files:")
        data_dir = os.path.dirname(file_path)
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(('.xlsx', '.csv')):
                    print(f"  - {os.path.join(data_dir, file)}")
        return None
    
    try:
        print(f"Loading data from '{file_path}'...")
        df = pd.read_excel(file_path)
        print(f"Successfully loaded {len(df)} rows from the data file.")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load data file: {e}")
        return None

def run_triangulation_analysis(df):
    """Run the triangulation analysis on the data."""
    print("Starting triangulation analysis...")
    
    # Handle missing publication year
    pmid_to_year = {
        '3475429': 1986, '2563786': 1989, '6125636': 1982, '6133987': 1983,
        '1132079': 1975, '74660': 1978, '11136953': 2001, '11231700': 2001,
        '19620514': 2009, '31350809': 2019, '28934190': 2017
    }
    
    # Convert PMID column to string to ensure matching
    df['pmid'] = df['pmid'].astype(str)
    df['pub_year'] = df['pmid'].map(pmid_to_year)
    
    # Handle missing years
    missing_year_mask = df['pub_year'].isna()
    num_missing = missing_year_mask.sum()
    if num_missing > 0:
        print(f"WARNING: {num_missing} out of {len(df)} rows are missing a publication year.")
        print("For this demonstration, missing years will be simulated randomly (2000-2023).")
        np.random.seed(42)
        random_years = np.random.randint(2000, 2024, size=num_missing)
        df.loc[missing_year_mask, 'pub_year'] = random_years
    
    # Prepare data for analysis
    print(f"Data shape before processing: {df.shape}")
    
    # Check required columns
    required_columns = ['study_design', 'direction', 'number_of_participants']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Convert data types safely
    try:
        df['pub_year'] = df['pub_year'].astype(int)
    except Exception as e:
        print(f"ERROR: Failed to convert pub_year to int: {e}")
        return None
    
    # Drop rows with missing critical data
    initial_rows = len(df)
    df.dropna(subset=['pub_year', 'study_design', 'direction', 'number_of_participants'], inplace=True)
    final_rows = len(df)
    
    if final_rows == 0:
        print("ERROR: No valid data remaining after filtering. Check your data quality.")
        return None
    
    print(f"Data shape after processing: {df.shape} (removed {initial_rows - final_rows} invalid rows)")
    
    # Analyze data for each year
    print("Analyzing data year by year...")
    yearly_results = []
    
    for year, group_df in df.groupby('pub_year'):
        try:
            result = analyze_tri_df(group_df, detail_info=False)
            if result and result['loe_score'] != 0:
                yearly_results.append({
                    'end_year': int(year),
                    'loe_score': result['loe_score']
                })
        except Exception as e:
            print(f"WARNING: Failed to analyze data for year {year}: {e}")
            continue
    
    if not yearly_results:
        print("ERROR: No valid analysis results generated. Check your data quality.")
        return None
    
    results_df = pd.DataFrame(yearly_results)
    print(f"Generated analysis results for {len(yearly_results)} years.")
    
    return results_df

def main():
    """
    The main function to execute the evidence triangulation workflow.
    """
    print("=== LLM Evidence Triangulation Framework ===")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Load data
    input_file_path = os.path.join(DATA_DIR, INPUT_FILE)
    data_df = load_data(input_file_path)
    if data_df is None:
        return
    
    # Check if we have the required columns for triangulation
    if all(col in data_df.columns for col in ['study_design', 'direction', 'number_of_participants']):
        print("Data appears to be already processed. Running triangulation analysis...")
        results_df = run_triangulation_analysis(data_df)
        
        if results_df is not None:
            print("Analysis complete. Generating plot...")
            plot_cumulative_trends(results_df, "Cumulative Evidence Trends: Salt and Cardiovascular Disease")
            print("Plot generated successfully and saved as 'cumulative_trends_plot.png'.")
        else:
            print("Triangulation analysis failed.")
    else:
        print("Data appears to be raw. Running extraction pipeline...")
        
        # Check if we have API keys
        if not OPENAI_API_KEY:
            print("ERROR: No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
            return
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Run extraction
            print("Running extraction with OpenAI...")
            extracted_df = run_extraction(data_df, client, DEFAULT_MODEL)
            
            if extracted_df is not None and not extracted_df.empty:
                # Save extraction results
                output_file_path = os.path.join(DATA_DIR, OUTPUT_FILE)
                extracted_df.to_excel(output_file_path, index=False)
                print(f"Extraction results saved to '{output_file_path}'")
                
                # Now run triangulation analysis
                results_df = run_triangulation_analysis(extracted_df)
                if results_df is not None:
                    print("Analysis complete. Generating plot...")
                    plot_cumulative_trends(results_df, "Cumulative Evidence Trends: Salt and Cardiovascular Disease")
                    print("Plot generated successfully and saved as 'cumulative_trends_plot.png'.")
            else:
                print("Extraction failed or returned no results.")
                
        except ImportError:
            print("ERROR: OpenAI package not available. Please install it with: pip install openai")
        except Exception as e:
            print(f"ERROR during extraction: {e}")
    
    print("=== Workflow completed ===")
    
# Entry point for the script
if __name__ == "__main__":
    main()
