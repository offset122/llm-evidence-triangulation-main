import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys

__all__ = ['analyze_tri_df', 'plot_cumulative_trends']


def analyze_tri_df(input_df, detail_info):
    """
    Analyzes a DataFrame to determine the probabilities of different relationships
    and a custom "Level of Evidence" (LOE) score.

    The function weights non-MR studies (RCT and OS) by the number of participants
    before calculating the probabilities, while MR studies are not weighted.
    This is a custom analytical approach.

    Args:
        input_df (pd.DataFrame): The input DataFrame containing study data.
        detail_info (bool): If True, prints a detailed summary of the grouped data.

    Returns:
        dict: A dictionary with the calculated probabilities, LOE, and the
              biggest relationship type.
    """
    # Separate MR from non-MR studies
    df_non_mr = input_df[input_df['study_design'] != 'MR']
    df_mr = input_df[input_df['study_design'] == 'MR']

    # Filter non-MR studies based on exposure and outcome directions
    df_non_mr = df_non_mr[df_non_mr['exposure_direction'].isin(['increased', 'decreased'])]
    df_non_mr = df_non_mr[df_non_mr['direction'].isin(['increase', 'decrease', 'no_change'])]

    # Filter non-MR group by participant count, removing outliers
    lower_bound = df_non_mr['number_of_participants'].quantile(0.05)
    upper_bound = df_non_mr['number_of_participants'].quantile(0.95)
    df_non_mr = df_non_mr[
        (df_non_mr['number_of_participants'] >= lower_bound) &
        (df_non_mr['number_of_participants'] <= upper_bound)
        ]

    # Assign weights and calculate probabilities for non-MR studies
    total_participants = df_non_mr['number_of_participants'].sum()
    if total_participants > 0:
        df_non_mr['weight'] = df_non_mr['number_of_participants'] / total_participants
    else:
        df_non_mr['weight'] = 0

    non_mr_probs = df_non_mr.groupby('direction')['weight'].sum()

    # Calculate probabilities for MR studies
    mr_probs = df_mr['direction'].value_counts(normalize=True)

    # Combine results and calculate LOE score
    combined_probs = (non_mr_probs.fillna(0) + mr_probs.fillna(0)) / 2

    # LOE score calculation (custom logic)
    loe = combined_probs.get('decrease', 0) - combined_probs.get('increase', 0)

    biggest_type = combined_probs.idxmax() if not combined_probs.empty else 'no_data'

    results = {
        'probabilities': combined_probs.to_dict(),
        'loe_score': loe,
        'biggest_type': biggest_type
    }

    if detail_info:
        print("\nDetailed Analysis:")
        print(f"Non-MR Studies:\n{non_mr_probs}")
        print(f"\nMR Studies:\n{mr_probs}")
        print(f"\nCombined Probabilities:\n{combined_probs}")
        print(f"\nLevel of Evidence (LOE) Score: {loe:.4f}")
        print(f"Biggest Relationship Type: {biggest_type}")

    return results


def plot_cumulative_trends(results_df, title, focus_year=None, show_legend=True):
    """
    Plots the cumulative trends of the "Level of Evidence" (LOE) score,
    weighted by the number of participants.

    Args:
        results_df (pd.DataFrame): DataFrame containing `end_year` and `loe_score`.
        title (str): The title of the plot.
        focus_year (int, optional): An optional year to highlight with a vertical line.
        show_legend (bool, optional): Whether to display the legend.
    """
    results_df_disp = results_df.sort_values('end_year').copy()
    results_df_disp['cumulative_loe'] = results_df_disp['loe_score'].cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    base_fontsize = 12

    # Plot cumulative LOE score
    ax1.plot(results_df_disp['end_year'], results_df_disp['cumulative_loe'],
             color='blue', marker='o', linestyle='-', label='Cumulative LOE Score')
    ax1.set_xlabel('End Year of Study', fontsize=base_fontsize + 2)
    ax1.set_ylabel('Cumulative Level of Evidence (LOE) Score', color='blue', fontsize=base_fontsize + 2)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

    # Add a second y-axis for LOE score per year
    ax2 = ax1.twinx()
    bars = ax2.bar(results_df_disp['end_year'], results_df_disp['loe_score'],
                   color='gray', alpha=0.5, label='LOE Score per Year')
    ax2.set_ylabel('LOE Score per Year', color='gray', fontsize=base_fontsize + 2)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(min(results_df_disp['loe_score']) * 1.1, max(results_df_disp['loe_score']) * 1.1)

    plt.title(title, fontsize=base_fontsize + 4, fontweight='bold')

    ax1.set_xlim(results_df_disp['end_year'].min() - 1, results_df_disp['end_year'].max() + 1)

    # 5-year ticks for x-axis
    start_tick = (results_df_disp['end_year'].min() // 5) * 5
    end_tick = (results_df_disp['end_year'].max() // 5) * 5
    xticks = np.arange(start_tick, end_tick + 1, 5)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=base_fontsize)

    # Focus year marker
    if focus_year and focus_year in results_df_disp['end_year'].values:
        ax1.axvline(focus_year, linestyle='dashed', linewidth=1.5, color='red')
        ax1.text(focus_year, ax1.get_ylim()[1] * 0.9, str(focus_year),
                 color='red', fontsize=base_fontsize + 2,
                 ha='center', fontweight='bold')

    # Create a single legend with custom patches
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Custom patches for a more informative legend
    patch1 = Patch(color='blue', label='Cumulative LOE')
    patch2 = Patch(color='gray', alpha=0.5, label='LOE Per Year')

    if show_legend:
        plt.legend(handles=[patch1, patch2], loc='best', fontsize=base_fontsize)

    fig.tight_layout()
    plt.savefig('cumulative_trends_plot.png')
    plt.show()


if __name__ == "__main__":
    # --- Main script execution ---
    # This section loads your data and calls the analysis and plotting functions.
    # You can customize the file paths and data handling logic here.

    print("Loading and processing data...")

    try:
        # Load the relevant CSV files.
        # Note: The `cdsr_salt_bp...` files seem to be more suitable for this analysis
        # as they contain 'direction', 'Significance', 'participants', and 'PMID'.
        # The 'pub_year' needs to be extracted or looked up from another file like `input_df_sample_20.xlsx`.

        # For demonstration, we will combine data from a few files.
        # In a real-world scenario, you would have a single consolidated file.
        df1 = pd.read_csv("cdsr_salt_bp_hypertensive.xlsx - pub1.SBP.csv")
        df2 = pd.read_csv("cdsr_salt_bp_hypertensive.xlsx - pub1.DBP.csv")
        df3 = pd.read_csv("all_got_df_final_step_2_salt_cvd_021025.xlsx - Sheet1.csv")

        # Let's combine the data and add a 'pub_year' and 'study_design' column for consistency.
        # This is a simplified example; your actual data might require a more complex merge.
        combined_df = pd.concat([df1, df2]).rename(
            columns={'Significance': 'significance', 'participants': 'number_of_participants'})

        # Add publication years and study design from other files or hardcode for now
        # In a full pipeline, this data would be merged from a lookup table.
        pmid_to_year = {
            '3475429': 1986, '2563786': 1989, '6125636': 1982, '6133987': 1983,
            '1132079': 1975, '74660': 1978, '11136953': 2001, '11231700': 2001,
            '19620514': 2009, '31350809': 2019, '28934190': 2017
        }

        # Add the 'pub_year' and a dummy 'study_design'
        combined_df['pub_year'] = combined_df['PMID'].map(pmid_to_year)
        combined_df['study_design'] = 'RCT'  # Assuming these are RCTs based on the file name.

        # Filter out rows with missing publication year
        combined_df.dropna(subset=['pub_year'], inplace=True)

        # Analyze the data for each year and collect results
        yearly_results = []
        for year, group_df in combined_df.groupby('pub_year'):
            result = analyze_tri_df(group_df, detail_info=False)
            yearly_results.append({
                'end_year': year,
                'loe_score': result['loe_score']
            })

        results_df = pd.DataFrame(yearly_results)

        print("Analysis complete. Generating plot...")

        # Generate and save the plot. The image will be saved to the same directory
        # as this script.
        plot_cumulative_trends(results_df, "Cumulative Trends of Salt and Blood Pressure Evidence")

        print("Plot generated successfully and saved as 'cumulative_trends_plot.png'.")

    except FileNotFoundError as e:
        print(f"Error: The required file was not found. Please ensure all CSV files are in the same directory.")
        print(f"Missing file: {e.filename}")
        sys.exit(1)
