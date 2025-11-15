"""
Sectoral Growth Analysis - Configurable Plot Generator

This script generates cumulative growth plots for Indian economic sectors.

KEY PRINCIPLE: Never compare absolute values across datasets with different base years!
- Case A (1950-2004): Base year unknown
- Case B (2004-2011): Base year unknown  
- Case C (2011-2024): Different base year

We only use GROWTH RATES within each dataset, then chain them together.

Usage:
    python plot_sectoral_growth.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES
# ============================================================================

# Data file paths
F51_04 = "./mnt/data/1951-2004.csv"
F04_11 = "./mnt/data/2004-2011.csv"
F11_24 = "./mnt/data/2011-2024.csv"

# Set the year range (format: 'YYYY-YY')
START_YEAR = '1980-81'  # Available: 1950-51 onwards
END_YEAR = '2022-23'    # Available: up to 2022-23

# Toggle sectors on/off (True = show, False = hide)
SECTOR_TOGGLES = {
    'Agriculture': True,
    'Mining': True,
    'Manufacturing': True,
    'Electricity': True,
    'Construction': True,
    'Trade': True,
    'Transport, storage & communication': True,
    'Financial services': True,
    'Real estate': True,
    'Community, social & personal services': True
}

# Plot settings
PLOT_WIDTH = 20
PLOT_HEIGHT = 10
SHOW_GRID = True
SHOW_LEGEND = True
SAVE_PLOT = True  # Set to False if you only want to display, not save
OUTPUT_FILENAME = 'sectoral_growth_plot.png'

# ============================================================================
# MAIN CODE - DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Remove 'S.No.' column if it exists
        if 'S.No.' in df.columns:
            df = df.drop('S.No.', axis=1)
        df.set_index('Sector', inplace=True)
        
        # Convert all columns to numeric, handling commas and quotes
        for col in df.columns:
            # Remove commas and quotes, then convert to numeric
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def standardize_sector_names(df, case_type):
    """Standardize sector names across different datasets"""
    if case_type == 'A':  # 1951-2004
        mapping = {
            'Agriculture': 'Agriculture',
            'Mining': 'Mining',
            'Manufacturing': 'Manufacturing',
            'Electricity': 'Electricity',
            'Construction': 'Construction',
            'Trade': 'Trade',
            'transport, storage & communication': 'Transport, storage & communication',
            'financing, business and real estate': 'Financial services + Real estate',
            'community, social & personal services': 'Community, social & personal services'
        }
    elif case_type == 'B':  # 2004-2011
        mapping = {
            'Agriculture': 'Agriculture',
            'Mining': 'Mining',
            'Manufacturing': 'Manufacturing',
            'Electricity': 'Electricity',
            'Construction': 'Construction',
            'Trade': 'Trade',
            'Transport, storage & communication': 'Transport, storage & communication',
            'Financing, business and real estate': 'Financial services + Real estate',
            'Community, social & personal services': 'Community, social & personal services'
        }
    else:  # case_type == 'C', 2011-2024
        mapping = {
            'Agriculture': 'Agriculture',
            'Mining': 'Mining',
            'Manufacturing': 'Manufacturing',
            'Electricity': 'Electricity',
            'Construction': 'Construction',
            'Trade': 'Trade',
            'transport, storage & communication': 'Transport, storage & communication',
            'Financial services': 'Financial services',
            'Real estate': 'Real estate',
            'community, social & personal services': 'Community, social & personal services'
        }
    
    existing_mapping = {k: v for k, v in mapping.items() if k in df.index}
    df = df.rename(index=existing_mapping)
    return df

def calculate_cumulative_growth(df):
    """
    Calculate cumulative growth for each sector across all years WITHIN this dataset
    """
    cumulative_growth = pd.DataFrame(index=df.index)
    years = df.columns.tolist()
    
    # Initialize with 0% growth for the first year
    cumulative_growth[years[0]] = 0.0
    
    # Calculate year-over-year growth and cumulative growth
    for i in range(1, len(years)):
        prev_year = years[i-1]
        curr_year = years[i]
        
        # Calculate growth rate WITHIN this dataset: (V(t) - V(t-1)) / V(t-1)
        growth_rate = (df[curr_year] - df[prev_year]) / df[prev_year]
        
        # Calculate cumulative growth
        prev_cumulative = cumulative_growth[prev_year] / 100
        cumulative_growth[curr_year] = ((1 + prev_cumulative) * (1 + growth_rate) - 1) * 100
    
    return cumulative_growth

def combine_periods(cum_growth_1, cum_growth_2, cum_growth_3):
    """
    Combine three periods by chaining growth rates
    
    Key: We use growth rates (%) from each period, NOT absolute values
    """
    
    years_1 = cum_growth_1.columns.tolist()
    years_2 = cum_growth_2.columns.tolist()
    years_3 = cum_growth_3.columns.tolist()
    
    all_sectors = [
        'Agriculture', 'Mining', 'Manufacturing', 'Electricity', 'Construction',
        'Trade', 'Transport, storage & communication', 'Financial services',
        'Real estate', 'Community, social & personal services'
    ]
    
    combined = pd.DataFrame(index=all_sectors)
    
    # ========================================================================
    # PERIOD 1: 1950-51 to 2004-05
    # ========================================================================
    for year in years_1:
        combined[year] = 0.0
        for sector in all_sectors:
            if sector in ['Financial services', 'Real estate']:
                if 'Financial services + Real estate' in cum_growth_1.index:
                    combined.loc[sector, year] = cum_growth_1.loc['Financial services + Real estate', year]
            else:
                if sector in cum_growth_1.index:
                    combined.loc[sector, year] = cum_growth_1.loc[sector, year]
    
    # Get ending cumulative growth from Period 1
    ending_growth_p1 = {}
    for sector in all_sectors:
        if sector in ['Financial services', 'Real estate']:
            if 'Financial services + Real estate' in cum_growth_1.index:
                ending_growth_p1[sector] = cum_growth_1.loc['Financial services + Real estate', years_1[-1]]
        else:
            if sector in cum_growth_1.index:
                ending_growth_p1[sector] = cum_growth_1.loc[sector, years_1[-1]]
    
    # ========================================================================
    # PERIOD 2: 2005-06 to 2011-12 (skip 2004-05 duplicate)
    # ========================================================================
    for i, year in enumerate(years_2):
        if i == 0:  # Skip first year (2004-05) as it's duplicate
            continue
            
        combined[year] = 0.0
        for sector in all_sectors:
            if sector in ['Financial services', 'Real estate']:
                if 'Financial services + Real estate' in cum_growth_2.index:
                    growth_in_p2 = cum_growth_2.loc['Financial services + Real estate', year]
                    base = ending_growth_p1.get(sector, 0) / 100
                    period_growth = growth_in_p2 / 100
                    combined.loc[sector, year] = ((1 + base) * (1 + period_growth) - 1) * 100
            else:
                if sector in cum_growth_2.index:
                    growth_in_p2 = cum_growth_2.loc[sector, year]
                    base = ending_growth_p1.get(sector, 0) / 100
                    period_growth = growth_in_p2 / 100
                    combined.loc[sector, year] = ((1 + base) * (1 + period_growth) - 1) * 100
    
    # Get ending cumulative growth from Period 2
    ending_growth_p2 = {}
    last_year_p2 = years_2[-1]
    for sector in all_sectors:
        if sector in ['Financial services', 'Real estate']:
            if 'Financial services + Real estate' in cum_growth_2.index:
                growth_in_p2 = cum_growth_2.loc['Financial services + Real estate', last_year_p2]
                base = ending_growth_p1.get(sector, 0) / 100
                period_growth = growth_in_p2 / 100
                ending_growth_p2[sector] = ((1 + base) * (1 + period_growth) - 1) * 100
        else:
            if sector in cum_growth_2.index:
                growth_in_p2 = cum_growth_2.loc[sector, last_year_p2]
                base = ending_growth_p1.get(sector, 0) / 100
                period_growth = growth_in_p2 / 100
                ending_growth_p2[sector] = ((1 + base) * (1 + period_growth) - 1) * 100
    
    # ========================================================================
    # PERIOD 3: 2012-13 to 2022-23 (skip 2011-12 duplicate)
    # ========================================================================
    for i, year in enumerate(years_3):
        if i == 0:  # Skip first year (2011-12) as it's duplicate
            continue
            
        combined[year] = 0.0
        for sector in all_sectors:
            if sector in cum_growth_3.index:
                growth_in_p3 = cum_growth_3.loc[sector, year]
                base = ending_growth_p2.get(sector, 0) / 100
                period_growth = growth_in_p3 / 100
                combined.loc[sector, year] = ((1 + base) * (1 + period_growth) - 1) * 100
    
    return combined

def recalculate_growth_from_start(cum_growth_1, cum_growth_2, cum_growth_3, start_year, end_year):
    """
    Recalculate cumulative growth starting fresh from start_year
    
    KEY: We only use GROWTH RATES from within each dataset, never absolute values across datasets
    """
    # Get all years from each dataset
    years_1 = cum_growth_1.columns.tolist()
    years_2 = cum_growth_2.columns.tolist()
    years_3 = cum_growth_3.columns.tolist()
    
    # Create full year list (removing duplicates)
    all_years = years_1.copy()
    for year in years_2:
        if year not in all_years:
            all_years.append(year)
    for year in years_3:
        if year not in all_years:
            all_years.append(year)
    
    # Find start and end indices
    try:
        start_idx = all_years.index(start_year)
    except ValueError:
        print(f"Warning: Start year {start_year} not found. Using first available: {all_years[0]}")
        start_year = all_years[0]
        start_idx = 0
    
    try:
        end_idx = all_years.index(end_year)
    except ValueError:
        print(f"Warning: End year {end_year} not found. Using last available: {all_years[-1]}")
        end_year = all_years[-1]
        end_idx = len(all_years) - 1
    
    selected_years = all_years[start_idx:end_idx+1]
    
    # Initialize result
    all_sectors = [
        'Agriculture', 'Mining', 'Manufacturing', 'Electricity', 'Construction',
        'Trade', 'Transport, storage & communication', 'Financial services',
        'Real estate', 'Community, social & personal services'
    ]
    
    result = pd.DataFrame(index=all_sectors, columns=selected_years)
    
    # First year is always 0%
    result[selected_years[0]] = 0.0
    
    # Calculate growth for each subsequent year
    for i in range(1, len(selected_years)):
        prev_year = selected_years[i-1]
        curr_year = selected_years[i]
        
        # Determine which dataset contains BOTH prev_year and curr_year
        if prev_year in years_1 and curr_year in years_1:
            # Both years in Case A - use Case A growth rate
            source_cum_growth = cum_growth_1
            source_label = 'Case A'
        elif prev_year in years_2 and curr_year in years_2:
            # Both years in Case B - use Case B growth rate
            source_cum_growth = cum_growth_2
            source_label = 'Case B'
        elif prev_year in years_3 and curr_year in years_3:
            # Both years in Case C - use Case C growth rate
            source_cum_growth = cum_growth_3
            source_label = 'Case C'
        else:
            # Years span across datasets - we need to use the growth rate from the dataset containing curr_year
            # We use growth from the FIRST year of the new dataset to curr_year
            if curr_year in years_2 and curr_year != years_2[0]:
                # Current year is in Case B (not the first year)
                source_cum_growth = cum_growth_2
                source_label = 'Case B'
            elif curr_year in years_3 and curr_year != years_3[0]:
                # Current year is in Case C (not the first year)
                source_cum_growth = cum_growth_3
                source_label = 'Case C'
            elif curr_year == years_2[0]:
                # Transition year from Case A to Case B
                # Use the last year of Case A and first year of Case B to get growth within Case B
                # Actually, at transition we use growth from prev year within Case A to last year of Case A
                # Then from first year of Case B to curr year within Case B
                # But this gets complicated. Let's use the simpler approach:
                # At transition years, use 0% growth (no jump)
                for sector in all_sectors:
                    prev_cum = result.loc[sector, prev_year] / 100
                    result.loc[sector, curr_year] = (1 + prev_cum - 1) * 100  # Same as previous
                continue
            elif curr_year == years_3[0]:
                # Transition year from Case B to Case C
                for sector in all_sectors:
                    prev_cum = result.loc[sector, prev_year] / 100
                    result.loc[sector, curr_year] = (1 + prev_cum - 1) * 100  # Same as previous
                continue
            else:
                print(f"Warning: Cannot determine growth from {prev_year} to {curr_year}")
                continue
        
        # Get growth rate from the source dataset
        for sector in all_sectors:
            # Map sector names for combined Financial services + Real estate
            if sector in ['Financial services', 'Real estate']:
                source_sector = 'Financial services + Real estate' if 'Financial services + Real estate' in source_cum_growth.index else sector
            else:
                source_sector = sector
            
            if source_sector in source_cum_growth.index:
                # Get cumulative growth at prev_year and curr_year from the source dataset
                cum_at_prev = source_cum_growth.loc[source_sector, prev_year] / 100
                cum_at_curr = source_cum_growth.loc[source_sector, curr_year] / 100
                
                # Calculate the growth rate between these two years
                # If value grew from (1 + cum_at_prev) to (1 + cum_at_curr),
                # the growth rate is: (1 + cum_at_curr) / (1 + cum_at_prev) - 1
                year_growth_rate = (1 + cum_at_curr) / (1 + cum_at_prev) - 1
                
                # Apply this growth rate to our cumulative growth
                prev_cum_in_result = result.loc[sector, prev_year] / 100
                result.loc[sector, curr_year] = ((1 + prev_cum_in_result) * (1 + year_growth_rate) - 1) * 100
    
    return result, start_year, end_year

def plot_cumulative_growth(cumulative_growth, sector_toggles, plot_settings):
    """Generate the cumulative growth plot"""
    
    colors = {
        'Agriculture': '#2E7D32',
        'Mining': '#795548',
        'Manufacturing': '#1976D2',
        'Electricity': '#F57C00',
        'Construction': '#455A64',
        'Trade': '#C2185B',
        'Transport, storage & communication': '#7B1FA2',
        'Financial services': '#0097A7',
        'Real estate': '#00897B',
        'Community, social & personal services': '#E64A19'
    }
    
    fig, ax = plt.subplots(figsize=(plot_settings['width'], plot_settings['height']))
    
    years = cumulative_growth.columns.tolist()
    
    for sector in cumulative_growth.index:
        if sector_toggles.get(sector, False):
            values = cumulative_growth.loc[sector].values
            ax.plot(years, values, label=sector, linewidth=2.5, 
                   color=colors.get(sector, '#000000'), marker='o', markersize=3, alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Growth (%)', fontsize=14, fontweight='bold')
    
    start_year = years[0]
    end_year = years[-1]
    ax.set_title(f'Cumulative Growth by Sector ({start_year} to {end_year})', 
                fontsize=18, fontweight='bold', pad=20)
    
    if plot_settings['show_legend']:
        enabled_count = sum(1 for v in sector_toggles.values() if v)
        if enabled_count <= 5:
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        else:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    
    if plot_settings['show_grid']:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    n_years = len(years)
    if n_years > 50:
        step = 5
    elif n_years > 30:
        step = 3
    else:
        step = 2
    
    plt.xticks(range(0, len(years), step), [years[i] for i in range(0, len(years), step)], 
               rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

def print_summary(cumulative_growth, sector_toggles, start_year, end_year):
    """Print summary statistics for enabled sectors"""
    print("\n" + "="*80)
    print(f"CUMULATIVE GROWTH SUMMARY ({start_year} to {end_year})")
    print("="*80)
    
    final_growth = cumulative_growth.iloc[:, -1].sort_values(ascending=False)
    
    enabled_sectors = [s for s in final_growth.index if sector_toggles.get(s, False)]
    
    if not enabled_sectors:
        print("No sectors enabled!")
        return
    
    print(f"\nShowing {len(enabled_sectors)} sector(s):\n")
    
    rank = 1
    for sector in final_growth.index:
        if sector in enabled_sectors:
            growth = final_growth[sector]
            print(f"{rank:2d}. {sector:45s} : {growth:15,.2f}%")
            rank += 1
    
    print("\n" + "="*80)

def main():
    print("="*80)
    print("SECTORAL GROWTH ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Start Year: {START_YEAR}")
    print(f"  End Year: {END_YEAR}")
    print(f"  Enabled Sectors: {sum(1 for v in SECTOR_TOGGLES.values() if v)}/{len(SECTOR_TOGGLES)}")
    print("\n" + "="*80)
    
    # Load data
    print("\nLoading data...")
    df_1951_2004 = load_data(F51_04)
    df_2004_2011 = load_data(F04_11)
    df_2011_2024 = load_data(F11_24)
    
    if df_1951_2004 is None or df_2004_2011 is None or df_2011_2024 is None:
        print("Error: Could not load all data files. Exiting.")
        return
    
    # Standardize sector names
    print("Standardizing sector names...")
    df_1951_2004 = standardize_sector_names(df_1951_2004, 'A')
    df_2004_2011 = standardize_sector_names(df_2004_2011, 'B')
    df_2011_2024 = standardize_sector_names(df_2011_2024, 'C')
    
    # Calculate cumulative growth within each dataset
    print("\nCalculating cumulative growth within each dataset...")
    cum_growth_1951_2004 = calculate_cumulative_growth(df_1951_2004)
    cum_growth_2004_2011 = calculate_cumulative_growth(df_2004_2011)
    cum_growth_2011_2024 = calculate_cumulative_growth(df_2011_2024)
    
    # Recalculate cumulative growth from START_YEAR to END_YEAR
    # This uses only growth rates WITHIN each dataset
    print(f"\nRecalculating cumulative growth from {START_YEAR} to {END_YEAR}...")
    print("  (Using only growth rates within same base year datasets)")
    filtered_growth, actual_start, actual_end = recalculate_growth_from_start(
        cum_growth_1951_2004,
        cum_growth_2004_2011,
        cum_growth_2011_2024,
        START_YEAR,
        END_YEAR
    )
    
    # Print summary
    print_summary(filtered_growth, SECTOR_TOGGLES, actual_start, actual_end)
    
    # Generate plot
    print("\nGenerating plot...")
    plot_settings = {
        'width': PLOT_WIDTH,
        'height': PLOT_HEIGHT,
        'show_grid': SHOW_GRID,
        'show_legend': SHOW_LEGEND
    }
    
    fig = plot_cumulative_growth(filtered_growth, SECTOR_TOGGLES, plot_settings)
    
    # Save or show plot
    if SAVE_PLOT:
        fig.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as: {OUTPUT_FILENAME}")
    else:
        print("\nDisplaying plot...")
        plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
