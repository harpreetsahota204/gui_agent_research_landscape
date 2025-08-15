#!/usr/bin/env python3
"""
Research Timeline Analysis for GUI Agent Research Landscape

This script analyzes the temporal trends in GUI agent research by creating three visualizations:
1. Overall paper count by year (line chart)
2. Platform research trends over time (multi-line chart)
3. Innovation type trends over time (multi-line chart)

All charts use absolute counts and include data from 2016-2025.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style for beautiful charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(filepath):
    """Load and return the JSON data."""
    print("Loading research data...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} papers")
    return data

def create_output_directory():
    """Create the timeline_analysis directory if it doesn't exist."""
    output_dir = 'timeline_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")
    return output_dir

def create_timeline_chart(data, output_dir):
    """Create a line chart showing total papers published by year with 2025 projection."""
    print("Creating overall timeline chart with 2025 projection...")
    
    # Count papers by year and analyze 2025 monthly data
    year_counts = Counter()
    month_2025_counts = Counter()
    
    for paper in data:
        year = paper.get('year')
        month = paper.get('month')
        if year:
            year_counts[year] += 1
            # Track monthly data for 2025
            if year == 2025 and month:
                month_2025_counts[month] += 1
    
    # Analyze 2025 data for projection
    if 2025 in year_counts and month_2025_counts:
        latest_month_2025 = max(month_2025_counts.keys())
        papers_so_far_2025 = year_counts[2025]
        
        # Simple linear projection: (papers_so_far / months_elapsed) * 12
        projected_2025 = int((papers_so_far_2025 / latest_month_2025) * 12)
        
        print(f"2025 Analysis:")
        print(f"  Papers through month {latest_month_2025}: {papers_so_far_2025}")
        print(f"  Projected full-year 2025: ~{projected_2025} papers")
    
    # Convert to sorted lists for plotting
    years = sorted(year_counts.keys())
    counts = [year_counts[year] for year in years]
    
    # Create trend line for recent years (2020 onwards for better trend)
    recent_years = [y for y in years if y >= 2020 and y <= 2024]
    recent_counts = [year_counts[y] for y in recent_years]
    
    # Fit polynomial trend line
    if len(recent_years) >= 3:
        z = np.polyfit(recent_years, recent_counts, 2)  # Quadratic fit
        trend_years = np.linspace(2020, 2026, 100)
        trend_counts = np.polyval(z, trend_years)
        
        # Project 2025 and 2026 using trend
        trend_2025 = int(np.polyval(z, 2025))
        trend_2026 = int(np.polyval(z, 2026))
    
    # Create the chart
    plt.figure(figsize=(14, 8))
    
    # Plot actual data
    plt.plot(years, counts, marker='o', linewidth=2.5, markersize=8, 
             label='Actual Papers', color='#2E86AB', zorder=3)
    
    # Plot trend line
    if len(recent_years) >= 3:
        plt.plot(trend_years, trend_counts, '--', linewidth=2, alpha=0.7, 
                 label='Trend Projection', color='#A23B72', zorder=2)
    
    plt.title('GUI Agent Research Publications by Year (2016-2025)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Ensure every year has a mark on x-axis
    plt.xticks(years)
    
    # Add value labels on actual data points
    for year, count in zip(years, counts):
        if year == 2025:
            # Special annotation for 2025 partial data
            plt.annotate(f'{count}\n(through month {latest_month_2025})', 
                        (year, count), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        else:
            plt.annotate(str(count), (year, count), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
    
    # Add projection annotations
    if len(recent_years) >= 3:
        plt.annotate(f'Projected: ~{trend_2025}', (2025, trend_2025), 
                    textcoords="offset points", xytext=(20,-10), ha='left', 
                    fontsize=9, color='#A23B72', style='italic')
        plt.annotate(f'Projected: ~{trend_2026}', (2026, trend_2026), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=9, color='#A23B72', style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeline_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Timeline spans {min(years)} to {max(years)}")
    print(f"Peak year so far: {max(year_counts, key=year_counts.get)} with {max(year_counts.values())} papers")
    if len(recent_years) >= 3:
        print(f"Trend projection for 2025: ~{trend_2025} papers")
        print(f"Trend projection for 2026: ~{trend_2026} papers")

def create_platform_trends_chart(data, output_dir):
    """Create a multi-line chart showing platform research trends over time."""
    print("Creating platform trends chart...")
    
    # Track platform mentions by year
    platform_by_year = defaultdict(lambda: defaultdict(int))
    
    for paper in data:
        year = paper.get('year')
        platforms = paper.get('platform', [])
        
        if year and platforms:
            for platform in platforms:
                platform_by_year[year][platform] += 1
    
    # Get top platforms overall for cleaner visualization
    all_platforms = Counter()
    for paper in data:
        platforms = paper.get('platform', [])
        for platform in platforms:
            all_platforms[platform] += 1
    
    # Use top 8 platforms to keep chart readable
    top_platforms = [platform for platform, _ in all_platforms.most_common(8)]
    
    # Create DataFrame for easier plotting
    years = sorted(platform_by_year.keys())
    platform_data = {}
    
    for platform in top_platforms:
        platform_data[platform] = [platform_by_year[year][platform] for year in years]
    
    # Create the line chart
    plt.figure(figsize=(12, 8))
    
    for platform in top_platforms:
        plt.plot(years, platform_data[platform], marker='o', 
                linewidth=2, markersize=6, label=platform)
    
    plt.title('GUI Agent Research by Target Platform Over Time', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Ensure every year has a mark on x-axis
    plt.xticks(years)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeline_platforms.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Top platforms: {', '.join(top_platforms[:5])}")

def has_model_or_architecture_innovation(paper):
    """Check if paper introduces new model OR architecture (avoiding double counting)."""
    return paper.get('introduces_new_model') == True or paper.get('introduces_new_architecture') == True

def create_innovation_trends_chart(data, output_dir):
    """Create a multi-line chart showing innovation type trends over time."""
    print("Creating innovation trends chart...")
    
    # Innovation types to track (combining model and architecture to avoid double counting)
    innovation_types = [
        'introduces_new_model_or_architecture',  # Combined category
        'introduces_new_dataset', 
        'introduces_new_benchmark'
    ]
    
    # Track innovation types by year
    innovation_by_year = defaultdict(lambda: defaultdict(int))
    
    for paper in data:
        year = paper.get('year')
        if year:
            # Handle combined model/architecture category
            if has_model_or_architecture_innovation(paper):
                innovation_by_year[year]['introduces_new_model_or_architecture'] += 1
            
            # Handle other categories normally
            if paper.get('introduces_new_dataset') == True:
                innovation_by_year[year]['introduces_new_dataset'] += 1
            if paper.get('introduces_new_benchmark') == True:
                innovation_by_year[year]['introduces_new_benchmark'] += 1
    
    # Create DataFrame for plotting
    years = sorted(innovation_by_year.keys())
    innovation_data = {}
    
    # Clean up labels for display
    label_mapping = {
        'introduces_new_model_or_architecture': 'New Models/Architectures',
        'introduces_new_dataset': 'New Datasets',
        'introduces_new_benchmark': 'New Benchmarks'
    }
    
    for innovation_type in innovation_types:
        innovation_data[innovation_type] = [innovation_by_year[year][innovation_type] for year in years]
    
    # Create the line chart
    plt.figure(figsize=(12, 8))
    
    for innovation_type in innovation_types:
        label = label_mapping[innovation_type]
        plt.plot(years, innovation_data[innovation_type], marker='o', 
                linewidth=2, markersize=6, label=label)
    
    plt.title('Research Contributions by Innovation Type (2016-2025)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Ensure every year has a mark on x-axis
    plt.xticks(years)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeline_innovations.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some insights
    total_by_type = {}
    for innovation_type in innovation_types:
        total = sum(innovation_data[innovation_type])
        total_by_type[innovation_type] = total
    
    print("Total contributions by type (avoiding double counting):")
    for innovation_type, total in total_by_type.items():
        print(f"  {label_mapping[innovation_type]}: {total}")

def create_compound_innovation_chart(data, output_dir):
    """Create a stacked area chart showing papers with 0, 1, 2, or 3+ innovations over time."""
    print("Creating compound innovation analysis chart...")
    
    # Innovation types to track (combining model and architecture to avoid double counting)
    innovation_types = [
        'introduces_new_model_or_architecture',  # Combined category
        'introduces_new_dataset', 
        'introduces_new_benchmark'
    ]
    
    # Track compound innovations by year
    compound_by_year = defaultdict(lambda: defaultdict(int))
    innovation_lifecycle_data = defaultdict(lambda: defaultdict(int))
    
    for paper in data:
        year = paper.get('year')
        if year:
            # Count total innovations per paper (avoiding double counting)
            innovation_count = 0
            
            # Check model/architecture (count as one)
            if has_model_or_architecture_innovation(paper):
                innovation_count += 1
                innovation_lifecycle_data[year]['introduces_new_model_or_architecture'] += 1
            
            # Check dataset
            if paper.get('introduces_new_dataset') == True:
                innovation_count += 1
                innovation_lifecycle_data[year]['introduces_new_dataset'] += 1
                
            # Check benchmark
            if paper.get('introduces_new_benchmark') == True:
                innovation_count += 1
                innovation_lifecycle_data[year]['introduces_new_benchmark'] += 1
            
            # Categorize by innovation count (now max is 3 since we combined model/arch)
            if innovation_count == 0:
                compound_by_year[year]['0_innovations'] += 1
            elif innovation_count == 1:
                compound_by_year[year]['1_innovation'] += 1
            elif innovation_count == 2:
                compound_by_year[year]['2_innovations'] += 1
            else:  # 3 (all three types)
                compound_by_year[year]['3_innovations'] += 1
    
    # Prepare data for stacked area chart
    years = sorted(compound_by_year.keys())
    categories = ['0_innovations', '1_innovation', '2_innovations', '3_innovations']
    category_labels = ['0 Innovations', '1 Innovation', '2 Innovations', '3 Innovations (All Types)']
    
    # Create data arrays for each category
    category_data = {}
    for category in categories:
        category_data[category] = [compound_by_year[year][category] for year in years]
    
    # Create stacked area chart
    plt.figure(figsize=(14, 10))
    
    # Create the stacked areas
    bottom = np.zeros(len(years))
    colors = ['#E8E8E8', '#A8D8EA', '#AA96DA', '#FCBAD3']  # Light to vibrant progression
    
    for i, category in enumerate(categories):
        label = category_labels[i]
        plt.fill_between(years, bottom, bottom + category_data[category], 
                        alpha=0.8, label=label, color=colors[i])
        bottom += category_data[category]
    
    plt.title('Distribution of Papers by Number of Innovations per Paper', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Ensure every year has a mark on x-axis
    plt.xticks(years)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compound_innovations.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and display insights
    total_papers_by_year = {}
    comprehensive_papers_by_year = {}  # 2+ innovations
    
    for year in years:
        total = sum(category_data[category][years.index(year)] for category in categories)
        comprehensive = (category_data['2_innovations'][years.index(year)] + 
                        category_data['3_innovations'][years.index(year)])
        total_papers_by_year[year] = total
        comprehensive_papers_by_year[year] = comprehensive
    
    print("\nCompound Innovation Insights:")
    print("=" * 50)
    
    # Recent trends (last 3 years with data)
    recent_years = [y for y in years if y >= 2022]
    if recent_years:
        recent_comprehensive_pct = []
        for year in recent_years:
            if total_papers_by_year[year] > 0:
                pct = (comprehensive_papers_by_year[year] / total_papers_by_year[year]) * 100
                recent_comprehensive_pct.append(pct)
                print(f"{year}: {comprehensive_papers_by_year[year]}/{total_papers_by_year[year]} papers ({pct:.1f}%) with 2+ innovations")
        
        if len(recent_comprehensive_pct) > 1:
            trend = "increasing" if recent_comprehensive_pct[-1] > recent_comprehensive_pct[0] else "decreasing"
            print(f"\nTrend: Comprehensive contributions are {trend} in recent years")
    
    # Overall statistics
    total_comprehensive = sum(comprehensive_papers_by_year.values())
    total_all_papers = sum(total_papers_by_year.values())
    overall_comprehensive_pct = (total_comprehensive / total_all_papers) * 100
    print(f"\nOverall: {total_comprehensive}/{total_all_papers} papers ({overall_comprehensive_pct:.1f}%) introduce 2+ innovations")
    
    # Innovation lifecycle analysis
    print("\nInnovation Lifecycle Analysis (Corrected):")
    print("=" * 50)
    
    # Calculate peak years for each innovation type
    lifecycle_peaks = {}
    for innovation_type in innovation_types:
        max_year = max(years, key=lambda y: innovation_lifecycle_data[y][innovation_type])
        max_count = innovation_lifecycle_data[max_year][innovation_type]
        lifecycle_peaks[innovation_type] = (max_year, max_count)
    
    label_mapping = {
        'introduces_new_model_or_architecture': 'New Models/Architectures',
        'introduces_new_dataset': 'New Datasets',
        'introduces_new_benchmark': 'New Benchmarks'
    }
    
    for innovation_type, (peak_year, peak_count) in lifecycle_peaks.items():
        print(f"{label_mapping[innovation_type]}: Peak in {peak_year} ({peak_count} papers)")
    
    print(f"\nNote: Models and Architectures are now combined to avoid double-counting")
    print(f"since they represent essentially the same type of contribution.")
    
    return category_data, years

def analyze_platform_consolidation(data, output_dir):
    """Analyze platform consolidation vs fragmentation trends."""
    print("Analyzing platform consolidation trends...")
    
    # Track platform diversity metrics by year
    platform_metrics_by_year = {}
    
    for year in range(2016, 2026):  # Cover full timeline
        year_papers = [paper for paper in data if paper.get('year') == year]
        if not year_papers:
            continue
            
        # Count unique platforms mentioned
        all_platforms = set()
        platform_counts = Counter()
        
        for paper in year_papers:
            platforms = paper.get('platforms', [])  # Note: checking both 'platform' and 'platforms'
            if not platforms:
                platforms = paper.get('platform', [])
            
            for platform in platforms:
                all_platforms.add(platform)
                platform_counts[platform] += 1
        
        # Calculate diversity metrics
        total_platforms = len(all_platforms)
        total_mentions = sum(platform_counts.values())
        
        # Herfindahl-Hirschman Index (HHI) for concentration
        # HHI = sum of squared market shares (0 = perfect competition, 1 = monopoly)
        if total_mentions > 0:
            hhi = sum((count / total_mentions) ** 2 for count in platform_counts.values())
            # Convert to concentration ratio (higher = more concentrated)
            concentration = hhi
        else:
            concentration = 0
        
        # Top 3 platform dominance
        top3_share = sum(count for _, count in platform_counts.most_common(3)) / max(total_mentions, 1)
        
        platform_metrics_by_year[year] = {
            'unique_platforms': total_platforms,
            'total_mentions': total_mentions,
            'concentration_index': concentration,
            'top3_dominance': top3_share,
            'platform_counts': platform_counts
        }
    
    # Generate insights
    years = sorted(platform_metrics_by_year.keys())
    
    print("\nPlatform Consolidation Analysis:")
    print("=" * 50)
    
    if len(years) >= 3:
        early_years = years[:3]
        recent_years = years[-3:]
        
        early_diversity = np.mean([platform_metrics_by_year[y]['unique_platforms'] for y in early_years])
        recent_diversity = np.mean([platform_metrics_by_year[y]['unique_platforms'] for y in recent_years])
        
        early_concentration = np.mean([platform_metrics_by_year[y]['concentration_index'] for y in early_years])
        recent_concentration = np.mean([platform_metrics_by_year[y]['concentration_index'] for y in recent_years])
        
        print(f"Platform Diversity:")
        print(f"  Early years ({early_years[0]}-{early_years[-1]}): {early_diversity:.1f} avg platforms/year")
        print(f"  Recent years ({recent_years[0]}-{recent_years[-1]}): {recent_diversity:.1f} avg platforms/year")
        
        diversity_trend = "increasing" if recent_diversity > early_diversity else "decreasing"
        print(f"  Trend: Platform diversity is {diversity_trend}")
        
        print(f"\nPlatform Concentration:")
        print(f"  Early years: {early_concentration:.3f} concentration index")
        print(f"  Recent years: {recent_concentration:.3f} concentration index")
        
        concentration_trend = "increasing" if recent_concentration > early_concentration else "decreasing"
        print(f"  Trend: Platform concentration is {concentration_trend}")
        
        # Identify dominant platforms over time
        print(f"\nDominant Platforms by Era:")
        for era, era_years in [("Early", early_years), ("Recent", recent_years)]:
            era_platform_counts = Counter()
            for year in era_years:
                for platform, count in platform_metrics_by_year[year]['platform_counts'].items():
                    era_platform_counts[platform] += count
            
            print(f"  {era} era top platforms: {', '.join([p for p, _ in era_platform_counts.most_common(3)])}")
    
    return platform_metrics_by_year

def analyze_acceleration_indicators(data, output_dir):
    """Analyze acceleration indicators and growth sustainability."""
    print("Analyzing acceleration and growth sustainability...")
    
    # Count papers by year
    year_counts = Counter()
    for paper in data:
        year = paper.get('year')
        if year:
            year_counts[year] += 1
    
    years = sorted(year_counts.keys())
    counts = [year_counts[year] for year in years]
    
    print("\nAcceleration Analysis:")
    print("=" * 40)
    
    # Calculate year-over-year growth rates
    growth_rates = []
    for i in range(1, len(counts)):
        if counts[i-1] > 0:
            growth_rate = ((counts[i] - counts[i-1]) / counts[i-1]) * 100
            growth_rates.append(growth_rate)
            print(f"{years[i-1]} → {years[i]}: {growth_rate:+.1f}% growth ({counts[i-1]} → {counts[i]} papers)")
    
    # Analyze growth pattern
    if len(growth_rates) >= 3:
        recent_growth = np.mean(growth_rates[-3:])
        early_growth = np.mean(growth_rates[:3]) if len(growth_rates) >= 6 else np.mean(growth_rates[:len(growth_rates)//2])
        
        print(f"\nGrowth Pattern Analysis:")
        print(f"  Early period average growth: {early_growth:.1f}%")
        print(f"  Recent period average growth: {recent_growth:.1f}%")
        
        if recent_growth > early_growth * 1.5:
            trend = "accelerating (exponential-like growth)"
        elif recent_growth < early_growth * 0.5:
            trend = "decelerating (approaching saturation?)"
        else:
            trend = "steady (linear-like growth)"
        
        print(f"  Growth trend: {trend}")
        
        # Sustainability analysis
        print(f"\nSustainability Indicators:")
        if recent_growth > 50:
            print("  ⚠️  Very high growth rates may be unsustainable")
        elif recent_growth > 20:
            print("  📈 Strong growth, monitor for inflection points")
        elif recent_growth > 0:
            print("  📊 Steady growth, appears sustainable")
        else:
            print("  📉 Growth has slowed or reversed")
    
    # Project next few years based on current trend
    if len(years) >= 3 and len(counts) >= 3:
        # Fit polynomial trend
        z = np.polyfit(years[-5:], counts[-5:], 2)  # Use last 5 years for trend
        
        future_years = [2026, 2027, 2028]
        projections = [int(np.polyval(z, year)) for year in future_years]
        
        print(f"\nProjected Growth (based on recent trend):")
        for year, projection in zip(future_years, projections):
            print(f"  {year}: ~{projection} papers (projected)")
    
    return year_counts, growth_rates

def main():
    """Main analysis function."""
    # Create output directory
    output_dir = create_output_directory()
    
    # Load data
    data = load_data('keyword_filtered_enriched_qwen3_8b.json')
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE TIMELINE ANALYSIS")
    print("="*60)
    
    # Generate all charts
    create_timeline_chart(data, output_dir)
    print()
    
    create_platform_trends_chart(data, output_dir)
    print()
    
    create_innovation_trends_chart(data, output_dir)
    print()
    
    create_compound_innovation_chart(data, output_dir)
    print()
    
    # Additional analyses
    analyze_platform_consolidation(data, output_dir)
    print()
    
    analyze_acceleration_indicators(data, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete! Generated charts and insights:")
    print(f"  1. {os.path.join(output_dir, 'timeline_overall.png')} - Overall research timeline")
    print(f"  2. {os.path.join(output_dir, 'timeline_platforms.png')} - Platform trends over time") 
    print(f"  3. {os.path.join(output_dir, 'timeline_innovations.png')} - Innovation type trends over time")
    print(f"  4. {os.path.join(output_dir, 'compound_innovations.png')} - Compound innovation analysis")
    print(f"  5. Platform consolidation analysis (printed above)")
    print(f"  6. Acceleration and sustainability analysis (printed above)")
    print("="*60)

if __name__ == "__main__":
    main()
