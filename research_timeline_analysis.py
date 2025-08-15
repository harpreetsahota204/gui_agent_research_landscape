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
    
    plt.title('GUI Agent Research Timeline\nTotal Papers Published by Year (with 2025-2026 Projection)', 
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
    
    plt.title('Platform Research Trends Over Time\nNumber of Papers by Target Platform', 
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

def create_innovation_trends_chart(data, output_dir):
    """Create a multi-line chart showing innovation type trends over time."""
    print("Creating innovation trends chart...")
    
    # Innovation types to track
    innovation_types = [
        'introduces_new_model',
        'introduces_new_dataset', 
        'introduces_new_benchmark',
        'introduces_new_architecture'
    ]
    
    # Track innovation types by year
    innovation_by_year = defaultdict(lambda: defaultdict(int))
    
    for paper in data:
        year = paper.get('year')
        if year:
            for innovation_type in innovation_types:
                if paper.get(innovation_type) == True:
                    innovation_by_year[year][innovation_type] += 1
    
    # Create DataFrame for plotting
    years = sorted(innovation_by_year.keys())
    innovation_data = {}
    
    # Clean up labels for display
    label_mapping = {
        'introduces_new_model': 'New Models',
        'introduces_new_dataset': 'New Datasets',
        'introduces_new_benchmark': 'New Benchmarks', 
        'introduces_new_architecture': 'New Architectures'
    }
    
    for innovation_type in innovation_types:
        innovation_data[innovation_type] = [innovation_by_year[year][innovation_type] for year in years]
    
    # Create the line chart
    plt.figure(figsize=(12, 8))
    
    for innovation_type in innovation_types:
        label = label_mapping[innovation_type]
        plt.plot(years, innovation_data[innovation_type], marker='o', 
                linewidth=2, markersize=6, label=label)
    
    plt.title('Innovation Type Trends Over Time\nNumber of Papers by Contribution Type', 
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
    
    print("Total contributions by type:")
    for innovation_type, total in total_by_type.items():
        print(f"  {label_mapping[innovation_type]}: {total}")

def main():
    """Main analysis function."""
    # Create output directory
    output_dir = create_output_directory()
    
    # Load data
    data = load_data('keyword_filtered_enriched_qwen3_8b.json')
    
    print("\n" + "="*60)
    print("GENERATING TIMELINE ANALYSIS CHARTS")
    print("="*60)
    
    # Generate all three charts
    create_timeline_chart(data, output_dir)
    print()
    
    create_platform_trends_chart(data, output_dir)
    print()
    
    create_innovation_trends_chart(data, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete! Generated 3 charts in 'timeline_analysis/' directory:")
    print(f"  1. {os.path.join(output_dir, 'timeline_overall.png')} - Overall research timeline")
    print(f"  2. {os.path.join(output_dir, 'timeline_platforms.png')} - Platform trends over time") 
    print(f"  3. {os.path.join(output_dir, 'timeline_innovations.png')} - Innovation type trends over time")
    print("="*60)

if __name__ == "__main__":
    main()
