#!/usr/bin/env python3
"""
Generate papers table for README.md from keyword_filtered_enriched_qwen3_8b.json
"""

import json
import re
from collections import defaultdict

def get_quarter(month):
    """Convert month to quarter."""
    if not month:
        return "Unknown"
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    else:
        return "Q4"

def get_time_period(year, month):
    """Get time period bucket for a paper."""
    if not year:
        return "Unknown"
    
    if 2016 <= year <= 2021:
        return "2016-2021_Early_Era"
    elif year == 2022:
        return "2022_Growth_Year"
    elif year >= 2023:
        quarter = get_quarter(month)
        return f"{year}_{quarter}"
    else:
        return f"{year}_Other"

def create_arxiv_badge(arxiv_id):
    """Create an arXiv badge markdown link."""
    if not arxiv_id:
        return "No arXiv"
    
    # Clean up arXiv ID (remove version if present for URL)
    arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)
    
    return f"[![arXiv](https://img.shields.io/badge/arXiv-{arxiv_id}-b31b1b.svg)](https://arxiv.org/abs/{arxiv_id_clean})"

def truncate_text(text, max_length=150):
    """Truncate text to specified length with ellipsis."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."

def escape_markdown(text):
    """Escape special markdown characters in text."""
    if not text:
        return ""
    # Escape pipe characters and other markdown special chars
    text = text.replace('|', '\\|')
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = text.replace('\r', ' ')  # Replace carriage returns with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def load_and_organize_papers(filepath):
    """Load papers and organize by time periods."""
    print(f"Loading papers from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers")
    
    # Group papers by time periods
    time_periods = defaultdict(list)
    
    for paper in papers:
        year = paper.get('year')
        month = paper.get('month')
        arxiv_id = paper.get('arxiv_id')
        
        # Skip papers without arXiv ID or year
        if not arxiv_id or not year:
            continue
            
        period = get_time_period(year, month)
        time_periods[period].append(paper)
    
    # Sort periods chronologically
    period_order = []
    
    # Add early era
    if "2016-2021_Early_Era" in time_periods:
        period_order.append("2016-2021_Early_Era")
    
    # Add growth year
    if "2022_Growth_Year" in time_periods:
        period_order.append("2022_Growth_Year")
    
    # Add quarterly periods for 2023+
    years_with_quarters = set()
    for period in time_periods.keys():
        if period.startswith(('2023_', '2024_', '2025_')):
            year = int(period.split('_')[0])
            years_with_quarters.add(year)
    
    for year in sorted(years_with_quarters):
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            period_key = f"{year}_{quarter}"
            if period_key in time_periods:
                period_order.append(period_key)
    
    # Sort papers within each period by citation count (descending)
    for period in period_order:
        time_periods[period].sort(key=lambda p: len(p.get('cited_by', [])), reverse=True)
    
    return time_periods, period_order

def format_period_name(period):
    """Format period name for display."""
    if period == "2016-2021_Early_Era":
        return "2016-2021: Early Era"
    elif period == "2022_Growth_Year":
        return "2022: Growth Year"
    elif "_" in period:
        year, quarter = period.split("_", 1)
        return f"{year}: {quarter}"
    else:
        return period

def generate_period_table(papers, period_name):
    """Generate markdown table for a specific time period."""
    if not papers:
        return ""
    
    # Table header
    table = f"\n### {format_period_name(period_name)}\n\n"
    table += "| arXiv | Title | Summary | Contributions |\n"
    table += "|-------|-------|---------|---------------|\n"
    
    for paper in papers:
        arxiv_id = paper.get('arxiv_id', '')
        title = escape_markdown(paper.get('title', 'No title'))
        summary = escape_markdown(truncate_text(paper.get('key_contributions', '')))
        contributions = escape_markdown(truncate_text(paper.get('contribution', '')))
        
        # Create arXiv badge
        arxiv_badge = create_arxiv_badge(arxiv_id)
        
        # Truncate title if too long
        title = truncate_text(title, 80)
        
        table += f"| {arxiv_badge} | {title} | {summary} | {contributions} |\n"
    
    return table

def generate_full_table(time_periods, period_order):
    """Generate the complete papers table."""
    table_content = "\n## Papers by Time Period\n\n"
    table_content += "The following tables organize all papers in the dataset by time periods, showing their key contributions and innovations.\n\n"
    
    # Generate statistics
    total_papers = sum(len(papers) for papers in time_periods.values())
    table_content += f"**Total Papers**: {total_papers}\n\n"
    
    # Add period statistics
    table_content += "**Papers by Period**:\n"
    for period in period_order:
        period_papers = time_periods[period]
        table_content += f"- {format_period_name(period)}: {len(period_papers)} papers\n"
    table_content += "\n"
    
    # Generate tables for each period
    for period in period_order:
        period_papers = time_periods[period]
        if period_papers:  # Only add table if there are papers
            table_content += generate_period_table(period_papers, period)
            table_content += "\n"
    
    return table_content

def append_to_readme(table_content, readme_path="README.md"):
    """Append the table content to the README."""
    print(f"Appending table to {readme_path}...")
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        existing_content = f.read()
    
    # Check if papers table already exists
    if "## Papers by Time Period" in existing_content:
        print("Papers table already exists in README. Replacing...")
        # Remove existing table section
        parts = existing_content.split("## Papers by Time Period")
        existing_content = parts[0].rstrip()
    
    # Append new table
    updated_content = existing_content + table_content
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Table successfully added to README!")

def main():
    """Main function to generate and append the papers table."""
    # Load and organize papers
    time_periods, period_order = load_and_organize_papers('keyword_filtered_enriched_qwen3_8b.json')
    
    print("\nTime period distribution:")
    for period in period_order:
        print(f"  {format_period_name(period)}: {len(time_periods[period])} papers")
    
    # Generate table content
    table_content = generate_full_table(time_periods, period_order)
    
    # Append to README
    append_to_readme(table_content)
    
    print(f"\n✅ Successfully generated and added papers table to README!")
    print(f"   Total periods: {len(period_order)}")
    print(f"   Total papers: {sum(len(papers) for papers in time_periods.values())}")

if __name__ == "__main__":
    main()
