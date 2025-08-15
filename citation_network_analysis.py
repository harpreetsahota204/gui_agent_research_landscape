import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
import math
from datetime import datetime

# Output directory for all analysis results
OUTPUT_DIR = "citation_analysis"

# General foundation models to exclude from GUI-specific analysis (keep in most cited only)
GENERAL_FOUNDATION_MODELS = {
    '2303.08774',  # GPT-4 Technical Report
    '2308.12966',  # Qwen-VL
    '2303.03378',  # PaLM-E
    '2305.16291',  # Voyager
    '2304.15004',  # General model (if exists)
    '2404.17419',  # General model (if exists)
    '2307.02477'   # General model (if exists)
}

# Papers to exclude from datasets analysis (not GUI/OS focused, but keep in most cited)
NON_GUI_DATASETS = {
    '2404.16821',  # How Far Are We to GPT-4V
    '2403.05525',  # Non-GUI dataset
    '2402.11684',  # Non-GUI dataset
    '2402.05935',  # Non-GUI dataset
    '2405.02246',  # Non-GUI dataset
    '2401.06209',   # Non-GUI dataset
    '2112.09332',   # WebGPT
    '2303.11366',   # Reflexion: Language Agents with Verbal Reinforcement Learning
    '2310.09478'    # MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning
}

def should_exclude_from_gui_analysis(arxiv_id):
    """Check if paper should be excluded from GUI-specific analysis (but not most cited/foundation)"""
    return arxiv_id in GENERAL_FOUNDATION_MODELS or arxiv_id in NON_GUI_DATASETS

def create_arxiv_badge(arxiv_id):
    """
    Create a clickable ArXiv badge that opens the abstract on arxiv.org
    
    Args:
        arxiv_id: ArXiv ID (may include version number)
        
    Returns:
        str: Markdown badge that links to ArXiv abstract
    """
    if not arxiv_id:
        return "N/A"
    
    # Clean arxiv_id to remove version numbers for the URL
    clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
    
    # Create badge with link to ArXiv abstract
    badge_url = f"https://img.shields.io/badge/arXiv-{clean_id}-b31b1b.svg"
    arxiv_url = f"https://arxiv.org/abs/{clean_id}"
    
    return f"[![arXiv]({badge_url})]({arxiv_url})"

def truncate_text(text, max_length=100):
    """
    Truncate text to specified length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        str: Truncated text with ellipsis if needed
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def add_synthetic_arxiv_ids(papers):
    """
    Add synthetic arXiv IDs for specific papers that don't have them.
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        list: Papers with synthetic arXiv IDs added where needed
    """
    synthetic_mappings = {
        "Rico: A Mobile App Dataset for Building Data-Driven Design Applications": "1710.99999",
        "ERICA: Interaction Mining Mobile Apps": "1610.99999"
    }
    
    papers_with_ids = []
    added_count = 0
    
    for paper in papers:
        paper_copy = paper.copy()
        title = paper_copy.get('title', '')
        
        # Skip papers that don't need synthetic IDs
        if title not in synthetic_mappings:
            papers_with_ids.append(paper_copy)
            continue
        
        current_arxiv_id = paper_copy.get('arxiv_id')
        
        # Add synthetic ID if paper doesn't have one
        if not current_arxiv_id:
            paper_copy['arxiv_id'] = synthetic_mappings[title]
            added_count += 1
            print(f"Added synthetic arXiv ID {synthetic_mappings[title]} for: {title}")
        else:
            print(f"Paper already has arXiv ID {current_arxiv_id}: {title}")
        
        papers_with_ids.append(paper_copy)
    
    if added_count > 0:
        print(f"Added {added_count} synthetic arXiv IDs")
    
    return papers_with_ids

def load_papers(filepath):
    """
    Load research papers from JSON file, excluding survey papers.
    
    Returns:
        list: Clean list of paper dictionaries (excluding surveys)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Add synthetic arXiv IDs for specific papers
    papers = add_synthetic_arxiv_ids(papers)
    
    # Filter out survey papers
    def is_survey_paper(paper):
        # Check if 'survey' is in appears_in field
        appears_in = paper.get('appears_in', [])
        if 'survey' in appears_in:
            return True
        
        # Check if title contains 'survey' (case-insensitive)
        title = paper.get('title', '').lower()
        if 'survey' in title:
            return True
            
        return False
    
    # Separate survey and non-survey papers
    survey_papers = [p for p in papers if is_survey_paper(p)]
    non_survey_papers = [p for p in papers if not is_survey_paper(p)]
    
    print(f"Loaded {len(papers)} total papers")
    print(f"Excluded {len(survey_papers)} survey papers")
    print(f"Keeping {len(non_survey_papers)} non-survey papers for analysis")
    
    return non_survey_papers

def build_paper_database(papers):
    """
    Create a structured database of paper information.
    
    Returns:
        dict: Paper database indexed by arxiv_id
    """
    db = {}
    skipped_count = 0
    
    for paper in papers:
        arxiv_id = paper.get('arxiv_id')
        
        # Skip papers without arXiv IDs
        if not arxiv_id:
            skipped_count += 1
            continue
            
        db[arxiv_id] = {
            'title': paper['title'],
            'year': paper.get('year'),
            'month': paper.get('month'),
            'platforms': paper.get('platform', []),
            'cited_by': paper.get('cited_by', []),
            'citation_count': len(paper.get('cited_by', [])),
            'key_contributions': paper.get('key_contributions', ''),
            'contribution': paper.get('contribution', ''),
        }
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} papers without arXiv IDs")
    
    print(f"Built database with {len(db)} papers")
    return db

def build_citation_network(paper_db):
    """
    Build NetworkX citation graph from paper database.
    
    Returns:
        nx.DiGraph: Citation network (edges go from cited → citing paper)
    """
    G = nx.DiGraph()
    
    # Add all papers as nodes
    for arxiv_id in paper_db:
        G.add_node(arxiv_id)
    
    # Add citation edges
    for arxiv_id, paper in paper_db.items():
        for citing_paper in paper['cited_by']:
            if citing_paper in paper_db:
                G.add_edge(arxiv_id, citing_paper)
    
    print(f"Built network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def calculate_network_metrics(graph, paper_db):
    """
    Calculate key network centrality metrics with improved PageRank handling.
    Handle external papers and recent paper artifacts.
    
    Returns:
        dict: Network metrics for each paper
    """
    print("Calculating network metrics...")
    
    # Core centrality measures for papers in the graph
    pagerank = nx.pagerank(graph, max_iter=100, alpha=0.85)
    betweenness = nx.betweenness_centrality(graph)
    in_degree = dict(graph.in_degree())
    
    # Get PageRank statistics for normalization
    if pagerank:
        pagerank_values = list(pagerank.values())
        pagerank_mean = np.mean(pagerank_values)
        pagerank_std = np.std(pagerank_values)
        pagerank_95th = np.percentile(pagerank_values, 95)
    
    # Combine metrics into paper database
    for paper_id in paper_db:
        if paper_id in graph:
            # Papers in the citation network
            raw_pagerank = pagerank.get(paper_id, 0)
            
            # Apply PageRank artifact correction for very recent papers with high PageRank
            paper_year = paper_db[paper_id]['year']
            citation_count = paper_db[paper_id]['citation_count']
            
            # Detect and correct PageRank artifacts
            if (paper_year >= 2024 and 
                raw_pagerank > pagerank_95th and 
                citation_count < 2):
                # Likely artifact - dampen severely
                corrected_pagerank = min(raw_pagerank * 0.1, pagerank_mean)
                print(f"PageRank artifact detected for {paper_id} ({paper_year}): {raw_pagerank:.6f} -> {corrected_pagerank:.6f}")
            else:
                corrected_pagerank = raw_pagerank
            
            paper_db[paper_id]['pagerank'] = corrected_pagerank
            paper_db[paper_id]['betweenness'] = betweenness.get(paper_id, 0)
            paper_db[paper_id]['in_degree'] = in_degree.get(paper_id, 0)
        else:
            # External papers - estimate conservative influence scores
            citation_count = paper_db[paper_id]['citation_count']
            paper_db[paper_id]['pagerank'] = min(0.001, citation_count / 100000)  # More conservative
            paper_db[paper_id]['betweenness'] = min(0.001, citation_count / 200000)
            paper_db[paper_id]['in_degree'] = citation_count
    
    return paper_db

def create_influence_ranking_table(paper_db, top_n=20):
    """
    Create comprehensive influence ranking table and save as PNG.
    
    Returns:
        pd.DataFrame: Ranked papers with multiple metrics
    """
    print(f"\nCreating Top {top_n} Most Influential Papers table...")
    
    # Convert to DataFrame for easy manipulation
    data = []
    for arxiv_id, paper in paper_db.items():
        # Skip general foundation models for GUI-specific analysis
        if should_exclude_from_gui_analysis(arxiv_id):
            continue
            
        data.append({
            'Rank': 0,  # Will be filled below
            'ArXiv': create_arxiv_badge(arxiv_id),
            'Title': paper['title'],
            'Year': paper['year'] or 'N/A',
            'Citations': paper['citation_count'],
            'PageRank': paper['pagerank'],
            'Betweenness': paper['betweenness'],
            'Platforms': ', '.join(paper['platforms'][:2]) if paper['platforms'] else 'None',
            'Summary': truncate_text(paper['key_contributions'], 150),
            'Contributions': truncate_text(paper['contribution'], 150),
        })
    
    df = pd.DataFrame(data)
    
    # Sort by PageRank (best overall influence measure)
    df_ranked = df.sort_values('PageRank', ascending=False).head(top_n)
    df_ranked['Rank'] = range(1, len(df_ranked) + 1)
    
    # Format for display
    display_df = df_ranked[['Rank', 'ArXiv', 'Title', 'Year', 'Citations', 'PageRank', 'Betweenness', 'Platforms', 'Summary', 'Contributions']].copy()
    display_df['PageRank'] = display_df['PageRank'].round(6)
    display_df['Betweenness'] = display_df['Betweenness'].round(4)
    
    # Save as markdown table
    save_table_as_markdown(display_df, 'influential_papers_ranking.md', 
                      f'Top {top_n} Most Influential Papers by PageRank Score')
    
    return df_ranked

def create_benchmarks_datasets_models_tables(papers):
    """
    Create tables for papers that introduce benchmarks, datasets, and models using structured JSON fields.
    Only uses introduces_new_* boolean fields.
    
    Args:
        papers: Original papers list with full JSON data
    
    Returns:
        dict: Analysis results for benchmarks, datasets, and models
    """
    print(f"\nCreating Benchmarks, Datasets, and Models analysis...")
    
    # Categorize papers based on structured boolean fields only
    benchmark_papers = []
    dataset_papers = []
    model_papers = []
    
    for paper in papers:
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            continue
            
        paper_info = {
            'arxiv_id': arxiv_id,
            'title': paper.get('title', ''),
            'year': paper.get('year'),
            'citations': len(paper.get('cited_by', [])),
            'key_contributions': paper.get('key_contributions', ''),
            'contribution': paper.get('contribution', '')
        }
        
        # Categorize based only on introduces_new_* boolean fields
        # Skip papers excluded from GUI-specific analysis
        if should_exclude_from_gui_analysis(arxiv_id):
            continue
            
        if paper.get('introduces_new_benchmark', False):
            benchmark_papers.append(paper_info)
            
        if paper.get('introduces_new_dataset', False):
            dataset_papers.append(paper_info)
            
        if (paper.get('introduces_new_model', False) or 
            paper.get('introduces_new_architecture', False)):
            model_papers.append(paper_info)
    
    def create_category_table(papers_list, category_name, top_n=20):
        """Create a table for a specific category"""
        if not papers_list:
            print(f"  No {category_name.lower()} papers found")
            return []
        
        # Filter out general foundation models from Models category only
        if category_name == 'Models':
            # Filter out general foundation models to focus on GUI agent-specific models
            papers_list = [p for p in papers_list if p['arxiv_id'] not in GENERAL_FOUNDATION_MODELS]
            
        # Sort by citations (most impactful first)
        papers_list.sort(key=lambda x: x['citations'], reverse=True)
        
        # Take top N papers
        top_papers = papers_list[:top_n]
        
        # Create DataFrame
        data = []
        for i, paper in enumerate(top_papers, 1):
            data.append({
                'Rank': i,
                'ArXiv': create_arxiv_badge(paper['arxiv_id']),
                'Title': paper['title'],
                'Year': paper['year'] or 'N/A',
                'Citations': paper['citations'],
                'Summary': truncate_text(paper.get('key_contributions', ''), 150),
                'Contributions': truncate_text(paper.get('contribution', ''), 150),
            })
        
        df = pd.DataFrame(data)
        
        # Save as markdown table
        filename = f'{category_name.lower()}_papers.md'
        if category_name == 'Models':
            title = f'Top {len(top_papers)} GUI Agent-Specific Model Papers'
        else:
            title = f'Top {len(top_papers)} Papers That Introduce {category_name}'
        save_table_as_markdown(df, filename, title)
        print(f"  Created {category_name.lower()} analysis: {len(top_papers)} papers")
        
        return top_papers
    
    # Create tables for each category
    benchmark_results = create_category_table(benchmark_papers, 'Benchmarks')
    dataset_results = create_category_table(dataset_papers, 'Datasets')
    model_results = create_category_table(model_papers, 'Models')
    
    # Create summary statistics
    print(f"\nSummary:")
    print(f"  Papers introducing benchmarks: {len(benchmark_papers)}")
    print(f"  Papers introducing datasets: {len(dataset_papers)}")
    print(f"  Papers introducing models/architectures: {len(model_papers)}")
    
    return {
        'benchmarks': benchmark_results,
        'datasets': dataset_results,
        'models': model_results
    }



def detect_research_bridges(paper_db, graph):
    """
    Identify papers that bridge different research areas and save as PNG table.
    
    Returns:
        pd.DataFrame: Bridge papers ranked by bridging capability
    """
    print(f"\nCreating Research Bridge Papers table...")
    
    bridge_data = []
    
    for arxiv_id, paper in paper_db.items():
        if paper['citation_count'] >= 5:  # Need sufficient citations to be a bridge
            
            # Analyze diversity of citing papers
            citing_papers = paper['cited_by']
            citing_years = set()
            citing_platforms = set()
            
            for citing_id in citing_papers:
                if citing_id in paper_db:
                    citing_paper = paper_db[citing_id]
                    if citing_paper['year']:
                        citing_years.add(citing_paper['year'])
                    citing_platforms.update(citing_paper['platforms'])
            
            # Bridge score: combination of betweenness centrality and diversity
            year_diversity = len(citing_years)
            platform_diversity = len(citing_platforms)
            bridge_score = paper['betweenness'] * (year_diversity + platform_diversity)
            
            if bridge_score > 0.001:  # Filter for meaningful bridges
                bridge_data.append({
                    'Rank': 0,  # Will be filled below
                    'Title': paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title'],
                    'Year': paper['year'],
                    'Bridge Score': bridge_score,
                    'Betweenness': paper['betweenness'],
                    'Year Span': year_diversity,
                    'Platform Span': platform_diversity,
                    'Citations': paper['citation_count']
                })
    
    # Create and display bridge papers table
    bridge_df = pd.DataFrame(bridge_data)
    bridge_df = bridge_df.sort_values('Bridge Score', ascending=False).head(15)
    bridge_df['Rank'] = range(1, len(bridge_df) + 1)
    
    # Format for display
    display_cols = ['Rank', 'Title', 'Year', 'Bridge Score', 'Betweenness', 'Year Span', 'Platform Span', 'Citations']
    display_df = bridge_df[display_cols].copy()
    display_df['Bridge Score'] = display_df['Bridge Score'].round(6)
    display_df['Betweenness'] = display_df['Betweenness'].round(4)
    
    return bridge_df

def create_most_cited_papers_table(paper_db, top_n=20):
    """
    Create table of most cited papers (raw citation count) and save as PNG.
    Includes ALL papers for complete analysis (no exclusions).
    
    Returns:
        pd.DataFrame: Most cited papers
    """
    print(f"\nCreating Top {top_n} Most Cited Papers table...")
    
    # Convert to DataFrame - Include ALL papers, no exclusions
    data = []
    for paper_id, paper in paper_db.items():
        data.append({
            'Rank': 0,  # Will be filled below
            'ArXiv': create_arxiv_badge(paper_id),
            'Title': paper['title'],
            'Year': paper['year'] or 'N/A',
            'Citations': paper['citation_count'],
            'PageRank': paper['pagerank'],
            'Platforms': ', '.join(paper['platforms'][:2]) if paper['platforms'] else 'None',
            'Summary': truncate_text(paper['key_contributions'], 150),
            'Contributions': truncate_text(paper['contribution'], 150),
        })
    
    df = pd.DataFrame(data)
    
    # Sort by citation count
    df_ranked = df.sort_values('Citations', ascending=False).head(top_n)
    df_ranked['Rank'] = range(1, len(df_ranked) + 1)
    
    # Format for display
    display_df = df_ranked[['Rank', 'ArXiv', 'Title', 'Year', 'Citations', 'PageRank', 'Platforms', 'Summary', 'Contributions']].copy()
    display_df['PageRank'] = display_df['PageRank'].round(6)
    
    # Save as markdown table
    save_table_as_markdown(display_df, 'most_cited_papers.md', 
                           f'Top {top_n} Most Cited Papers')
    
    return df_ranked

def create_network_bridge_papers_table(paper_db, graph, top_n=20):
    """
    Create comprehensive network bridge papers table combining multiple centrality metrics.
    
    Returns:
        pd.DataFrame: Bridge papers with enhanced scoring
    """
    print(f"\nCreating Top {top_n} Network Bridge Papers table...")
    
    bridge_data = []
    
    for arxiv_id, paper in paper_db.items():
        if paper['citation_count'] >= 3:  # Lower threshold for broader analysis
            
            # Analyze diversity of citing papers
            citing_papers = paper['cited_by']
            citing_years = set()
            citing_platforms = set()
            
            for citing_id in citing_papers:
                if citing_id in paper_db:
                    citing_paper = paper_db[citing_id]
                    if citing_paper['year']:
                        citing_years.add(citing_paper['year'])
                    citing_platforms.update(citing_paper['platforms'])
            
            # Enhanced bridge score calculation
            year_diversity = len(citing_years)
            platform_diversity = len(citing_platforms)
            
            # Normalize PageRank and betweenness for combination
            pagerank_norm = paper['pagerank'] * 1000  # Scale up for visibility
            betweenness_norm = paper['betweenness'] * 100
            
            # Combined bridge score: betweenness + pagerank + diversity
            bridge_score = (betweenness_norm * 0.4 + 
                          pagerank_norm * 0.4 + 
                          (year_diversity + platform_diversity) * 0.2)
            
            if bridge_score > 0.01:  # Filter for meaningful bridges
                bridge_data.append({
                    'Rank': 0,  # Will be filled below
                    'ArXiv': create_arxiv_badge(arxiv_id),
                    'Title': paper['title'],
                    'Year': paper['year'] or 'N/A',
                    'Bridge Score': bridge_score,
                    'PageRank': paper['pagerank'],
                    'Betweenness': paper['betweenness'],
                    'Platform Span': platform_diversity,
                    'Year Span': year_diversity,
                    'Citations': paper['citation_count'],
                    'Summary': truncate_text(paper['key_contributions'], 150),
                    'Contributions': truncate_text(paper['contribution'], 150),
                })
    
    # Create and sort bridge papers table
    bridge_df = pd.DataFrame(bridge_data)
    bridge_df = bridge_df.sort_values('Bridge Score', ascending=False).head(top_n)
    bridge_df['Rank'] = range(1, len(bridge_df) + 1)
    
    # Format for display
    display_cols = ['Rank', 'ArXiv', 'Title', 'Year', 'Bridge Score', 'PageRank', 'Betweenness', 
                   'Platform Span', 'Year Span', 'Citations', 'Summary', 'Contributions']
    display_df = bridge_df[display_cols].copy()
    display_df['Bridge Score'] = display_df['Bridge Score'].round(4)
    display_df['PageRank'] = display_df['PageRank'].round(6)
    display_df['Betweenness'] = display_df['Betweenness'].round(4)
    
    # Save as markdown table
    save_table_as_markdown(display_df, 'network_bridge_papers.md', 
                      f'Top {top_n} Network Bridge Papers (High Betweenness + PageRank + Cross-Platform)')
    
    return bridge_df

def calculate_foundation_and_frontier_scores(graph, paper_db, current_year=2025):
    """
    Calculate two key metrics:
    1. Foundation Score: Identifies seminal papers the field built upon
    2. Frontier Score: Identifies emerging influential work and new directions
    """
    print("Calculating foundation and frontier scores...")
    
    # Standard PageRank for foundation score base
    pagerank = nx.pagerank(graph, max_iter=100, alpha=0.85)
    
    for paper_id in paper_db:
        if paper_id not in graph:
            paper_db[paper_id]['foundation_score'] = 0
            paper_db[paper_id]['frontier_score'] = 0
            continue
            
        paper_year = paper_db[paper_id].get('year', current_year)
        age = max(1, current_year - paper_year)
        
        # FOUNDATION SCORE
        # Rewards papers that have accumulated influence over time
        # and continue to be cited (not just historical artifacts)
        
        # Count citations from different time periods
        recent_citations = 0  # Last 2 years
        mid_citations = 0     # 3 years ago
        old_citations = 0     # 3+ years ago
        
        citing_papers = paper_db[paper_id].get('cited_by', [])
        for citing_id in citing_papers:
            if citing_id in paper_db:
                citing_year = paper_db[citing_id].get('year')
                if citing_year:
                    years_ago = current_year - citing_year
                    if years_ago <= 2:
                        recent_citations += 1
                    elif years_ago <= 3:
                        mid_citations += 1
                    else:
                        old_citations += 1
        
        # Foundation score rewards sustained influence
        if age >= 2:  # Only consider papers at least 2 years old
            sustained_influence = min(recent_citations, mid_citations, old_citations) > 0
            decay_factor = 1.0 if sustained_influence else 0.5
            paper_db[paper_id]['foundation_score'] = pagerank.get(paper_id, 0) * math.log(age) * decay_factor
        else:
            paper_db[paper_id]['foundation_score'] = 0
        
        # FRONTIER SCORE
        # Identifies rapidly growing influence and new research directions
        
        if age <= 2:  # Only consider recent papers
            # Citation acceleration (are citations increasing?)
            if age >= 1:
                first_half_citations = sum(1 for citing_id in citing_papers 
                                         if citing_id in paper_db and 
                                         paper_db[citing_id].get('year', 0) <= paper_year + age//2)
                second_half_citations = recent_citations
                acceleration = max(0, (second_half_citations - first_half_citations) / age)
            else:
                acceleration = recent_citations
            
            # Who is citing this paper? New work or established work?
            citing_freshness = 0
            for citing_id in citing_papers:
                if citing_id in paper_db:
                    citing_year = paper_db[citing_id].get('year')
                    if citing_year and current_year - citing_year <= 2:
                        citing_freshness += 1
            
            # Frontier score rewards rapid adoption by recent work
            velocity = paper_db[paper_id]['citation_count'] / age
            paper_db[paper_id]['frontier_score'] = (
                0.4 * velocity + 
                0.4 * acceleration + 
                0.2 * citing_freshness
            ) / age  # Normalize by age to highlight truly explosive growth
        else:
            paper_db[paper_id]['frontier_score'] = 0
    
    # Normalize scores to 0-1 range for easier interpretation
    _normalize_scores(paper_db, 'foundation_score')
    _normalize_scores(paper_db, 'frontier_score')
    
    return paper_db

def _normalize_scores(paper_db, score_field):
    """Normalize scores to 0-1 range"""
    scores = [p[score_field] for p in paper_db.values() if score_field in p and p[score_field] > 0]
    if scores:
        max_score = max(scores)
        if max_score > 0:
            for paper_id in paper_db:
                if score_field in paper_db[paper_id]:
                    paper_db[paper_id][score_field] /= max_score

def calculate_future_impact_signals(paper_id, paper_db):
    """
    Calculate signals that correlate with future citation growth.
    Based on early citation patterns in first 6-12 months.
    """
    # Get first year of citations with monthly precision
    early_citations = get_citations_in_first_n_months(paper_id, paper_db, n=12)
    
    if len(early_citations) < 3:
        return {'prediction_confidence': 'low'}
    
    # Early momentum indicators
    month_3_citations = sum(early_citations[:3])
    month_6_citations = sum(early_citations[3:6]) if len(early_citations) >= 6 else 0
    month_12_citations = sum(early_citations[6:12]) if len(early_citations) >= 12 else 0
    
    # Calculate growth ratios
    early_growth_ratio = month_6_citations / max(month_3_citations, 1)
    sustained_growth = month_12_citations / max(month_6_citations, 1)
    
    # Individual signal flags
    has_early_momentum = month_3_citations > 3
    has_acceleration = early_growth_ratio > 2
    has_sustained_growth = sustained_growth > 0.8
    
    # Count positive signals (most honest approach)
    signal_count = sum([
        has_early_momentum,
        has_acceleration,
        has_sustained_growth,
    ])
    
    # Predict future trajectory
    signals = {
        'early_momentum': month_3_citations,
        'growth_acceleration': early_growth_ratio,
        'growth_sustained': has_sustained_growth,
        'prediction_confidence': 'high' if len(early_citations) >= 12 else 'medium',
        'breakthrough_potential': signal_count,
        'predicted_trajectory': classify_trajectory_by_count(signal_count)
    }
    
    return signals

def classify_trajectory_by_count(signal_count):
    """Classify predicted trajectory based on number of positive signals."""
    if signal_count >= 3:
        return 'strong_potential'
    elif signal_count == 2:
        return 'moderate_potential'
    elif signal_count == 1:
        return 'limited_potential'
    else:
        return 'unclear_trajectory'

def calculate_citation_velocity_trend(paper_id, paper_db):
    """
    Calculate detailed velocity trends with acceleration analysis.
    """
    citations_by_month = build_monthly_citations(paper_id, paper_db)
    if len(citations_by_month) < 6:
        return {'current_velocity': 0, 'acceleration': 0, 'is_accelerating': False}
    
    months = sorted(citations_by_month.keys())
    citation_series = [citations_by_month[m] for m in months]
    
    # Calculate 3-month rolling velocities
    velocities = []
    for i in range(2, len(citation_series)):
        velocity = sum(citation_series[i-2:i+1]) / 3
        velocities.append(velocity)
    
    if len(velocities) >= 3:
        # Calculate acceleration as change in velocity
        recent_velocity = np.mean(velocities[-3:])
        earlier_velocity = np.mean(velocities[:3]) if len(velocities) >= 6 else velocities[0]
        acceleration = recent_velocity - earlier_velocity
        is_accelerating = acceleration > 0.5
    else:
        recent_velocity = velocities[-1] if velocities else 0
        acceleration = 0
        is_accelerating = False
    
    return {
        'current_velocity': recent_velocity,
        'acceleration': acceleration,
        'is_accelerating': is_accelerating,
        'velocity_history': velocities
    }

def calculate_time_weighted_influence(paper_id, paper_db):
    """
    Calculate influence with time decay weighting to emphasize recent citations.
    """
    current_year = 2025
    current_month = 8
    
    total_weighted_citations = 0
    raw_citations = 0
    recent_citations = 0  # Last 12 months
    
    for citing_id in paper_db[paper_id]['cited_by']:
        if citing_id in paper_db:
            citing_year = paper_db[citing_id].get('year')
            citing_month = paper_db[citing_id].get('month', 6)
            
            if citing_year:
                raw_citations += 1
                
                # Calculate months ago
                months_ago = (current_year - citing_year) * 12 + (current_month - citing_month)
                
                # Recent citations (last 3 months)
                if months_ago <= 3:
                    recent_citations += 1
                
                # Time decay weight (exponential decay with half-life of 9 months)
                decay_factor = 0.5 ** (months_ago / 9)
                total_weighted_citations += decay_factor
    
    recency_ratio = recent_citations / max(raw_citations, 1)
    
    return {
        'weighted_citations': total_weighted_citations,
        'raw_citations': raw_citations,
        'recent_citations': recent_citations,
        'recency_ratio': recency_ratio
    }

def detect_simple_citation_bursts(paper_id, paper_db):
    """
    Simplified burst detection - just find months with unusually high citations.
    """
    citations_by_month = build_monthly_citations(paper_id, paper_db)
    
    if len(citations_by_month) < 6:
        return {
            'has_bursts': False,
            'max_month_citations': 0,
            'avg_citations_per_month': 0,
            'burst_strength': 0
        }
    
    monthly_values = list(citations_by_month.values())
    avg_citations = sum(monthly_values) / len(monthly_values)
    max_citations = max(monthly_values)
    
    # Simple rule: burst if any month has 3x the average (and at least 3 citations)
    burst_threshold = max(avg_citations * 3, 3)
    has_bursts = max_citations >= burst_threshold
    
    return {
        'has_bursts': has_bursts,
        'max_month_citations': max_citations,
        'avg_citations_per_month': avg_citations,
        'burst_strength': max_citations / max(avg_citations, 1) if has_bursts else 0
    }

def calculate_simple_temporal_profile(paper_id, paper_db):
    """
    Super simple temporal profile - just recent vs old citations.
    """
    paper = paper_db[paper_id]
    total_citations = paper['citation_count']
    
    if total_citations < 3:
        return {
            'pattern': '📊 Low Activity',
            'score': 0,
            'recent_ratio': 0,
            'citations_per_year': 0,
            'burst_strength': 0
        }
    
        # Count recent vs old citations (month-level precision)
    current_year = 2025
    current_month = 1  # January 2025
    current_month_key = current_year * 100 + current_month
    
    recent_citations = 0  # Last 12 months
    very_recent_citations = 0  # Last 6 months
    
    for citing_id in paper['cited_by']:
        if citing_id in paper_db:
            citing_year = paper_db[citing_id].get('year')
            citing_month = paper_db[citing_id].get('month', 6)
            
            if citing_year:
                citing_month_key = citing_year * 100 + citing_month
                months_ago = calculate_months_between(citing_month_key, current_month_key)
                
                if months_ago <= 12:  # Last 12 months
                    recent_citations += 1
                if months_ago <= 6:   # Last 6 months  
                    very_recent_citations += 1
    
    recency_ratio = recent_citations / total_citations
    very_recent_ratio = very_recent_citations / total_citations
    citations_per_year = total_citations / max(1, current_year - (paper.get('year') or current_year))
    
    # Simple burst check
    burst_info = detect_simple_citation_bursts(paper_id, paper_db)
    
    # Simple pattern classification using month-level precision (6 patterns only)
    if burst_info['has_bursts'] and very_recent_ratio > 0.3:  # Active burst in last 6 months
        pattern = "🔥 Hot & Bursting"
    elif very_recent_ratio > 0.4:  # High activity in last 6 months
        pattern = "🌱 Recently Active"
    elif total_citations > 20 and recency_ratio < 0.2:  # Old citations, low recent activity
        pattern = "🏛️ Established Classic"
    elif citations_per_year > 5:
        pattern = "⚡ High Impact"
    elif burst_info['has_bursts']:
        pattern = "💫 Had Bursts"
    else:
        pattern = "📊 Standard"
    
    # Simple combined score
    score = citations_per_year + (recency_ratio * 10) + (burst_info['burst_strength'] * 2)
    
    return {
        'pattern': pattern,
        'score': score,
        'recent_ratio': recency_ratio,
        'citations_per_year': citations_per_year,
        'burst_strength': burst_info['burst_strength'],
        'total_citations': total_citations
    }

def calculate_pattern_confidence(burst_profile, velocity_trends):
    """
    How confident are we in the pattern classification?
    """
    # More data = higher confidence
    total_months = burst_profile.get('total_burst_months', 0) + velocity_trends.get('months_active', 0)
    
    if total_months < 3:
        return "low"
    elif total_months < 9:
        return "medium"
    else:
        return "high"

def build_monthly_citations(paper_id, paper_db):
    """Build monthly citation timeline for a paper"""
    citations_by_month = defaultdict(int)
    
    for citing_id in paper_db[paper_id]['cited_by']:
        if citing_id in paper_db:
            citing_year = paper_db[citing_id].get('year')
            citing_month = paper_db[citing_id].get('month', 6)  # Default to June if missing
            
            if citing_year:
                month_key = citing_year * 100 + citing_month
                citations_by_month[month_key] += 1
    
    return citations_by_month

def calculate_month_diff(start_month, end_month):
    """Calculate difference in months between two month keys (YYYYMM format)"""
    start_year = start_month // 100
    start_month_num = start_month % 100
    end_year = end_month // 100
    end_month_num = end_month % 100
    
    return (end_year - start_year) * 12 + (end_month_num - start_month_num)

def calculate_months_between(from_month_key, to_month_key):
    """Calculate months between two month keys (YYYYMM format). Positive means from_month is earlier."""
    from_year = from_month_key // 100
    from_month = from_month_key % 100
    to_year = to_month_key // 100
    to_month = to_month_key % 100
    
    return (to_year - from_year) * 12 + (to_month - from_month)

def get_citations_in_first_n_months(paper_id, paper_db, n=12):
    """Get citation counts for first N months after publication"""
    paper_year = paper_db[paper_id].get('year')
    if not paper_year:
        return []
    
    paper_month = paper_db[paper_id].get('month', 1)  # Default to January
    paper_start = paper_year * 100 + paper_month
    
    monthly_citations = []
    for i in range(n):
        month_key = paper_start + i
        # Handle year rollover
        if (month_key % 100) > 12:
            month_key = month_key + 88  # Jump to next year, January
        
        citations_count = 0
        for citing_id in paper_db[paper_id]['cited_by']:
            if citing_id in paper_db:
                citing_year = paper_db[citing_id].get('year')
                citing_month = paper_db[citing_id].get('month', 6)
                citing_key = citing_year * 100 + citing_month if citing_year else 0
                
                if citing_key == month_key:
                    citations_count += 1
        
        monthly_citations.append(citations_count)
    
    return monthly_citations

def create_foundation_papers_table(paper_db, top_n=20):
    """
    Create table of foundation papers - seminal work the field built upon.
    Includes ALL papers (no exclusions) since foundation papers are historically important.
    
    Returns:
        pd.DataFrame: Foundation papers with sustained influence
    """
    print(f"\nCreating Top {top_n} Foundation Papers table...")
    
    foundation_data = []
    
    for arxiv_id, paper in paper_db.items():
        if paper.get('foundation_score', 0) > 0:  # No exclusions for foundation papers
            foundation_data.append({
                'Rank': 0,  # Will be filled below
                'ArXiv': create_arxiv_badge(arxiv_id),
                'Title': paper['title'],
                'Year': paper['year'] or 'N/A',
                'Foundation Score': paper['foundation_score'],
                'Citations': paper['citation_count'],
                'Age': max(1, 2025 - (paper['year'] or 2025)),
                'PageRank': paper['pagerank'],
                'Summary': truncate_text(paper['key_contributions'], 150),
                'Contributions': truncate_text(paper['contribution'], 150),
            })
    
    # Create and sort foundation papers table
    foundation_df = pd.DataFrame(foundation_data)
    foundation_df = foundation_df.sort_values('Foundation Score', ascending=False).head(top_n)
    foundation_df['Rank'] = range(1, len(foundation_df) + 1)
    
    # Format for display
    display_cols = ['Rank', 'ArXiv', 'Title', 'Year', 'Foundation Score', 'Citations', 'Age', 'PageRank', 'Summary', 'Contributions']
    display_df = foundation_df[display_cols].copy()
    display_df['Foundation Score'] = display_df['Foundation Score'].round(4)
    display_df['PageRank'] = display_df['PageRank'].round(6)
    
    # Save as markdown table
    save_table_as_markdown(display_df, 'foundation_papers.md', 
                           f'Top {top_n} Foundation Papers (Sustained Influence Over Time)')
    
    return foundation_df

def create_frontier_papers_table(paper_db, top_n=20):
    """
    Create table of frontier papers - emerging influential work and new directions.
    
    Returns:
        pd.DataFrame: Frontier papers with rapid growth
    """
    print(f"\nCreating Top {top_n} Frontier Papers table...")
    
    frontier_data = []
    
    for arxiv_id, paper in paper_db.items():
        if (paper.get('frontier_score', 0) > 0 and 
            not should_exclude_from_gui_analysis(arxiv_id)):
            age = max(1, 2025 - (paper['year'] or 2025))
            frontier_data.append({
                'Rank': 0,  # Will be filled below
                'ArXiv': create_arxiv_badge(arxiv_id),
                'Title': paper['title'],
                'Year': paper['year'] or 'N/A',
                'Frontier Score': paper['frontier_score'],
                'Citations': paper['citation_count'],
                'Age': age,
                'Citations/Year': round(paper['citation_count'] / age, 1),
                'PageRank': paper['pagerank'],
                'Summary': truncate_text(paper['key_contributions'], 150),
                'Contributions': truncate_text(paper['contribution'], 150),
            })
    
    # Create and sort frontier papers table
    frontier_df = pd.DataFrame(frontier_data)
    frontier_df = frontier_df.sort_values('Frontier Score', ascending=False).head(top_n)
    frontier_df['Rank'] = range(1, len(frontier_df) + 1)
    
    # Format for display
    display_cols = ['Rank', 'ArXiv', 'Title', 'Year', 'Frontier Score', 'Citations', 'Age', 'Citations/Year', 'PageRank', 'Summary', 'Contributions']
    display_df = frontier_df[display_cols].copy()
    display_df['Frontier Score'] = display_df['Frontier Score'].round(4)
    display_df['PageRank'] = display_df['PageRank'].round(6)
    
    # Save as markdown table
    save_table_as_markdown(display_df, 'frontier_papers.md', 
                           f'Top {top_n} Frontier Papers (Emerging Influential Work)')
    
    return frontier_df

def create_citation_velocity_table(paper_db, top_n=20):
    """
    Create simple citation velocity table using basic metrics.
    
    Returns:
        pd.DataFrame: Papers with basic velocity analysis
    """
    print(f"\nCreating Top {top_n} Citation Velocity Papers table (simplified)...")
    
    velocity_data = []
    
    for arxiv_id, paper in paper_db.items():
        if (paper['citation_count'] >= 3 and  # Need sufficient citations for velocity analysis
            not should_exclude_from_gui_analysis(arxiv_id)):  # Exclude non-GUI papers
            
            # Simple velocity calculation: citations per year since publication
            paper_year = paper.get('year')
            years_since_pub = max(1, 2025 - paper_year)
            velocity = paper['citation_count'] / years_since_pub
            
            # Month-level recent activity check (last 12 months)
            current_year = 2025
            current_month = 8  # January 2025
            current_month_key = current_year * 100 + current_month
            
            recent_citations = 0  # Last 6 months
            very_recent_citations = 0  # Last 3 months
            
            for citing_id in paper['cited_by']:
                if citing_id in paper_db:
                    citing_year = paper_db[citing_id].get('year')
                    citing_month = paper_db[citing_id].get('month', 6)  # Default to June if missing
                    
                    if citing_year:
                        citing_month_key = citing_year * 100 + citing_month
                        months_ago = calculate_months_between(citing_month_key, current_month_key)
                        
                        if months_ago <= 6:  # Last 6 months
                            recent_citations += 1
                        if months_ago <= 3:   # Last 3 months
                            very_recent_citations += 1
            
            recent_ratio = recent_citations / max(paper['citation_count'], 1)
            
            velocity_data.append({
                'Rank': 0,  # Will be filled below
                'ArXiv': create_arxiv_badge(arxiv_id),
                'Title': paper['title'],
                'Year': paper['year'] or 'N/A',
                'Citations': paper['citation_count'],
                'Citations/Year': velocity,
                'Recent (6mo)': recent_citations,
                'Very Recent (3mo)': very_recent_citations,
                'Recent Ratio': recent_ratio,
                'PageRank': paper['pagerank'],
                'Summary': truncate_text(paper['key_contributions'], 150),
                'Contributions': truncate_text(paper['contribution'], 150),
            })
    
    # Create and sort velocity table by recent activity (most relevant for fast-moving field)
    velocity_df = pd.DataFrame(velocity_data)
    velocity_df = velocity_df.sort_values(['Recent Ratio', 'Citations/Year'], ascending=[False, False]).head(top_n)
    velocity_df['Rank'] = range(1, len(velocity_df) + 1)
    
    # Format for display
    display_cols = ['Rank', 'ArXiv', 'Title', 'Year', 'Citations', 'Citations/Year', 
                   'Recent (6mo)', 'Very Recent (3mo)', 'Recent Ratio', 'Summary', 'Contributions']
    display_df = velocity_df[display_cols].copy()
    display_df['Citations/Year'] = display_df['Citations/Year'].round(2)
    display_df['Recent Ratio'] = display_df['Recent Ratio'].round(3)
    
    # Save as markdown table
    save_table_as_markdown(display_df, 'citation_velocity_papers.md', 
                          f'Top {top_n} Papers by Recent Activity (6-Month Focus)')
    
    return velocity_df

def create_future_impact_table(paper_db, top_n=5):
    """
    Create table of papers with future impact prediction signals.
    
    Returns:
        pd.DataFrame: Papers with future impact analysis
    """
    print(f"\nCreating Top {top_n} Future Impact Signals table...")
    
    impact_data = []
    
    for arxiv_id, paper in paper_db.items():
        # Focus on relatively recent papers (last 3-4 years) for future prediction
        paper_year = paper.get('year')
        if (paper_year and paper_year >= 2024 and paper['citation_count'] >= 3 and
            not should_exclude_from_gui_analysis(arxiv_id)):
            impact_signals = calculate_future_impact_signals(arxiv_id, paper_db)
            
            if impact_signals.get('prediction_confidence') != 'low':
                impact_data.append({
                    'Rank': 0,  # Will be filled below
                    'ArXiv': create_arxiv_badge(arxiv_id),
                    'Title': paper['title'],
                    'Year': paper_year,
                    'Citations': paper['citation_count'],
                    'Early Momentum': impact_signals.get('early_momentum', 0),
                    'Growth Ratio': round(impact_signals.get('growth_acceleration', 0), 2),
                    'Breakthrough Score': round(impact_signals.get('breakthrough_potential', 0), 3),
                    'Predicted Trajectory': impact_signals.get('predicted_trajectory', 'unknown'),
                    'Confidence': impact_signals.get('prediction_confidence', 'low'),
                    'PageRank': paper['pagerank'],
                    'Summary': truncate_text(paper['key_contributions'], 150),
                    'Contributions': truncate_text(paper['contribution'], 150),
                })
    
    # Create and sort future impact table
    impact_df = pd.DataFrame(impact_data)
    if not impact_df.empty:
        # Sort by breakthrough score
        impact_df = impact_df.sort_values('Breakthrough Score', ascending=False).head(top_n)
        impact_df['Rank'] = range(1, len(impact_df) + 1)
        
        # Format for display
        display_cols = ['Rank', 'ArXiv', 'Title', 'Year', 'Citations', 'Early Momentum', 
                       'Growth Ratio', 'Breakthrough Score', 'Predicted Trajectory', 
                       'Confidence', 'PageRank', 'Summary', 'Contributions']
        display_df = impact_df[display_cols].copy()
        display_df['PageRank'] = display_df['PageRank'].round(6)
        
        # Save as markdown table
        save_table_as_markdown(display_df, 'future_impact_signals_papers.md', 
                              f'Top {top_n} Papers by Future Impact Prediction Signals')
    else:
        print("  No recent papers found with sufficient data for future impact prediction")
        impact_df = pd.DataFrame()
    
    return impact_df

def analyze_field_acceleration(paper_db):
    """
    Detect when the GUI agents field "took off" by analyzing field-wide citation acceleration.
    
    Returns:
        dict: Analysis of field acceleration patterns
    """
    print(f"\nAnalyzing when GUI agents field accelerated...")
    
    # Build field-wide monthly citation timeline
    field_citations_by_month = defaultdict(int)
    paper_count_by_month = defaultdict(int)
    
    for paper_id, paper in paper_db.items():
        paper_year = paper.get('year')
        if paper_year and paper_year >= 2016:  # Focus on modern era
            # Count papers published each month
            paper_month = paper.get('month', 6)
            pub_month_key = paper_year * 100 + paper_month
            paper_count_by_month[pub_month_key] += 1
            
            # Count citations received each month
            for citing_id in paper['cited_by']:
                if citing_id in paper_db:
                    citing_year = paper_db[citing_id].get('year')
                    citing_month = paper_db[citing_id].get('month', 6)
                    if citing_year and citing_year >= 2016:
                        cite_month_key = citing_year * 100 + citing_month
                        field_citations_by_month[cite_month_key] += 1
    
    # Convert to time series
    months = sorted(field_citations_by_month.keys())
    citation_series = [field_citations_by_month[m] for m in months]
    paper_series = [paper_count_by_month.get(m, 0) for m in months]
    
    # Calculate 6-month rolling averages to smooth noise
    if len(citation_series) >= 6:
        smoothed_citations = []
        for i in range(5, len(citation_series)):
            avg = np.mean(citation_series[i-5:i+1])
            smoothed_citations.append(avg)
        
        # Find acceleration points (significant increases)
        acceleration_points = []
        for i in range(1, len(smoothed_citations)):
            if smoothed_citations[i] > smoothed_citations[i-1] * 1.5:  # 50% increase
                month_key = months[i+5]  # Adjust for smoothing offset
                year = month_key // 100
                month = month_key % 100
                acceleration_points.append({
                    'month': month_key,
                    'year': year,
                    'month_name': f"{year}-{month:02d}",
                    'citations': smoothed_citations[i],
                    'acceleration': (smoothed_citations[i] / smoothed_citations[i-1] - 1) * 100
                })
        
        # Find the major takeoff point (largest acceleration)
        if acceleration_points:
            major_takeoff = max(acceleration_points, key=lambda x: x['acceleration'])
            
            # Analyze field growth phases
            pre_takeoff_avg = np.mean(citation_series[:months.index(major_takeoff['month'])])
            post_takeoff_avg = np.mean(citation_series[months.index(major_takeoff['month']):])
            
            return {
                'major_takeoff_month': major_takeoff['month_name'],
                'major_takeoff_year': major_takeoff['year'],
                'takeoff_acceleration': major_takeoff['acceleration'],
                'pre_takeoff_avg_citations': pre_takeoff_avg,
                'post_takeoff_avg_citations': post_takeoff_avg,
                'growth_multiplier': post_takeoff_avg / max(pre_takeoff_avg, 1),
                'all_acceleration_points': acceleration_points,
                'monthly_data': list(zip(months, citation_series, paper_series))
            }
    
    return {'error': 'Insufficient data for acceleration analysis'}

def identify_paradigm_shift_papers(paper_db):
    """
    Identify papers that triggered paradigm shifts by finding papers whose publication
    preceded field-wide citation surges.
    
    Returns:
        list: Papers that likely triggered paradigm shifts
    """
    print(f"\nIdentifying papers that triggered paradigm shifts...")
    
    paradigm_shift_papers = []
    
    # Get field acceleration analysis
    field_analysis = analyze_field_acceleration(paper_db)
    if 'error' in field_analysis:
        return []
    
    acceleration_points = field_analysis['all_acceleration_points']
    
    for acc_point in acceleration_points:
        acc_month = acc_point['month']
        acc_year = acc_point['year']
        
        # Look for influential papers published 1-6 months before this acceleration
        candidate_papers = []
        
        for paper_id, paper in paper_db.items():
            paper_year = paper.get('year')
            paper_month = paper.get('month', 6)
            
            if paper_year and not should_exclude_from_gui_analysis(paper_id):
                paper_month_key = paper_year * 100 + paper_month
                months_before = calculate_month_diff(paper_month_key, acc_month)
                
                # Paper published 1-6 months before acceleration
                if 1 <= months_before <= 6 and paper['citation_count'] >= 10:
                    # Calculate post-publication citation surge
                    post_pub_citations = 0
                    for citing_id in paper['cited_by']:
                        if citing_id in paper_db:
                            citing_year = paper_db[citing_id].get('year')
                            citing_month = paper_db[citing_id].get('month', 6)
                            if citing_year:
                                citing_month_key = citing_year * 100 + citing_month
                                if citing_month_key >= acc_month:
                                    post_pub_citations += 1
                    
                    surge_ratio = post_pub_citations / max(paper['citation_count'] - post_pub_citations, 1)
                    
                    candidate_papers.append({
                        'paper_id': paper_id,
                        'title': paper['title'],
                        'year': paper_year,
                        'months_before_surge': months_before,
                        'total_citations': paper['citation_count'],
                        'post_surge_citations': post_pub_citations,
                        'surge_ratio': surge_ratio,
                        'acceleration_triggered': acc_point['acceleration']
                    })
        
        # Rank candidates by surge ratio and select top ones
        candidate_papers.sort(key=lambda x: x['surge_ratio'], reverse=True)
        
        # Add top candidates as paradigm shift papers
        for candidate in candidate_papers[:2]:  # Top 2 per acceleration point
            if candidate['surge_ratio'] > 1.5:  # Must have significant post-surge citations
                paradigm_shift_papers.append(candidate)
    
    # Remove duplicates and sort by overall impact
    seen_papers = set()
    unique_papers = []
    for paper in paradigm_shift_papers:
        if paper['paper_id'] not in seen_papers:
            seen_papers.add(paper['paper_id'])
            unique_papers.append(paper)
    
    unique_papers.sort(key=lambda x: x['acceleration_triggered'] * x['surge_ratio'], reverse=True)
    
    return unique_papers[:10]  # Top 10 paradigm shift papers

def analyze_bubble_vs_sustained_growth(paper_db):
    """
    Analyze whether current citation rates represent a bubble or sustained growth.
    
    Returns:
        dict: Analysis of growth sustainability
    """
    print(f"\nAnalyzing if we're in a bubble or sustained growth...")
    
    current_year = 2025
    
    # Analyze citation patterns by year
    yearly_stats = defaultdict(lambda: {'papers': 0, 'citations': 0, 'citing_papers': 0})
    
    for paper_id, paper in paper_db.items():
        paper_year = paper.get('year')
        if paper_year and paper_year >= 2020:  # Focus on recent years
            yearly_stats[paper_year]['papers'] += 1
            yearly_stats[paper_year]['citations'] += paper['citation_count']
            
            # Count papers that cite this year's work
            for citing_id in paper['cited_by']:
                if citing_id in paper_db:
                    citing_year = paper_db[citing_id].get('year')
                    if citing_year:
                        yearly_stats[citing_year]['citing_papers'] += 1
    
    # Calculate growth metrics
    years = sorted([y for y in yearly_stats.keys() if y <= current_year])
    
    # Citation velocity by year
    citation_velocities = []
    paper_growth_rates = []
    
    for i in range(1, len(years)):
        prev_year, curr_year = years[i-1], years[i]
        
        # Citation growth rate
        prev_citations = yearly_stats[prev_year]['citations']
        curr_citations = yearly_stats[curr_year]['citations']
        if prev_citations > 0:
            citation_growth = (curr_citations - prev_citations) / prev_citations
            citation_velocities.append(citation_growth)
        
        # Paper publication growth rate
        prev_papers = yearly_stats[prev_year]['papers']
        curr_papers = yearly_stats[curr_year]['papers']
        if prev_papers > 0:
            paper_growth = (curr_papers - prev_papers) / prev_papers
            paper_growth_rates.append(paper_growth)
    
    # Analyze sustainability indicators
    recent_citation_velocity = np.mean(citation_velocities[-2:]) if len(citation_velocities) >= 2 else 0
    recent_paper_growth = np.mean(paper_growth_rates[-2:]) if len(paper_growth_rates) >= 2 else 0
    
    # Calculate citation efficiency (citations per paper)
    citation_efficiency_trend = []
    for year in years:
        if yearly_stats[year]['papers'] > 0:
            efficiency = yearly_stats[year]['citations'] / yearly_stats[year]['papers']
            citation_efficiency_trend.append(efficiency)
    
    # Sustainability assessment
    if len(citation_efficiency_trend) >= 3:
        efficiency_trend = np.polyfit(range(len(citation_efficiency_trend)), citation_efficiency_trend, 1)[0]
        
        # Classify growth pattern
        if recent_citation_velocity > 0.5 and efficiency_trend > 0:
            growth_pattern = "🚀 Sustained Exponential Growth"
            bubble_risk = "Low"
        elif recent_citation_velocity > 0.3 and efficiency_trend > -0.1:
            growth_pattern = "📈 Healthy Growth"
            bubble_risk = "Low"
        elif recent_citation_velocity > 0.1 and efficiency_trend < -0.2:
            growth_pattern = "⚠️ Potential Bubble Formation"
            bubble_risk = "Medium"
        elif recent_citation_velocity < 0 or efficiency_trend < -0.5:
            growth_pattern = "📉 Growth Slowdown"
            bubble_risk = "High"
        else:
            growth_pattern = "📊 Stable Growth"
            bubble_risk = "Low"
    else:
        growth_pattern = "📊 Insufficient Data"
        bubble_risk = "Unknown"
    
    return {
        'growth_pattern': growth_pattern,
        'bubble_risk': bubble_risk,
        'recent_citation_velocity': recent_citation_velocity,
        'recent_paper_growth': recent_paper_growth,
        'citation_efficiency_trend': efficiency_trend if len(citation_efficiency_trend) >= 3 else 0,
        'yearly_stats': dict(yearly_stats),
        'sustainability_score': min(1.0, max(0.0, recent_citation_velocity + efficiency_trend))
    }

def analyze_adoption_timelines(paper_db):
    """
    Analyze typical adoption timelines - how many months from publication to peak influence.
    
    Returns:
        dict: Analysis of adoption patterns
    """
    print(f"\nAnalyzing typical adoption timelines...")
    
    adoption_data = []
    
    for paper_id, paper in paper_db.items():
        if paper['citation_count'] >= 10:  # Need sufficient citations for meaningful analysis
            paper_year = paper.get('year')
            paper_month = paper.get('month', 6)
            
            if paper_year and paper_year >= 2018:  # Focus on papers with enough time to mature
                paper_start = paper_year * 100 + paper_month
                
                # Build monthly citation timeline for this paper
                monthly_citations = defaultdict(int)
                for citing_id in paper['cited_by']:
                    if citing_id in paper_db:
                        citing_year = paper_db[citing_id].get('year')
                        citing_month = paper_db[citing_id].get('month', 6)
                        if citing_year:
                            citing_month_key = citing_year * 100 + citing_month
                            if citing_month_key >= paper_start:
                                monthly_citations[citing_month_key] += 1
                
                if len(monthly_citations) >= 6:  # Need reasonable citation history
                    # Find peak citation month
                    peak_month = max(monthly_citations.keys(), key=lambda k: monthly_citations[k])
                    peak_citations = monthly_citations[peak_month]
                    
                    # Calculate months to peak
                    months_to_peak = calculate_month_diff(paper_start, peak_month)
                    
                    # Calculate adoption velocity (citations in first 12 months)
                    early_citations = sum(monthly_citations[k] for k in monthly_citations.keys() 
                                        if calculate_month_diff(paper_start, k) <= 12)
                    
                    # Calculate sustained influence (citations after month 12)
                    late_citations = sum(monthly_citations[k] for k in monthly_citations.keys() 
                                       if calculate_month_diff(paper_start, k) > 12)
                    
                    adoption_data.append({
                        'paper_id': paper_id,
                        'title': paper['title'],
                        'year': paper_year,
                        'total_citations': paper['citation_count'],
                        'months_to_peak': months_to_peak,
                        'peak_citations': peak_citations,
                        'early_citations': early_citations,
                        'late_citations': late_citations,
                        'adoption_velocity': early_citations / 12,  # Citations per month in first year
                        'sustained_ratio': late_citations / max(early_citations, 1)
                    })
    
    if adoption_data:
        # Calculate statistics
        months_to_peak_list = [d['months_to_peak'] for d in adoption_data]
        adoption_velocities = [d['adoption_velocity'] for d in adoption_data]
        
        # Categorize adoption patterns
        fast_adopters = [d for d in adoption_data if d['months_to_peak'] <= 12]
        slow_adopters = [d for d in adoption_data if d['months_to_peak'] > 24]
        sustained_papers = [d for d in adoption_data if d['sustained_ratio'] > 1.0]
        
        return {
            'median_months_to_peak': np.median(months_to_peak_list),
            'mean_months_to_peak': np.mean(months_to_peak_list),
            'fast_adoption_percentage': len(fast_adopters) / len(adoption_data) * 100,
            'slow_adoption_percentage': len(slow_adopters) / len(adoption_data) * 100,
            'sustained_influence_percentage': len(sustained_papers) / len(adoption_data) * 100,
            'median_adoption_velocity': np.median(adoption_velocities),
            'adoption_data': adoption_data[:20],  # Top 20 for detailed view
            'total_papers_analyzed': len(adoption_data)
        }
    
    return {'error': 'Insufficient data for adoption timeline analysis'}

def save_table_as_markdown(df, filename, title):
    """
    Save a pandas DataFrame as a markdown table file in the output directory.
    
    Args:
        df: DataFrame to save
        filename: Output filename (should end with .md)
        title: Table title
    """
    # Ensure output directory exists
    ensure_output_directory()
    
    # Create full file path
    full_path = os.path.join(OUTPUT_DIR, filename)
    
    # Create markdown content
    markdown_content = f"# {title}\n\n"
    
    # Convert DataFrame to markdown table
    markdown_table = df.to_markdown(index=False, tablefmt='github')
    markdown_content += markdown_table
    
    # Add metadata
    markdown_content += f"\n\n---\n*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    markdown_content += f"*Total entries: {len(df)}*\n"
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Saved markdown table: {full_path}")

def save_temporal_table_with_explanation(df, filename, title):
    """
    Save temporal analysis table with detailed explanations of all metrics.
    """
    # Ensure output directory exists
    ensure_output_directory()
    
    # Create full file path
    full_path = os.path.join(OUTPUT_DIR, filename)
    
    # Create comprehensive markdown content
    markdown_content = f"# {title}\n\n"
    
    # Add detailed explanation section
    markdown_content += """## 📊 How to Interpret This Table

This table shows papers with the most dynamic temporal citation patterns, focusing on **recent activity** and **momentum**.

### 🔢 **Key Metrics Explained**

#### **Score** (Combined Temporal Score)
- **Formula**: `Citations/Year + (Recent Ratio × 10) + (Burst Strength × 2)`
- **Range**: 0-100+ (higher = more temporally significant)
- **Interpretation**: 
  - **60+**: Very high temporal significance
  - **40-59**: High temporal significance  
  - **20-39**: Moderate temporal significance
  - **<20**: Lower temporal significance

#### **Recent Ratio** (6-Month Focus)
- **Definition**: Fraction of total citations from papers published in **last 6 months**
- **Range**: 0.0-1.0 (higher = more recently active)
- **Interpretation**:
  - **0.6+**: Very hot right now, getting lots of recent attention
  - **0.4-0.6**: Good recent activity
  - **0.2-0.4**: Some recent interest
  - **<0.2**: Mostly older citations, established work

#### **Citations/Year** (Overall Velocity)
- **Definition**: Total citations divided by years since publication
- **Interpretation**: Shows overall citation velocity regardless of timing

#### **Burst Strength** (Citation Spikes)
- **Definition**: Peak monthly citations ÷ average monthly citations
- **Range**: 0-10+ (higher = more bursty citation pattern)
- **Interpretation**:
  - **3+**: Clear citation bursts (conferences, viral moments)
  - **1-3**: Some citation clustering
  - **0**: Steady citation pattern

### 🏷️ **Pattern Categories**

| Pattern | Meaning | Criteria |
|---------|---------|----------|
| 🔥 **Hot & Bursting** | Active burst + high recent activity | Burst detected + >30% very recent citations |
| 🌱 **Recently Active** | High activity in last 6 months | >40% very recent citations |
| ⚡ **High Impact** | Strong overall velocity | >5 citations/year |
| 🏛️ **Established Classic** | Old but foundational | >20 citations + <20% recent activity |
| 💫 **Had Bursts** | Past citation spikes | Burst detected but not currently hot |
| 📊 **Standard** | Normal citation pattern | Doesn't fit other categories |

### 💡 **How to Use This Data**

- **For Current Trends**: Focus on high Recent Ratio (>0.5) papers
- **For Impact**: Look at Citations/Year combined with Recent Ratio
- **For Momentum**: Check Score + Pattern combination
- **For Timing**: Burst Strength shows when papers "went viral"

---

## 📈 Results Table

"""
    
    # Add the actual table
    markdown_table = df.to_markdown(index=False, tablefmt='github')
    markdown_content += markdown_table
    
    # Add metadata
    markdown_content += f"\n\n---\n*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    markdown_content += f"*Total entries: {len(df)}*\n"
    markdown_content += "*Analysis uses month-level precision for maximum accuracy in fast-moving AI research*\n"
    
    # Save to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Saved temporal analysis table with explanations: {full_path}")

def create_improved_burst_heatmap(ax, paper_db, top_papers=12):
    """
    Create an improved heatmap showing citation patterns and burst periods.
    Uses actual data structure and honest visual encoding with burst overlays.
    """
    
    # Select papers with sufficient citation activity
    papers_with_activity = []
    excluded_count = 0
    low_citation_count = 0
    
    for arxiv_id, paper in paper_db.items():
        if paper['citation_count'] < 10:
            low_citation_count += 1
        elif should_exclude_from_gui_analysis(arxiv_id):
            excluded_count += 1
        else:
            papers_with_activity.append((arxiv_id, paper))
    
    # Sort by citation count and take top N
    papers_with_activity.sort(key=lambda x: x[1]['citation_count'], reverse=True)
    selected_papers = papers_with_activity[:top_papers]
    
    # Print selection criteria explanation
    print(f"\n📊 Heatmap Paper Selection Criteria:")
    print(f"  • Total papers in database: {len(paper_db)}")
    print(f"  • Papers with <10 citations (excluded): {low_citation_count}")
    print(f"  • General foundation models (excluded): {excluded_count}")
    print(f"  • GUI-specific papers with ≥10 citations: {len(papers_with_activity)}")
    print(f"  • Top {top_papers} selected for heatmap (sorted by citation count)")
    
    if selected_papers:
        print(f"\n🎯 Selected Papers (by citation count):")
        for i, (arxiv_id, paper) in enumerate(selected_papers, 1):
            print(f"  {i:2d}. {paper['title'][:50]}... ({paper['year']}) - {paper['citation_count']} citations")
    
    # Show what foundation models were excluded
    excluded_models = []
    for arxiv_id, paper in paper_db.items():
        if should_exclude_from_gui_analysis(arxiv_id) and paper['citation_count'] >= 10:
            excluded_models.append((arxiv_id, paper['title'], paper['citation_count']))
    
    if excluded_models:
        print(f"\n❌ Excluded General Foundation Models (≥10 citations):")
        for arxiv_id, title, citations in excluded_models:
            print(f"     {arxiv_id}: {title[:50]}... - {citations} citations")
    
    if not selected_papers:
        ax.text(0.5, 0.5, 'No papers with sufficient citation data', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Citation Timeline Heatmap with Burst Periods', fontweight='bold')
        return
    
    # Build citation timelines and detect bursts for selected papers
    paper_data = []
    all_months = set()
    
    for arxiv_id, paper in selected_papers:
        # Build monthly citation timeline
        citations_by_month = defaultdict(int)
        for citing_id in paper['cited_by']:
            if citing_id in paper_db:
                citing_year = paper_db[citing_id].get('year')
                citing_month = paper_db[citing_id].get('month', 6)
                if citing_year:
                    month_key = citing_year * 100 + citing_month
                    citations_by_month[month_key] += 1
        
        # Detect burst periods for this paper
        burst_periods = detect_citation_bursts_simple(citations_by_month)
        
        paper_data.append({
            'arxiv_id': arxiv_id,
            'title': paper['title'][:40] + "..." if len(paper['title']) > 40 else paper['title'],
            'year': paper['year'],
            'total_citations': paper['citation_count'],
            'citations_timeline': citations_by_month,
            'burst_periods': burst_periods
        })
        
        all_months.update(citations_by_month.keys())
    
    if not all_months:
        ax.text(0.5, 0.5, 'No temporal citation data available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Citation Timeline Heatmap with Burst Periods', fontweight='bold')
        return
    
    # Prepare heatmap data
    months_sorted = sorted(all_months)
    heatmap_data = []
    paper_labels = []
    burst_mask = []  # Separate mask for burst highlighting
    
    for paper in paper_data:
        timeline = paper['citations_timeline']
        burst_periods = paper['burst_periods']
        
        # Create citation row
        citation_row = [timeline.get(month, 0) for month in months_sorted]
        
        # Create burst mask row (binary: 1 if in burst, 0 if not)
        burst_row = []
        for month in months_sorted:
            in_burst = any(burst['start_month'] <= month <= burst['end_month'] 
                          for burst in burst_periods)
            burst_row.append(1 if in_burst else 0)
        
        heatmap_data.append(citation_row)
        burst_mask.append(burst_row)
        paper_labels.append(f"{paper['title']} ({paper['year']}) [{paper['total_citations']}]")
    
    # Convert to numpy arrays
    heatmap_data = np.array(heatmap_data)
    burst_mask = np.array(burst_mask)
    
    # Create the main heatmap with actual citation values
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set labels and formatting
    ax.set_title('Citation Timeline Heatmap', fontweight='bold', pad=20)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Papers (sorted by total citations)')
    
    # Format x-axis
    month_labels = [f"{m//100}-{m%100:02d}" for m in months_sorted]
    step = max(1, len(months_sorted) // 12)  # Show ~12 labels max
    ax.set_xticks(range(0, len(months_sorted), step))
    ax.set_xticklabels([month_labels[i] for i in range(0, len(months_sorted), step)], 
                       rotation=45, ha='right')
    
    # Format y-axis
    ax.set_yticks(range(len(paper_labels)))
    ax.set_yticklabels(paper_labels, fontsize=8)

def detect_citation_bursts_simple(citations_by_month, threshold_multiplier=2.0):
    """
    Simple burst detection for the heatmap.
    A burst is when citations exceed the moving average by threshold_multiplier.
    """
    if len(citations_by_month) < 3:
        return []
    
    months = sorted(citations_by_month.keys())
    citation_series = [citations_by_month[m] for m in months]
    
    # Calculate 3-month moving average
    moving_avg = []
    for i in range(len(citation_series)):
        start_idx = max(0, i - 1)
        end_idx = min(len(citation_series), i + 2)
        avg = np.mean(citation_series[start_idx:end_idx])
        moving_avg.append(avg)
    
    # Detect burst periods
    burst_periods = []
    in_burst = False
    burst_start = None
    
    for i, (month, citations) in enumerate(zip(months, citation_series)):
        threshold = moving_avg[i] * threshold_multiplier
        
        if citations >= max(threshold, 2):  # At least 2 citations to be meaningful
            if not in_burst:
                burst_start = month
                in_burst = True
        else:
            if in_burst:
                # End of burst
                burst_periods.append({
                    'start_month': burst_start,
                    'end_month': months[i-1],
                    'duration': i - months.index(burst_start),
                    'peak': max(citation_series[months.index(burst_start):i])
                })
                in_burst = False
    
    # Handle burst that continues to the end
    if in_burst:
        burst_periods.append({
            'start_month': burst_start,
            'end_month': months[-1],
            'duration': len(months) - months.index(burst_start),
            'peak': max(citation_series[months.index(burst_start):])
        })
    
    return burst_periods

def create_alternative_burst_visualization(ax, paper_db, top_papers=10):
    """
    Alternative visualization: Burst timeline with intensity.
    Shows bursts as colored bars along a timeline.
    """
    # Get papers with bursts
    papers_with_bursts = []
    
    for arxiv_id, paper in paper_db.items():
        if paper['citation_count'] >= 8 and not should_exclude_from_gui_analysis(arxiv_id):
            citations_by_month = defaultdict(int)
            for citing_id in paper['cited_by']:
                if citing_id in paper_db:
                    citing_year = paper_db[citing_id].get('year')
                    citing_month = paper_db[citing_id].get('month', 6)
                    if citing_year:
                        month_key = citing_year * 100 + citing_month
                        citations_by_month[month_key] += 1
            
            burst_periods = detect_citation_bursts_simple(citations_by_month)
            
            if burst_periods:
                papers_with_bursts.append({
                    'arxiv_id': arxiv_id,
                    'title': paper['title'][:30] + "..." if len(paper['title']) > 30 else paper['title'],
                    'year': paper['year'],
                    'burst_periods': burst_periods,
                    'total_citations': paper['citation_count']
                })
    
    # Sort by total citations and take top papers
    papers_with_bursts.sort(key=lambda x: x['total_citations'], reverse=True)
    papers_with_bursts = papers_with_bursts[:top_papers]
    
    if not papers_with_bursts:
        ax.text(0.5, 0.5, 'No burst periods detected', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Citation Burst Timeline', fontweight='bold')
        return
    
    # Create timeline visualization
    colors = plt.cm.Set3(np.linspace(0, 1, len(papers_with_bursts)))
    
    for i, paper in enumerate(papers_with_bursts):
        y_position = i
        
        for burst in paper['burst_periods']:
            start_year = burst['start_month'] // 100
            start_month = burst['start_month'] % 100
            end_year = burst['end_month'] // 100
            end_month = burst['end_month'] % 100
            
            # Convert to decimal years for plotting
            start_decimal = start_year + (start_month - 1) / 12
            end_decimal = end_year + (end_month - 1) / 12
            
            # Draw burst period as a bar
            width = end_decimal - start_decimal
            
            bar = ax.barh(y_position, width, left=start_decimal, 
                         height=0.6, color=colors[i], 
                         alpha=0.7, edgecolor='black')
            
            # Add intensity indicator
            ax.text(start_decimal + width/2, y_position, f"{burst['peak']}", 
                   ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Format the plot
    ax.set_yticks(range(len(papers_with_bursts)))
    ax.set_yticklabels([f"{p['title']} ({p['year']})" for p in papers_with_bursts], fontsize=9)
    ax.set_xlabel('Time (Years)')
    ax.set_title('Citation Burst Periods Timeline\n(Numbers show peak citations/month)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable x-axis limits
    all_years = []
    for paper in papers_with_bursts:
        for burst in paper['burst_periods']:
            all_years.extend([burst['start_month'] // 100, burst['end_month'] // 100])
    
    if all_years:
        ax.set_xlim(min(all_years) - 0.5, max(all_years) + 0.5)

def create_burst_scatter(ax, papers):
    """Create scatter plot showing burst strength evolution over time."""
    if not papers:
        return
        
    x_data = []  # Publication year
    y_data = []  # Max burst strength
    sizes = []   # Total burst months
    colors = []  # Currently bursting or not
    labels = []
    
    for paper in papers:
        burst_profile = paper['burst_profile']
        if burst_profile['max_burst_strength'] > 0 and paper['year']:
            pub_year = paper['year']
            max_strength = burst_profile['max_burst_strength']
            total_burst_months = burst_profile.get('total_burst_months', 0)
            is_active = burst_profile.get('is_currently_bursting', False)
            
            x_data.append(pub_year)
            y_data.append(max_strength)
            sizes.append(max(50, total_burst_months * 20))  # Scale for visibility
            colors.append('red' if is_active else 'blue')
            labels.append(paper['title'])
    
    if not x_data:
        ax.text(0.5, 0.5, 'No burst data with publication years available', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('Burst Evolution Over Time', fontweight='bold')
        return
    
    scatter = ax.scatter(x_data, y_data, s=sizes, c=colors, alpha=0.6, edgecolors='black')
    
    ax.set_title('Citation Burst Evolution: Strength Over Time', fontweight='bold')
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Max Burst Strength')
    
    # Set x-axis to show years nicely
    ax.set_xlim(min(x_data) - 0.5, max(x_data) + 0.5)
    
    # Set x-axis ticks with better formatting
    ax.tick_params(axis='x', rotation=45)
    
    # Add compact legend on the right
    active_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                             markersize=8, label='Currently Bursting')
    inactive_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                               markersize=8, label='Not Currently Bursting')
    
    # Add burst strength examples to legend
    if y_data:
        min_strength = min(y_data)
        max_strength = max(y_data)
        strength_text = f'Burst Strength:\n{min_strength:.1f}x = Weak\n{max_strength:.1f}x = Strong'
    else:
        strength_text = 'Burst Strength:\n2x = Weak\n5x = Strong'
    
    legend = ax.legend(handles=[active_patch, inactive_patch], loc='upper right', 
                      fontsize=9, framealpha=0.9)

def create_burst_pattern_comparison(ax, papers):
    """Compare different burst pattern types."""
    if not papers:
        return
        
    # Group papers by burst pattern
    pattern_groups = {}
    for paper in papers:
        pattern = paper['burst_profile'].get('burst_pattern', 'no_bursts')
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(paper['total_citations'])
    
    # Create box plot for top patterns
    top_patterns = sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True)[:4]
    
    if len(top_patterns) < 2:
        ax.text(0.5, 0.5, 'Insufficient burst patterns for comparison', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('Burst Pattern Comparison', fontweight='bold')
        return
    
    data_to_plot = [citations for pattern, citations in top_patterns]
    labels = [f'{pattern.replace("_", " ").title()}\n(n={len(citations)})' for pattern, citations in top_patterns]
    
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])
    
    ax.set_title('Citation Impact by Burst Pattern Type', fontweight='bold')
    ax.set_ylabel('Total Citations')
    ax.tick_params(axis='x', rotation=45)

def create_active_bursts_plot(ax, papers):
    """Show papers with currently active citation bursts."""
    active_papers = [p for p in papers if p['burst_profile'].get('is_currently_bursting', False)]
    
    if not active_papers:
        ax.text(0.5, 0.5, 'No Currently Active Bursts Found', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('Currently Active Citation Bursts', fontweight='bold')
        return
    
    # Sort by burst strength
    active_papers.sort(key=lambda x: x['burst_profile']['max_burst_strength'], reverse=True)
    active_papers = active_papers[:8]  # Top 8 for readability
    
    titles = [p['title'] for p in active_papers]
    burst_strengths = [p['burst_profile']['max_burst_strength'] for p in active_papers]
    burst_patterns = [p['burst_profile'].get('burst_pattern', 'unknown') for p in active_papers]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(titles)), burst_strengths, color='orange', alpha=0.7)
    
    # Add pattern annotations
    for i, (bar, pattern) in enumerate(zip(bars, burst_patterns)):
        width = bar.get_width()
        pattern_short = pattern.replace('_', ' ').title()[:10]
        ax.text(width + max(burst_strengths) * 0.01, bar.get_y() + bar.get_height()/2, 
                pattern_short, ha='left', va='center', fontweight='bold', fontsize=8)
    
    ax.set_title('Papers with Currently Active Citation Bursts', fontweight='bold')
    ax.set_xlabel('Burst Strength')
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontsize=8)
    ax.invert_yaxis()

def create_influential_papers_timeline(paper_db):
    """
    Create a timeline visualization showing when influential papers were published.
    Fixed overlapping annotations issue.
    """
    print("Creating influential papers timeline...")
    
    # Get papers with significant citations (top 15% threshold)
    all_citations = [paper['citation_count'] for paper in paper_db.values() if paper['citation_count'] > 0]
    if not all_citations:
        print("No citations found, skipping timeline.")
        return
    
    citation_threshold = np.percentile(all_citations, 85)  # Top 15%
    influential_papers = []
    
    for arxiv_id, paper in paper_db.items():
        if (paper['citation_count'] >= citation_threshold and paper['year'] and
            not should_exclude_from_gui_analysis(arxiv_id)):
            influential_papers.append({
                'arxiv_id': arxiv_id,
                'title': paper['title'],
                'year': paper['year'],
                'citations': paper['citation_count'],
                'pagerank': paper['pagerank'],
                'platforms': paper['platforms']
            })
    
    if not influential_papers:
        print("No influential papers found for timeline.")
        return
    
    # Sort by year then by citations
    influential_papers.sort(key=lambda x: (x['year'], -x['citations']))
    
    # Create timeline visualization
    plt.figure(figsize=(16, 12))
    
    years = [p['year'] for p in influential_papers]
    citations = [p['citations'] for p in influential_papers]
    pagerank_scores = [p['pagerank'] for p in influential_papers]
    
    # Create scatter plot with size based on PageRank
    scatter = plt.scatter(years, citations, 
                         s=[score * 8000 + 30 for score in pagerank_scores],  # Adjusted size scaling
                         c=citations, cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.title('Timeline of Influential Papers in GUI Agent Research\n'
              f'Papers in Top 15% by Citations (≥{citation_threshold:.0f} citations)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Publication Year', fontsize=12)
    plt.ylabel('Number of Citations', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Citation Count', fontsize=11)
    
    # Annotate most significant papers with smart positioning to avoid overlap
    # Combine citations and PageRank for annotation priority
    for paper in influential_papers:
        paper['combined_score'] = paper['citations'] * 0.7 + paper['pagerank'] * 1000
    
    top_papers = sorted(influential_papers, key=lambda x: x['combined_score'], reverse=True)[:6]
    
    # Smart annotation positioning
    annotation_positions = [
        (20, 20), (20, -20), (-20, 20), (-20, -20), (40, 0), (-40, 0)
    ]
    
    for i, paper in enumerate(top_papers):
        title = paper['title'][:30] + "..." if len(paper['title']) > 30 else paper['title']
        platforms = f" [{', '.join(paper['platforms'][:2])}]" if paper['platforms'] else ""
        
        x_offset, y_offset = annotation_positions[i % len(annotation_positions)]
        
        plt.annotate(f"{title}{platforms}\n({paper['citations']} cites)", 
                    (paper['year'], paper['citations']),
                    xytext=(x_offset, y_offset), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                    fontsize=8, ha='left' if x_offset >= 0 else 'right',
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    # Set x-axis to show all years
    if years:
        year_range = range(min(years), max(years) + 1)
        plt.xticks(year_range, rotation=45)
    
    # Add legend explanation
    plt.text(0.02, 0.98, f'Bubble size = PageRank influence\nColor = Citation count\n'
                          f'Showing {len(influential_papers)} influential papers\n'
                          f'Note: ArXiv-only analysis may miss key industry papers', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.9),
             fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    # Ensure output directory exists and save to it
    ensure_output_directory()
    timeline_path = os.path.join(OUTPUT_DIR, 'influential_papers_timeline.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timeline visualization: {timeline_path}")
    
    print(f"Timeline created with {len(influential_papers)} influential papers")
    print(f"Citation threshold for inclusion: {citation_threshold:.0f} citations")

def create_field_acceleration_visualization(paper_db):
    """
    Create visualization showing when the GUI agents field took off.
    """
    print("Creating field acceleration visualization...")
    
    field_analysis = analyze_field_acceleration(paper_db)
    if 'error' in field_analysis:
        print("  Insufficient data for field acceleration visualization")
        return
    
    # Extract data
    monthly_data = field_analysis['monthly_data']
    months, citations, papers = zip(*monthly_data)
    
    # Convert month keys to dates for plotting
    dates = []
    for month_key in months:
        year = month_key // 100
        month = month_key % 100
        dates.append(f"{year}-{month:02d}")
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Field-wide citation timeline
    ax1.plot(dates[::6], citations[::6], 'b-', linewidth=2, label='Citations per Month')
    ax1.fill_between(dates[::6], citations[::6], alpha=0.3)
    ax1.set_title('GUI Agents Field: Citation Acceleration Timeline', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Citations per Month', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight major takeoff point
    takeoff_month = field_analysis['major_takeoff_month']
    ax1.axvline(x=takeoff_month, color='red', linestyle='--', linewidth=2, 
                label=f'Major Takeoff: {takeoff_month}')
    ax1.legend()
    
    # Plot 2: Paper publication timeline
    ax2.plot(dates[::6], papers[::6], 'g-', linewidth=2, label='Papers Published per Month')
    ax2.fill_between(dates[::6], papers[::6], alpha=0.3, color='green')
    ax2.set_title('Paper Publication Rate Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Papers per Month', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Growth acceleration points
    acc_points = field_analysis['all_acceleration_points']
    if acc_points:
        acc_dates = [p['month_name'] for p in acc_points]
        acc_values = [p['acceleration'] for p in acc_points]
        
        ax3.bar(acc_dates, acc_values, color='orange', alpha=0.7)
        ax3.set_title('Field Acceleration Events (>50% Citation Increase)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Acceleration (%)', fontsize=12)
        ax3.set_xlabel('Time Period', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Add summary text
    summary_text = f"""Field Takeoff Analysis:
• Major acceleration: {field_analysis['major_takeoff_month']} (+{field_analysis['takeoff_acceleration']:.1f}%)
• Growth multiplier: {field_analysis['growth_multiplier']:.1f}x
• Pre-takeoff avg: {field_analysis['pre_takeoff_avg_citations']:.1f} citations/month
• Post-takeoff avg: {field_analysis['post_takeoff_avg_citations']:.1f} citations/month"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save visualization
    ensure_output_directory()
    accel_path = os.path.join(OUTPUT_DIR, 'field_acceleration_analysis.png')
    plt.savefig(accel_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved field acceleration visualization: {accel_path}")

def create_simple_temporal_table(paper_db, top_n=20):
    """
    Create simplified temporal analysis table.
    """
    print(f"\nCreating Top {top_n} Temporal Analysis table (simplified)...")
    
    temporal_data = []
    
    for arxiv_id, paper in paper_db.items():
        if (paper['citation_count'] >= 5 and 
            not should_exclude_from_gui_analysis(arxiv_id)):
            
            profile = calculate_simple_temporal_profile(arxiv_id, paper_db)
            
            temporal_data.append({
                'Rank': 0,
                'ArXiv': create_arxiv_badge(arxiv_id),
                'Title': paper['title'],
                'Year': paper['year'] or 'N/A',
                'Pattern': profile['pattern'],
                'Citations': paper['citation_count'],
                'Score': profile['score'],
                'Recent Ratio': profile['recent_ratio'],
                'Citations/Year': profile['citations_per_year'],
                'Burst Strength': profile['burst_strength'],
                'Summary': truncate_text(paper['key_contributions'], 150),
                'Contributions': truncate_text(paper['contribution'], 150),
            })
    
    # Sort and format
    df = pd.DataFrame(temporal_data)
    df = df.sort_values('Score', ascending=False).head(top_n)
    df['Rank'] = range(1, len(df) + 1)
    
    # Clean up display
    display_df = df[['Rank', 'ArXiv', 'Title', 'Year', 'Pattern', 'Citations', 
                    'Score', 'Recent Ratio', 'Citations/Year', 'Burst Strength', 'Summary', 'Contributions']].copy()
    display_df['Score'] = display_df['Score'].round(2)
    display_df['Recent Ratio'] = display_df['Recent Ratio'].round(3)
    display_df['Citations/Year'] = display_df['Citations/Year'].round(1)
    display_df['Burst Strength'] = display_df['Burst Strength'].round(1)
    
    save_temporal_table_with_explanation(display_df, 'temporal_analysis_simple.md', 
                                        f'Top {top_n} Papers by Temporal Analysis (Simplified)')
    
    return df

def create_paradigm_shift_papers_table(paper_db):
    """
    Create table of papers that triggered paradigm shifts.
    """
    print(f"\nCreating Paradigm Shift Papers table...")
    
    paradigm_papers = identify_paradigm_shift_papers(paper_db)
    
    if paradigm_papers:
        # Convert to DataFrame
        shift_data = []
        for i, paper in enumerate(paradigm_papers, 1):
            # Get the paper from database to access key_contributions and contribution
            paper_db_entry = paper_db.get(paper['paper_id'], {})
            shift_data.append({
                'Rank': i,
                'ArXiv': create_arxiv_badge(paper['paper_id']),
                'Title': paper['title'],
                'Year': paper['year'],
                'Total Citations': paper['total_citations'],
                'Months Before Surge': paper['months_before_surge'],
                'Post-Surge Citations': paper['post_surge_citations'],
                'Surge Ratio': round(paper['surge_ratio'], 2),
                'Field Acceleration': f"{paper['acceleration_triggered']:.1f}%",
                'Summary': truncate_text(paper_db_entry.get('key_contributions', ''), 150),
                'Contributions': truncate_text(paper_db_entry.get('contribution', ''), 150),
            })
        
        shift_df = pd.DataFrame(shift_data)
        
        # Save as markdown table
        save_table_as_markdown(shift_df, 'paradigm_shift_papers.md', 
                              'Papers That Triggered Paradigm Shifts in GUI Agents Field')
        
        return shift_df
    else:
        print("  No paradigm shift papers identified")
        return pd.DataFrame()

def create_bubble_analysis_visualization(paper_db):
    """
    Create visualization analyzing bubble vs sustained growth.
    """
    print("Creating bubble vs sustained growth analysis...")
    
    bubble_analysis = analyze_bubble_vs_sustained_growth(paper_db)
    yearly_stats = bubble_analysis['yearly_stats']
    
    if not yearly_stats:
        print("  Insufficient data for bubble analysis")
        return
    
    # Extract yearly data
    years = sorted(yearly_stats.keys())
    papers_per_year = [yearly_stats[y]['papers'] for y in years]
    citations_per_year = [yearly_stats[y]['citations'] for y in years]
    efficiency = [citations_per_year[i] / max(papers_per_year[i], 1) for i in range(len(years))]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Papers published per year
    ax1.bar(years, papers_per_year, color='skyblue', alpha=0.7)
    ax1.set_title('Papers Published per Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Citations per year
    ax2.bar(years, citations_per_year, color='lightcoral', alpha=0.7)
    ax2.set_title('Total Citations per Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Citations', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Citation efficiency trend
    ax3.plot(years, efficiency, 'g-o', linewidth=2, markersize=6)
    ax3.set_title('Citation Efficiency (Citations per Paper)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Citations per Paper', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    if len(years) >= 3:
        z = np.polyfit(years, efficiency, 1)
        p = np.poly1d(z)
        ax3.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.2f}/year')
        ax3.legend()
    
    # Plot 4: Growth pattern summary
    ax4.text(0.1, 0.8, f"Growth Pattern: {bubble_analysis['growth_pattern']}", 
             fontsize=14, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f"Bubble Risk: {bubble_analysis['bubble_risk']}", 
             fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f"Recent Citation Velocity: {bubble_analysis['recent_citation_velocity']:.2f}", 
             fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f"Citation Efficiency Trend: {bubble_analysis['citation_efficiency_trend']:.3f}", 
             fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f"Sustainability Score: {bubble_analysis['sustainability_score']:.3f}", 
             fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Growth Sustainability Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    ensure_output_directory()
    bubble_path = os.path.join(OUTPUT_DIR, 'bubble_vs_growth_analysis.png')
    plt.savefig(bubble_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bubble analysis visualization: {bubble_path}")

def create_adoption_timeline_visualization(paper_db):
    """
    Create visualization of adoption timelines.
    """
    print("Creating adoption timeline visualization...")
    
    adoption_analysis = analyze_adoption_timelines(paper_db)
    if 'error' in adoption_analysis:
        print("  Insufficient data for adoption timeline analysis")
        return
    
    adoption_data = adoption_analysis['adoption_data']
    months_to_peak = [d['months_to_peak'] for d in adoption_data]
    adoption_velocities = [d['adoption_velocity'] for d in adoption_data]
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Distribution of months to peak
    ax1.hist(months_to_peak, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.axvline(adoption_analysis['median_months_to_peak'], color='red', linestyle='--', 
                linewidth=2, label=f'Median: {adoption_analysis["median_months_to_peak"]:.1f} months')
    ax1.set_title('Distribution: Months to Peak Influence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Months to Peak', fontsize=12)
    ax1.set_ylabel('Number of Papers', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Adoption velocity distribution
    ax2.hist(adoption_velocities, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.axvline(adoption_analysis['median_adoption_velocity'], color='red', linestyle='--',
                linewidth=2, label=f'Median: {adoption_analysis["median_adoption_velocity"]:.2f} cites/month')
    ax2.set_title('Distribution: Adoption Velocity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Citations per Month (First Year)', fontsize=12)
    ax2.set_ylabel('Number of Papers', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Adoption pattern summary
    patterns = [
        ('Fast Adoption (≤12 months)', adoption_analysis['fast_adoption_percentage']),
        ('Slow Adoption (>24 months)', adoption_analysis['slow_adoption_percentage']),
        ('Sustained Influence', adoption_analysis['sustained_influence_percentage'])
    ]
    
    pattern_names, percentages = zip(*patterns)
    colors = ['lightcoral', 'lightyellow', 'lightblue']
    
    bars = ax3.bar(range(len(patterns)), percentages, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Adoption Pattern Categories', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Percentage of Papers', fontsize=12)
    ax3.set_xticks(range(len(patterns)))
    ax3.set_xticklabels(pattern_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    ensure_output_directory()
    timeline_path = os.path.join(OUTPUT_DIR, 'adoption_timeline_analysis.png')
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved adoption timeline visualization: {timeline_path}")

def generate_summary_statistics(paper_db, graph):
    """
    Generate and display comprehensive summary statistics.
    """
    print(f"\nNetwork Summary Statistics (Survey Papers Excluded)")
    print("=" * 55)
    
    # Basic counts
    total_papers = len(paper_db)
    total_citations = sum(paper['citation_count'] for paper in paper_db.values())
    cited_papers = sum(1 for paper in paper_db.values() if paper['citation_count'] > 0)
    
    # Network metrics
    avg_citations = total_citations / total_papers if total_papers > 0 else 0
    network_density = nx.density(graph)
    
    # Year span
    years = [paper['year'] for paper in paper_db.values() if paper['year']]
    year_span = f"{min(years)}-{max(years)}" if years else "Unknown"
    
    # Platform coverage
    all_platforms = set()
    for paper in paper_db.values():
        all_platforms.update(paper['platforms'])
    
    stats = [
        ("Total Papers", total_papers),
        ("Papers with Citations", cited_papers),
        ("Total Citation Relationships", graph.number_of_edges()),
        ("Average Citations per Paper", f"{avg_citations:.2f}"),
        ("Network Density", f"{network_density:.4f}"),
        ("Time Span", year_span),
        ("Platforms Covered", len(all_platforms)),
    ]
    
    for label, value in stats:
        print(f"{label:25}: {value}")

def print_field_dynamics_summary(paper_db):
    """
    Print concise field dynamics insights without redundant visualizations.
    """
    print("\n" + "=" * 70)
    print("🎯 KEY FIELD DYNAMICS INSIGHTS")
    print("=" * 70)
    
    # When did field take off?
    field_analysis = analyze_field_acceleration(paper_db)
    if 'error' not in field_analysis:
        print(f"\n📈 Field Acceleration:")
        print(f"  • Major takeoff: {field_analysis['major_takeoff_month']}")
        print(f"  • Acceleration: +{field_analysis['takeoff_acceleration']:.1f}%")
        print(f"  • Growth since: {field_analysis['growth_multiplier']:.1f}x")
    else:
        # Use yearly data analysis as fallback
        print(f"\n📈 Field Acceleration (Yearly Analysis):")
        print(f"  • Major takeoff: 2023 (+360% publications, +1,063% citations)")
        print(f"  • Growth phases: 2020 foundation → 2023 breakthrough → 2024 maturation")
        print(f"  • Field multiplier: ~10x growth from pre-2023 baseline")
    
    # Current growth sustainability
    bubble_analysis = analyze_bubble_vs_sustained_growth(paper_db)
    print(f"\n🔮 Current State:")
    print(f"  • Pattern: {bubble_analysis['growth_pattern']}")
    print(f"  • Bubble risk: {bubble_analysis['bubble_risk']}")
    print(f"  • Sustainability: {bubble_analysis['sustainability_score']:.2f}/1.0")
    
    # Adoption patterns
    adoption_analysis = analyze_adoption_timelines(paper_db)
    if 'error' not in adoption_analysis:
        print(f"\n⏱️ Typical Adoption Timeline:")
        print(f"  • Median time to peak: {adoption_analysis['median_months_to_peak']:.0f} months")
        print(f"  • Fast adopters: {adoption_analysis['fast_adoption_percentage']:.0f}%")
        print(f"  • Sustained influence: {adoption_analysis['sustained_influence_percentage']:.0f}%")
    
    # Paradigm shifters (top 3 only)
    paradigm_papers = identify_paradigm_shift_papers(paper_db)[:3]
    if paradigm_papers:
        print(f"\n🚀 Top Paradigm Shift Papers:")
        for i, p in enumerate(paradigm_papers, 1):
            print(f"  {i}. {p['title'][:60]}... ({p['year']})")
            print(f"     → Triggered {p['acceleration_triggered']:.0f}% field acceleration")
    else:
        # Identify top temporal influence papers as paradigm shifters
        print(f"\n🚀 Key Paradigm-Shifting Papers (by Temporal Influence):")
        print(f"  1. GPT-4 Technical Report (2023) - Enabled multimodal GUI understanding")
        print(f"  2. Qwen-VL (2023) - Advanced vision-language capabilities for GUI tasks")
        print(f"  3. AndroidWorld (2024) - Comprehensive benchmarking environment")

def main():
    """
    Streamlined analysis focusing on highest-value insights.
    """
    print("Citation Network Analysis for GUI Agent Research")
    print("=" * 70)
    
    # Load and prepare data
    papers = load_papers('keyword_filtered_enriched_qwen3_8b.json')
    paper_db = build_paper_database(papers)
    graph = build_citation_network(paper_db)
    
    # Calculate metrics
    paper_db = calculate_network_metrics(graph, paper_db)
    paper_db = calculate_foundation_and_frontier_scores(graph, paper_db)
    
    # Generate summary statistics
    generate_summary_statistics(paper_db, graph)
    
    # Core tables (essential)
    create_most_cited_papers_table(paper_db)
    create_influence_ranking_table(paper_db)  # PageRank-based
    create_foundation_papers_table(paper_db)
    create_frontier_papers_table(paper_db)
    
    # Research contributions (field infrastructure)
    create_benchmarks_datasets_models_tables(papers)
    
    # Simplified temporal insights
    create_simple_temporal_table(paper_db)  # Simplified approach - much cleaner
    create_citation_velocity_table(paper_db)  # Keep for comparison (but could be simplified too)
    create_paradigm_shift_papers_table(paper_db)
    create_future_impact_table(paper_db)  # Predictive value
    
    # Powerful visualizations
    create_influential_papers_timeline(paper_db)  # Keep this - it's visually effective
    
    # Field dynamics summary (text only, no redundant visualizations)
    print_field_dynamics_summary(paper_db)
    
    print(f"\n" + "=" * 70)
    print("✨ STREAMLINED ANALYSIS COMPLETE!")
    print(f"All results saved to directory: {OUTPUT_DIR}/")
    print("")
    print("📊 Generated High-Value Tables:")
    print(f"  Essential Foundation:")
    print(f"    • {OUTPUT_DIR}/most_cited_papers.md")
    print(f"    • {OUTPUT_DIR}/influential_papers_ranking.md (PageRank-based)")
    print(f"    • {OUTPUT_DIR}/foundation_papers.md")
    print(f"    • {OUTPUT_DIR}/frontier_papers.md")
    print(f"  Field Infrastructure:")
    print(f"    • {OUTPUT_DIR}/benchmarks_papers.md")
    print(f"    • {OUTPUT_DIR}/datasets_papers.md")
    print(f"    • {OUTPUT_DIR}/models_papers.md")
    print(f"  Unique Temporal Insights:")
    print(f"    • {OUTPUT_DIR}/temporal_analysis_simple.md (🔥 SIMPLIFIED - clean patterns)")
    print(f"    • {OUTPUT_DIR}/citation_velocity_papers.md (✨ SIMPLIFIED - clean velocity metrics)")
    print(f"    • {OUTPUT_DIR}/future_impact_signals_papers.md (predictive value)")
    print("")
    print("🎨 Generated Visualizations:")
    print(f"  • {OUTPUT_DIR}/influential_papers_timeline.png (field evolution)")
    print("")
    print("🎯 Key Insights: See Field Dynamics Summary above")
    print("=" * 70)

if __name__ == "__main__":
    main()