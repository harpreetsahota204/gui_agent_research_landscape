import json
import sys
import os
import re
import argparse
from datetime import datetime

class Logger:
    """Simple logger that writes to both terminal and file."""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        if self.log_file:
            # Create log file with timestamp header
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Filter Papers Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def log(self, message):
        """Print to terminal and write to log file."""
        print(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")

# Improved Multi-Tier Keyword Classification for GUI/OS/Visual Agent Papers
class KeywordFilter:
    def __init__(self):
        # High-precision keywords (very specific to GUI/OS agents) - 10 points each
        self.high_precision_keywords = [
            # Very specific agent types
            "gui agent", "gui agents", "visual gui agent", "visual gui agents",
            "ui agent", "ui agents", "screen agent", "desktop agent", "mobile agent", "mobile agents",
            "os agent", "os agents", "operating system agent", "operating system agents",
            "computer using agent", "vision action agent", "vision action agents",
            
            # Specific technical terms
            "gui automation", "gui control", "gui interaction", "gui navigation", "graphical user interfaces", "graphical user interface"
            "gui grounding", "gui planning", "gui evaluation", "gui testing", "ui screenshot", "ui screenshots", "gui screenshot", "gui screenshots", 
            "ui grounding", "ui planning", "ui element recognition", "screenshot","screenshots",
            "screenshot analysis", "screenshot based", "screen understanding",
            "visual grounding", "element grounding", "click point", "click points",
            "coordinate extraction", "click action", "scroll action",
            
            # Research-specific terms (datasets, benchmarks, tools), 
            "webarena", "miniwob", "mind2web", "omniact", "screenqa", "rico dataset",
            "windowsagentarena", "osworld", "screenspot", "mobile3m", "pixelweb", 
            "webvoyager", "guicourse", "groundui", "showui", "uibert", "MobileAgent",
            
            # Very specific compound terms
            "computer vision for gui", "vision language action", "autonomous gui",
            "desktop automation", "browser automation", "mobile automation",
            "android gui", "ios interface", "web interface",
            "foundation action model", "embodied interaction", "screen interaction"
        ]
        
        # Medium-precision keywords (need additional context) - 3 points each with context boost
        self.medium_precision_keywords = [
            "visual agent", "visual agents", "embodied agent", "embodied agents", "embodied ai", "embodiment ai"
            "user interface", "graphical user interface", "mobile ui", "document ai",
            "computer control", "computer use", "digital interface", "document images",
            "vision language assistant", "visual element recognition",
            "action sequence", "multi-step gui", "gui task", "interface task",
            "automation success rate", "element accuracy", "click accuracy",
            "gpt 4 technical report", "cogvlm", "webgpt", "ferret", "vision language action models"
        ]
        
        # Context keywords (strengthen matches) - 0.1 points each, boost medium-precision
        self.context_keywords = [
            "agent", "agents", "autonomous", "automation", "control", "interaction",
            "multimodal", "vision", "visual", "language model", "llm", "ai assistant",
            "interface", "gui", "ui", "screen", "screenshot", "click", "navigation",
            "element", "widget", "button", "mobile", "desktop", "web", "browser",
            "task", "action", "planning", "reasoning", "grounding", "understanding",
            "evaluation", "benchmark", "dataset", "performance"
        ]
        
        # Exclusion patterns (apply 70% penalty if found)
        self.exclusion_patterns = [
            r"\b(medical|clinical|hospital|patient|disease|diagnosis|treatment|therapy|drug|pharmaceutical)\b",
            r"\b(financial|banking|trading|investment|market|stock|economic|business|commerce)\b",
            r"\b(text mining|text analysis|natural language processing|nlp|text classification|sentiment analysis)\b",
        ]

    def normalize_text(self, text):
        """Normalize text for keyword matching - ignore case and special characters."""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Replace all non-alphanumeric characters with spaces (including hyphens, underscores)
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalize multiple spaces to single spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def find_keyword_matches(self, text, keywords):
        """Find keyword matches using word boundaries."""
        matches = []
        normalized_text = self.normalize_text(text)
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, normalized_text):
                matches.append(keyword)
        return matches

    def check_exclusion_context(self, text):
        """Check if text contains exclusion patterns."""
        normalized_text = self.normalize_text(text)
        for pattern in self.exclusion_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return True
        return False

    def calculate_relevance_score(self, text):
        """Calculate relevance score based on keyword matches and context."""
        matches = {
            'high_precision': self.find_keyword_matches(text, self.high_precision_keywords),
            'medium_precision': self.find_keyword_matches(text, self.medium_precision_keywords),
            'context': self.find_keyword_matches(text, self.context_keywords)
        }
        
        has_exclusion = self.check_exclusion_context(text)
        
        # Calculate base score
        score = 0.0
        score += len(matches['high_precision']) * 10.0
        
        # Medium precision with context boost
        medium_score = len(matches['medium_precision']) * 3.0
        context_boost = min(len(matches['context']) * 0.5, 5.0)
        score += medium_score * (1.0 + context_boost / 10.0)
        
        score += len(matches['context']) * 0.1
        
        # Apply exclusion penalty
        if has_exclusion:
            score *= 0.3
        
        return score, matches

# Initialize the improved filter globally
IMPROVED_FILTER = KeywordFilter()

def filter_gui_os_visual_papers(paper_data, threshold=2.5):
    """
    Filter papers for GUI/OS/Visual agent relevance using improved scoring system.
    
    Args:
        paper_data: Dictionary with paper information
        threshold: Minimum relevance score for inclusion (default: 5.0)
        
    Returns:
        tuple: (bool, dict) - (is_relevant, detailed_results)
    """
    # Get sections, handling missing fields gracefully
    sections = paper_data.get('sections', {})
    
    # Combine searchable text from title and available sections
    searchable_parts = [
        paper_data.get('title', ''),
        sections.get('abstract', ''),
        sections.get('introduction', ''),
        sections.get('conclusion', '')
    ]
    
    # Filter out empty strings and combine
    combined_text = ' '.join([part for part in searchable_parts if part])
    
    # Calculate relevance score using improved filter
    score, matches = IMPROVED_FILTER.calculate_relevance_score(combined_text)
    
    # Determine relevance
    is_relevant = score >= threshold
    
    # Prepare detailed results
    detailed_results = {
        'score': score,
        'threshold': threshold,
        'matches': matches,
        'total_matches': sum(len(match_list) for match_list in matches.values()),
        'has_high_precision': len(matches['high_precision']) > 0,
        'has_exclusion': IMPROVED_FILTER.check_exclusion_context(combined_text),
        'matched_keywords': matches['high_precision'] + matches['medium_precision']  # For backward compatibility
    }
    
    return is_relevant, detailed_results

def should_keep_paper(paper_data, output_filename="filtered_papers.json", verbose=False):
    """
    Main filtering function for paper processing pipeline.
    
    Args:
        paper_data: Paper dictionary structure
        
    Returns:
        bool: Whether to keep this paper
    """
    level = paper_data.get('level', 1)
    
    # Always keep level 1 papers
    if level == 1:
        return True
    
    # For level 2 and 3, apply GUI/OS/Visual filter
    if level in [2, 3]:
        is_relevant, _ = filter_gui_os_visual_papers(paper_data)
        return is_relevant
    
    return False

def filter_papers_and_save(input_filename, output_filename="filtered_papers.json", log_filename=None, verbose=False):
    """
    Filter papers and save results to JSON file.
    
    Args:
        input_filename: JSON file containing papers to filter
        output_filename: Where to save filtered results
        log_filename: Optional log file to save terminal output
        verbose: Whether to print matched papers
        
    Returns:
        list: Filtered papers
    """
    # Initialize logger
    logger = Logger(log_filename)
    # Check if input file exists
    if not os.path.exists(input_filename):
        logger.log(f"Error: Input file '{input_filename}' not found!")
        return []
    
    # Load papers
    logger.log(f"Loading papers from {input_filename}...")
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except json.JSONDecodeError as e:
        logger.log(f"Error: Invalid JSON in input file - {e}")
        return []
    except Exception as e:
        logger.log(f"Error loading file: {e}")
        return []
    
    logger.log(f"Loaded {len(papers)} papers")
    
    # Filter papers
    filtered = []
    stats = {'total': len(papers), 'level_1': 0, 'level_2_3_kept': 0, 'level_2_3_filtered': 0}
    
    logger.log("Filtering papers...")
    for i, paper in enumerate(papers):
        arxiv_id = paper.get('arxiv_id', '')
        level = paper.get('level', 1)
        title = paper.get('title', 'No title available')
        logger.log(f"Processing paper {i+1}/{len(papers)} (arxiv_id: {arxiv_id}): {title[:60]}{'...' if len(title) > 60 else ''}")
        
        if level == 1:
            filtered.append(paper)
            stats['level_1'] += 1
        elif level in [2, 3]:
            # Get detailed filtering results
            is_relevant, details = filter_gui_os_visual_papers(paper)
            if is_relevant:
                filtered.append(paper)
                stats['level_2_3_kept'] += 1
                high_precision = details['matches']['high_precision']
                medium_precision = details['matches']['medium_precision']
                score = details['score']
                logger.log(f"  ✅ KEPT (Level {level}, Score: {score:.2f})")
                if high_precision:
                    logger.log(f"    High-precision matches: {high_precision}")
                if medium_precision:
                    logger.log(f"    Medium-precision matches: {medium_precision}")
                if details['has_exclusion']:
                    logger.log(f"    ⚠️  Exclusion context detected (score penalized)")
            else:
                stats['level_2_3_filtered'] += 1
                score = details['score']
                logger.log(f"  ❌ FILTERED (Level {level}, Score: {score:.2f} < {details['threshold']})")
                if verbose and details['total_matches'] > 0:
                    logger.log(f"    Some matches found but score too low: {details['matched_keywords']}")
    
    # Save filtered results
    logger.log(f"Saving filtered papers to {output_filename}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.log(f"Error saving file: {e}")
        return []
    
    # Print statistics
    logger.log(f"\n{'='*50}")
    logger.log(f"FILTERING RESULTS")
    logger.log(f"{'='*50}")
    logger.log(f"Total papers processed: {stats['total']}")
    logger.log(f"Level 1 papers kept: {stats['level_1']}")
    logger.log(f"Level 2/3 papers kept: {stats['level_2_3_kept']}")
    logger.log(f"Level 2/3 papers filtered out: {stats['level_2_3_filtered']}")
    logger.log(f"Final count: {len(filtered)} papers")
    logger.log(f"Reduction: {stats['level_2_3_filtered']} papers removed")
    logger.log(f"Saved to: {output_filename}")
    logger.log(f"Log saved to: {log_filename}")
    logger.log(f"{'='*50}")
    return filtered

def main():
    """Main function to handle command line arguments and run filtering."""
    parser = argparse.ArgumentParser(
        description="Filter papers for GUI/OS/Visual agent relevance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s papers.json
    -> Creates: papers_filtered.json and papers_filtered.log
  
  %(prog)s papers.json -o my_output.json
    -> Creates: my_output.json and my_output.log
  
  %(prog)s papers.json -o results.json --log custom.log --verbose
    -> Creates: results.json and custom.log (with verbose output)

Note: A log file is always created automatically. If not specified with --log,
      it will be named based on the output file (e.g., output.json -> output.log)
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        help='JSON file containing papers to filter'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file for filtered papers (default: <input>_filtered.json)'
    )
    
    parser.add_argument(
        '--log',
        help='Log file to save terminal output (default: <output>.log)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Default output filename
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_filtered.json"
    
    # Generate default log filename if not specified
    if args.log:
        log_file = args.log
    else:
        output_base = os.path.splitext(output_file)[0]
        log_file = f"{output_base}.log"
    
    print(f"GUI/OS/Visual Agent Paper Filter")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print(f"Verbose: {args.verbose}")
    print("-" * 50)
    
    # Run filtering
    filtered_papers = filter_papers_and_save(args.input_file, output_file, log_file, args.verbose)
    
    if filtered_papers:
        print(f"\n✅ Filtering completed successfully!")
    else:
        print(f"\n❌ Filtering failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()