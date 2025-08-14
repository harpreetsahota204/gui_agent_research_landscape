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

# Essential Visual/GUI/OS Agent Keywords - Optimized for Low False Positives & High Coverage
GUI_OS_VISUAL_AGENT_KEYWORDS = [
    # Core Agent Types (highest priority - very specific to the field)
    "gui agent", "gui agents", "visual gui agent", "visual gui agents", "gui", "ui", 
    "user interface", "user interface agent", "user interface agents", "graphical user interface", "graphical user interface agent", "graphical user interface agents",
    "computer using agent", "computer using agent", "os agent", "os agents","operating system agent", "operating system agents",
    "visual agent", "visual agents", "screen agent", "desktop agent", "mobile ui", "ui agent", "ui agents", "embodied agents", "embodied agent", "embodied",
    
    # GUI-Specific Technical Terms 
    "gui grounding", "gui automation", "gui control", "gui interaction",
    "gui navigation", "gui testing", "gui element", "gui elements", "document images",
    "screenshot", "screenshot analysis", "screenshot based", "screen understanding",
    "visual grounding", "element grounding", "ui grounding", "click point", "click points", 
    
    # Vision-Language for Agents (high precision terms)
    "computer vision for gui", "vision language action", "vision action",
    "vision action agent", "vision action agents", "autonomous gui", "vision language assistant",
    
    # Interaction Modalities (specific to GUI/OS agents)
    "click action", "scroll action", "mouse control", "keyboard control",
    "coordinate extraction", 
    "ui element recognition", "visual element recognition",
    
    # Platform-Specific (helps distinguish from general AI)
    "mobile app", "android gui", "ios interface", "web interface", "android agent",
    "desktop automation", "windows os", "operating system control", "operating system", "os",
    "browser automation", "web navigation", "mobile automation", "mobile agent","mobile agents",
    
    # Research (very specific to field)
    "gpt 4 technical report","webarena", "miniwob", "mind2web", "omniact", "screenqa",
    "rico dataset", "windowsagentarena", "osworld", "screenspot", "erica", "rico",
    "mobile3m", "pixelweb", "webvoyager", "screenspot", "guicourse", "groundui", "showui",
    
    # Action Planning for GUI (distinguishes from general planning)
    "gui planning", "interface planning", "screen planning", "ui planning",
    "action sequence", "multi-step gui", "gui task", "interface task",
    
    # Evaluation Terms (field-specific)
    "gui evaluation", "interface evaluation", "automation success rate",
    "element accuracy", "click accuracy", "navigation accuracy",
    
    # Emerging Terms (2024-2025 papers)
    "foundation action model", "computer control", "computer use", "digital interface",
    "human-computer interaction", "embodied interaction", "screen interaction"
]

def filter_gui_os_visual_papers(paper_data, keywords=GUI_OS_VISUAL_AGENT_KEYWORDS):
    """
    Filter papers for GUI/OS/Visual agent relevance.
    
    Args:
        paper_data: Dictionary with paper information
        keywords: List of keywords to search for
        
    Returns:
        tuple: (bool, list) - (is_relevant, matched_keywords)
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
    
    # Filter out empty strings, join, convert to lowercase
    searchable_text = ' '.join([part for part in searchable_parts if part]).lower()
    
    # Remove punctuation and special characters, keep only letters, numbers, and spaces
    searchable_text = re.sub(r'[^\w\s]', ' ', searchable_text)
    
    # Normalize multiple spaces to single spaces
    searchable_text = re.sub(r'\s+', ' ', searchable_text).strip()
    
    # Check for keyword matches using whole word matching
    matches = []
    for keyword in keywords:
        # Use word boundaries to match whole words only, not substrings
        # This prevents false positives like "gui" matching "language" or "os" matching "cost"
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, searchable_text):
            matches.append(keyword)
    
    # Return True if at least one keyword match found
    return len(matches) > 0, matches

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
            if should_keep_paper(paper, verbose):
                filtered.append(paper)
                stats['level_2_3_kept'] += 1
                logger.log(f"  ✅ KEPT (Level {level} - classified as relevant)")
            else:
                stats['level_2_3_filtered'] += 1
                logger.log(f"  ❌ FILTERED (Level {level} - classified as not relevant)")
    
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