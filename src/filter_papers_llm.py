import json
import os
import gc
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Logger:
    """Simple logger that writes to both terminal and file."""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        if self.log_file:
            # Create log file with timestamp header
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== LLM Filter Papers Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def log(self, message):
        """Print to terminal and write to log file."""
        print(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")

class LLMPaperFilter:
    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507", logger=None):
        """Initialize the LLM-based paper filter"""
        self.logger = logger or Logger()
        self.logger.log(f"Loading LLM model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.logger.log("Model loaded successfully!")
    
    def cleanup_memory(self):
        """Essential memory cleanup after processing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def unload_model(self):
        """Unload model to free memory"""
        if hasattr(self, 'model') and self.model is not None:
            self.logger.log("Unloading model...")
            del self.model
            del self.tokenizer
            self.cleanup_memory()
            self.logger.log("Model unloaded successfully!")
    
    def create_prompt(self, paper_data):
        """Create classification prompt"""
        title = paper_data.get('title', 'No title available')
        sections = paper_data.get('sections', {})
        abstract = sections.get('abstract', '')
        introduction = sections.get('introduction', '')
        conclusion = sections.get('conclusion', '')
        related_work = sections.get('related_work', sections.get('related work', ''))
        
        # Combine available sections
        content_parts = [f"Title: {title}"]
        for section_name, content in [("Abstract", abstract), ("Introduction", introduction), 
                                    ("Related Work", related_work), ("Conclusion", conclusion)]:
            if content:
                content_parts.append(f"{section_name}: {content}")
        
        paper_content = "\n\n".join(content_parts)
        
        return f"""You are an expert researcher specializing in vision language models and GUI agents and visual agents.

You must analyze a research paper and determine if it is relevant to GUI/OS/Visual agents.

A paper is RELEVANT if it discusses ANY of these core topics:

**Agent Types & Systems:**
- GUI agents, visual agents, computer-using agents, OS agents, screen agents, desktop agents
- UI agents, mobile UI agents, or any automated systems that interact with user interfaces
- Foundation action models or embodied interaction systems

**Technical Approaches:**
- GUI automation, GUI control, GUI interaction, GUI navigation, or GUI testing
- Screenshot analysis, screen understanding, visual grounding, or element grounding
- UI element recognition, visual element recognition, or GUI element detection
- Computer vision applied to GUI/interface understanding
- Click actions, scroll actions, mouse/keyboard control, or coordinate extraction

**Platform & Application Areas:**
- Mobile app automation, Android GUI interaction, iOS interface automation
- Web interface automation, browser automation, or web navigation systems
- Desktop automation, Windows OS control, or operating system interaction
- Any automation of digital interfaces or computer control systems

**Planning & Evaluation:**
- GUI planning, interface planning, screen planning, or UI task planning
- Action sequences for GUI tasks, multi-step GUI operations
- GUI evaluation, interface evaluation, automation success rates
- Click accuracy, navigation accuracy, or element accuracy metrics

**Datasets & Benchmarks:**
- WebArena, MiniWoB, Mind2Web, OmniAct, ScreenQA, RICO dataset
- WindowsAgentArena, OSWorld, ScreenSpot, Mobile3M, PixelWeb, WebVoyager
- Any benchmarks for evaluating GUI/visual agent performance

Paper to analyze:
{paper_content}

Is this paper RELEVANT? Respond with exactly one word: "YES" or "NO"
"""
    
    def classify_paper(self, paper_data):
        """Classify if paper is relevant using LLM"""
        try:
            prompt = self.create_prompt(paper_data)
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            # Generate response
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=32768,        
                    temperature=0.6,   
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Parse response
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # Extract thinking content and response
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
            except ValueError:
                index = 0
            
            response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            
            # Clean up tensors
            del model_inputs, generated_ids, output_ids
            self.cleanup_memory()
            
            # Parse YES/NO response
            response_upper = response.upper().strip()
            if "YES" in response_upper and "NO" not in response_upper:
                return True, response
            elif "NO" in response_upper:
                return False, response
            else:
                # Default to False if ambiguous
                return False, f"{response} (AMBIGUOUS - defaulted to NO)"
            
        except Exception as e:
            self.cleanup_memory()
            return False, f"Error: {str(e)}"

def save_checkpoint(filtered_papers, processed_arxiv_ids, checkpoint_file, stats, logger):
    """Save progress checkpoint with both filtered papers and processed IDs"""
    try:
        checkpoint_data = {
            'filtered_papers': filtered_papers,
            'processed_arxiv_ids': processed_arxiv_ids,
            'stats': stats
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.log(f"Checkpoint saved: {len(filtered_papers)} filtered, {len(processed_arxiv_ids)} processed")
    except Exception as e:
        logger.log(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file, logger):
    """Load progress checkpoint"""
    if not os.path.exists(checkpoint_file):
        return [], set(), {'level_1': 0, 'level_2_3_kept': 0, 'level_2_3_filtered': 0}
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        filtered_papers = data.get('filtered_papers', [])
        processed_arxiv_ids = set(data.get('processed_arxiv_ids', []))
        stats = data.get('stats', {'level_1': 0, 'level_2_3_kept': 0, 'level_2_3_filtered': 0})
        logger.log(f"Checkpoint loaded: {len(filtered_papers)} filtered, {len(processed_arxiv_ids)} already processed")
        return filtered_papers, processed_arxiv_ids, stats
    except Exception as e:
        logger.log(f"Failed to load checkpoint: {e}")
        return [], set(), {'level_1': 0, 'level_2_3_kept': 0, 'level_2_3_filtered': 0}

def should_keep_paper(paper_data, llm_filter):
    """Determine if paper should be kept"""
    level = paper_data.get('level', 1)
    
    # Always keep level 1 papers
    if level == 1:
        return True
    
    # For level 2 and 3, use LLM classification
    if level in [2, 3]:
        is_relevant, _ = llm_filter.classify_paper(paper_data)
        return is_relevant
    
    return False

def filter_papers(input_file, output_file, batch_size=50, model_name="Qwen/Qwen3-4B-Thinking-2507", log_file=None):
    """Main filtering function"""
    # Initialize logger
    logger = Logger(log_file)
    
    # Load papers
    logger.log(f"Loading papers from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logger.log(f"Loaded {len(papers)} papers")
    
    # Initialize filter and checkpoint
    llm_filter = LLMPaperFilter(model_name=model_name, logger=logger)
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint.json"
    
    # Load existing progress
    filtered_papers, processed_arxiv_ids, stats = load_checkpoint(checkpoint_file, logger)
    processed_count = 0
    
    try:
        # Process papers
        for i, paper in enumerate(papers):
            arxiv_id = paper.get('arxiv_id', '')
            
            # Skip if already processed
            if arxiv_id in processed_arxiv_ids:
                continue
                
            title = paper.get('title', 'No title available')
            logger.log(f"Processing paper {i+1}/{len(papers)} (arxiv_id: {arxiv_id}): {title[:60]}{'...' if len(title) > 60 else ''}")
            
            level = paper.get('level', 1)
            
            if level == 1:
                filtered_papers.append(paper)
                stats['level_1'] += 1
            elif level in [2, 3]:
                if should_keep_paper(paper, llm_filter):
                    filtered_papers.append(paper)
                    stats['level_2_3_kept'] += 1
                    logger.log(f"  ✅ KEPT (Level {level} - LLM classified as relevant)")
                else:
                    stats['level_2_3_filtered'] += 1
                    logger.log(f"  ❌ FILTERED (Level {level} - LLM classified as not relevant)")
            
            # Mark as processed
            processed_arxiv_ids.add(arxiv_id)
            processed_count += 1
            
            # Save checkpoint every batch_size papers
            if processed_count % batch_size == 0:
                save_checkpoint(filtered_papers, list(processed_arxiv_ids), checkpoint_file, stats, logger)
                logger.log(f"Processed {processed_count} new papers, kept {len(filtered_papers)} total")
    
    finally:
        llm_filter.unload_model()
        # Final checkpoint save
        save_checkpoint(filtered_papers, list(processed_arxiv_ids), checkpoint_file, stats, logger)
    
    # Save final results
    logger.log(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_papers, f, indent=2, ensure_ascii=False)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Print results
    total = len(papers)
    logger.log(f"\n{'='*50}")
    logger.log(f"FILTERING RESULTS")
    logger.log(f"{'='*50}")
    logger.log(f"Total papers processed: {total}")
    logger.log(f"Level 1 papers kept: {stats['level_1']}")
    logger.log(f"Level 2/3 papers kept: {stats['level_2_3_kept']}")
    logger.log(f"Level 2/3 papers filtered out: {stats['level_2_3_filtered']}")
    logger.log(f"Final count: {len(filtered_papers)} papers")
    logger.log(f"Reduction: {stats['level_2_3_filtered']} papers removed")
    logger.log(f"Saved to: {output_file}")
    if log_file:
        logger.log(f"Log saved to: {log_file}")
    logger.log(f"{'='*50}")
    
    return filtered_papers

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Based Paper Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s papers.json
    -> Creates: papers_filtered.json and papers_filtered.log
  
  %(prog)s papers.json -o my_output.json
    -> Creates: my_output.json and my_output.log
  
  %(prog)s papers.json -o results.json --log custom.log --model Qwen/Qwen3-4B-Thinking-2507
    -> Creates: results.json and custom.log with specified model

Note: A log file is always created automatically. If not specified with --log,
      it will be named based on the output file (e.g., output.json -> output.log)
        """
    )
    
    # Required arguments
    parser.add_argument('input_file', help='JSON file containing papers to filter')
    
    # Optional arguments
    parser.add_argument('-o', '--output', help='Output JSON file for filtered papers (default: <input>_filtered.json)')
    parser.add_argument('--log', help='Log file to save terminal output (default: <output>.log)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for checkpoints')
    parser.add_argument('--model', default="Qwen/Qwen3-4B-Thinking-2507", 
                       help='HuggingFace model to use for classification')
    
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
    
    print(f"LLM-Based GUI/OS/Visual Agent Paper Filter")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)
    
    filter_papers(args.input_file, output_file, args.batch_size, args.model, log_file)
    print("✅ Filtering completed!")

if __name__ == "__main__":
    main()
