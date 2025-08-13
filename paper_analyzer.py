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
                f.write(f"=== Paper Analyzer Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def log(self, message):
        """Print to terminal and write to log file."""
        print(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")

class LLMPaperAnalyzer:
    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507", logger=None):
        """Initialize the LLM-based paper analyzer"""
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
    
    def create_analysis_prompt(self, paper_data):
        """Create analysis prompt for extracting key information"""
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
        
        return f"""You are an expert research analyst. Analyze the following research paper and extract key information.

Paper to analyze:
{paper_content}

Please provide your analysis in the following JSON format:

{{
    "key_contributions": "Brief summary of the key research themes and contributions based on abstract, introduction, and conclusion",
    "introduces_new_model": true/false,
    "introduces_new_dataset": true/false, 
    "introduces_new_benchmark": true/false,
    "introduces_new_architecture": true/false,
    "contribution": "Summary of how this work differs from related works based on the related work section"
}}

Guidelines:
- key_contributions: 2-3 sentences summarizing the main research themes and contributions
- introduces_new_model: true if the paper presents a new model/system/method
- introduces_new_dataset: true if the paper introduces or creates a new dataset
- introduces_new_benchmark: true if the paper introduces new evaluation benchmarks or metrics
- introduces_new_architecture: true if the paper proposes a new architectural design or framework
- contribution: 1-2 sentences explaining the key difference from existing work mentioned in related work

Respond with ONLY the JSON object, no additional text.
"""
    
    def generate_llm_response(self, prompt):
        """Generate response from LLM"""
        try:
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
                    temperature=0.3,   
                    top_p=0.9,
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
            
            return response
            
        except Exception as e:
            self.cleanup_memory()
            raise Exception(f"LLM generation error: {str(e)}")
    
    def create_json_cleanup_prompt(self, original_response):
        """Create prompt to clean up malformed JSON response"""
        return f"""The following text contains analysis information but is not in valid JSON format. Please convert it to valid JSON with exactly these fields:

Original response:
{original_response}

Convert this to valid JSON with exactly these fields (no additional fields):
{{
    "key_contributions": "string - brief summary of key research themes and contributions",
    "introduces_new_model": true/false,
    "introduces_new_dataset": true/false,
    "introduces_new_benchmark": true/false, 
    "introduces_new_architecture": true/false,
    "contribution": "string - summary of how this differs from related work"
}}

Extract the information from the original response and format it as valid JSON. If any information is missing, use reasonable defaults (empty string for text fields, false for boolean fields).

Respond with ONLY the JSON object, no additional text or explanation."""

    def analyze_paper(self, paper_data):
        """Analyze paper and extract key information"""
        try:
            prompt = self.create_analysis_prompt(paper_data)
            response = self.generate_llm_response(prompt)
            
            # Try to parse JSON response
            try:
                # Find JSON in response (in case there's extra text)
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                    
                # Validate required fields
                required_fields = ['key_contributions', 'introduces_new_model', 'introduces_new_dataset', 
                                 'introduces_new_benchmark', 'introduces_new_architecture', 'contribution']
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = "" if field in ['key_contributions', 'contribution'] else False
                
                return analysis, None
                
            except (json.JSONDecodeError, ValueError) as e:
                # Try to fix the JSON using LLM
                self.logger.log(f"  🔧 JSON parsing failed, attempting to clean up response...")
                try:
                    cleanup_prompt = self.create_json_cleanup_prompt(response)
                    cleaned_response = self.generate_llm_response(cleanup_prompt)
                    
                    # Try parsing the cleaned response
                    start_idx = cleaned_response.find('{')
                    end_idx = cleaned_response.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = cleaned_response[start_idx:end_idx]
                        analysis = json.loads(json_str)
                        
                        # Validate required fields
                        required_fields = ['key_contributions', 'introduces_new_model', 'introduces_new_dataset', 
                                         'introduces_new_benchmark', 'introduces_new_architecture', 'contribution']
                        for field in required_fields:
                            if field not in analysis:
                                analysis[field] = "" if field in ['key_contributions', 'contribution'] else False
                        
                        self.logger.log(f"  ✅ Successfully cleaned up JSON response")
                        return analysis, None
                    else:
                        raise ValueError("No JSON found in cleaned response")
                        
                except Exception as cleanup_error:
                    self.logger.log(f"  ❌ JSON cleanup also failed: {str(cleanup_error)}")
                    # Fall back to extracting what we can from the original response
                    return self.extract_fallback_analysis(response), f"JSON parsing and cleanup failed: {str(e)}"
                
        except Exception as e:
            self.cleanup_memory()
            default_analysis = {
                'key_contributions': "Error during analysis",
                'introduces_new_model': False,
                'introduces_new_dataset': False,
                'introduces_new_benchmark': False,
                'introduces_new_architecture': False,
                'contribution': "Error during analysis"
            }
            return default_analysis, f"Analysis error: {str(e)}"
    
    def extract_fallback_analysis(self, response):
        """Extract analysis from malformed response using simple text processing"""
        analysis = {
            'key_contributions': "",
            'introduces_new_model': False,
            'introduces_new_dataset': False,
            'introduces_new_benchmark': False,
            'introduces_new_architecture': False,
            'contribution': ""
        }
        
        # Try to extract key_contributions (look for first substantial text)
        lines = response.split('\n')
        substantial_text = ""
        for line in lines:
            line = line.strip()
            if len(line) > 50 and not line.startswith('{') and not line.startswith('}'):
                substantial_text = line
                break
        
        if substantial_text:
            analysis['key_contributions'] = substantial_text[:300] + "..." if len(substantial_text) > 300 else substantial_text
        else:
            analysis['key_contributions'] = response[:200] + "..." if len(response) > 200 else response
        
        # Try to extract boolean flags by looking for keywords
        response_lower = response.lower()
        
        # Look for model-related keywords
        if any(word in response_lower for word in ['new model', 'novel model', 'proposed model', 'introduces model']):
            analysis['introduces_new_model'] = True
            
        # Look for dataset-related keywords  
        if any(word in response_lower for word in ['new dataset', 'novel dataset', 'dataset', 'data collection']):
            analysis['introduces_new_dataset'] = True
            
        # Look for benchmark-related keywords
        if any(word in response_lower for word in ['benchmark', 'evaluation', 'new metric', 'novel metric']):
            analysis['introduces_new_benchmark'] = True
            
        # Look for architecture-related keywords
        if any(word in response_lower for word in ['architecture', 'framework', 'design', 'structure']):
            analysis['introduces_new_architecture'] = True
        
        # Try to extract contribution
        if 'contribution' in response_lower or 'differ' in response_lower:
            # Find text around these keywords
            for line in lines:
                if any(word in line.lower() for word in ['contribution', 'differ', 'novel', 'new']):
                    if len(line.strip()) > 20:
                        analysis['contribution'] = line.strip()[:200]
                        break
        
        if not analysis['contribution']:
            analysis['contribution'] = "Could not extract contribution from malformed response"
            
        return analysis

def save_checkpoint(analyzed_papers, processed_arxiv_ids, checkpoint_file, stats, logger):
    """Save progress checkpoint"""
    try:
        checkpoint_data = {
            'analyzed_papers': analyzed_papers,
            'processed_arxiv_ids': processed_arxiv_ids,
            'stats': stats
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.log(f"Checkpoint saved: {len(analyzed_papers)} analyzed, {len(processed_arxiv_ids)} processed")
    except Exception as e:
        logger.log(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file, logger):
    """Load progress checkpoint"""
    if not os.path.exists(checkpoint_file):
        return [], set(), {'analyzed': 0, 'errors': 0}
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        analyzed_papers = data.get('analyzed_papers', [])
        processed_arxiv_ids = set(data.get('processed_arxiv_ids', []))
        stats = data.get('stats', {'analyzed': 0, 'errors': 0})
        logger.log(f"Checkpoint loaded: {len(analyzed_papers)} analyzed, {len(processed_arxiv_ids)} already processed")
        return analyzed_papers, processed_arxiv_ids, stats
    except Exception as e:
        logger.log(f"Failed to load checkpoint: {e}")
        return [], set(), {'analyzed': 0, 'errors': 0}

def analyze_papers(input_file, output_file, batch_size=20, model_name="Qwen/Qwen3-4B-Thinking-2507", log_file=None):
    """Main analysis function"""
    # Initialize logger
    logger = Logger(log_file)
    
    # Load papers
    logger.log(f"Loading papers from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logger.log(f"Loaded {len(papers)} papers")
    
    # Initialize analyzer and checkpoint
    analyzer = LLMPaperAnalyzer(model_name=model_name, logger=logger)
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint.json"
    
    # Load existing progress
    analyzed_papers, processed_arxiv_ids, stats = load_checkpoint(checkpoint_file, logger)
    processed_count = 0
    
    try:
        # Process papers
        for i, paper in enumerate(papers):
            arxiv_id = paper.get('arxiv_id', f'paper_{i}')
            
            # Skip if already processed
            if arxiv_id in processed_arxiv_ids:
                continue
                
            title = paper.get('title', 'No title available')
            logger.log(f"Processing paper {i+1}/{len(papers)} (arxiv_id: {arxiv_id}): {title[:60]}{'...' if len(title) > 60 else ''}")
            
            # Analyze paper
            analysis, error = analyzer.analyze_paper(paper)
            
            # Add analysis fields to paper
            paper_with_analysis = paper.copy()
            paper_with_analysis.update(analysis)
            
            analyzed_papers.append(paper_with_analysis)
            
            if error:
                stats['errors'] += 1
                logger.log(f"  ⚠️  Analysis completed with error: {error}")
            else:
                stats['analyzed'] += 1
                logger.log(f"  ✅ Analysis completed successfully")
            
            # Mark as processed
            processed_arxiv_ids.add(arxiv_id)
            processed_count += 1
            
            # Save checkpoint every batch_size papers
            if processed_count % batch_size == 0:
                save_checkpoint(analyzed_papers, list(processed_arxiv_ids), checkpoint_file, stats, logger)
                logger.log(f"Processed {processed_count} new papers, analyzed {len(analyzed_papers)} total")
    
    finally:
        analyzer.unload_model()
        # Final checkpoint save
        save_checkpoint(analyzed_papers, list(processed_arxiv_ids), checkpoint_file, stats, logger)
    
    # Save final results
    logger.log(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analyzed_papers, f, indent=2, ensure_ascii=False)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Print results
    total = len(papers)
    logger.log(f"\n{'='*50}")
    logger.log(f"ANALYSIS RESULTS")
    logger.log(f"{'='*50}")
    logger.log(f"Total papers processed: {total}")
    logger.log(f"Successfully analyzed: {stats['analyzed']}")
    logger.log(f"Analysis errors: {stats['errors']}")
    logger.log(f"Final count: {len(analyzed_papers)} papers")
    logger.log(f"Saved to: {output_file}")
    if log_file:
        logger.log(f"Log saved to: {log_file}")
    logger.log(f"{'='*50}")
    
    return analyzed_papers

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Based Paper Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s papers.json
    -> Creates: papers_analyzed.json and papers_analyzed.log
  
  %(prog)s papers.json -o my_output.json
    -> Creates: my_output.json and my_output.log
  
  %(prog)s papers.json -o results.json --log custom.log --model Qwen/Qwen3-4B-Thinking-2507
    -> Creates: results.json and custom.log with specified model

Note: A log file is always created automatically. If not specified with --log,
      it will be named based on the output file (e.g., output.json -> output.log)
        """
    )
    
    # Required arguments
    parser.add_argument('input_file', help='JSON file containing papers to analyze')
    
    # Optional arguments
    parser.add_argument('-o', '--output', help='Output JSON file for analyzed papers (default: <input>_analyzed.json)')
    parser.add_argument('--log', help='Log file to save terminal output (default: <output>.log)')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for checkpoints')
    parser.add_argument('--model', default="Qwen/Qwen3-4B-Thinking-2507", 
                       help='HuggingFace model to use for analysis')
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Default output filename
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_analyzed.json"
    
    # Generate default log filename if not specified
    if args.log:
        log_file = args.log
    else:
        output_base = os.path.splitext(output_file)[0]
        log_file = f"{output_base}.log"
    
    print(f"LLM-Based Paper Analyzer")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)
    
    analyze_papers(args.input_file, output_file, args.batch_size, args.model, log_file)
    print("✅ Analysis completed!")

if __name__ == "__main__":
    main()
