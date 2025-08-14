import json
import os
import gc
import argparse
import threading
import queue
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor
import time

class Logger:
    """Thread-safe logger that writes to both terminal and file."""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.lock = threading.Lock()
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== GPU-Optimized Paper Analyzer Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def log(self, message):
        """Thread-safe logging"""
        with self.lock:
            print(message)
            if self.log_file:
                try:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(message + '\n')
                except Exception as e:
                    print(f"Warning: Could not write to log file: {e}")

class PaperAnalyzer:
    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507", logger=None, batch_size=4):
        """Initialize with GPU optimizations"""
        self.logger = logger or Logger()
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Check GPU availability and memory
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - this optimizer requires GPU")
        
        self.device = torch.device("cuda")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.logger.log(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        
        # Load model with optimizations
        self.logger.log(f"Loading model: {model_name}")
        self._load_model_optimized()
        
        # Threading components for async processing
        self.processing_queue = queue.Queue(maxsize=batch_size * 2)
        self.result_queue = queue.Queue()
        self.stop_processing = threading.Event()
        
    def _load_model_optimized(self):
        """Load model with GPU optimizations"""
        # Use bfloat16 for memory efficiency if supported
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Optimize loading
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="cuda",
            low_cpu_mem_usage=True,  # Reduces CPU memory usage during loading
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Use flash attention if available
        )
        
        # Enable optimizations
        self.model.eval()  # Set to eval mode
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()  # Disable for inference
            
        self.logger.log("Model loaded with GPU optimizations")
        self._log_gpu_usage()
    
    def _log_gpu_usage(self):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            self.logger.log(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def create_analysis_prompt(self, paper_data):
        """Create analysis prompt for extracting key information"""
        title = paper_data.get('title', 'No title available')
        sections = paper_data.get('sections', {})
        abstract = sections.get('abstract', '')
        introduction = sections.get('introduction', '')
        conclusion = sections.get('conclusion', '')
        related_work = sections.get('related_work', sections.get('related work', ''))
        
        # Include full content without truncation
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
    "key_contributions": "Brief summary of the key research themes and contributions based on the Abstract, Introduction, and Conclusion",
    "introduces_new_model": true/false,
    "introduces_new_dataset": true/false, 
    "introduces_new_benchmark": true/false,
    "introduces_new_architecture": true/false,
    "contribution": "Summary of how this work differs from Related Work"
}}

Respond with ONLY the JSON object, no additional text.
"""
    
    def generate_batch_responses(self, prompts):
        """Generate responses for a batch of prompts efficiently"""
        try:
            # Prepare all messages
            all_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            
            # Apply chat templates
            texts = []
            for messages in all_messages:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                texts.append(text)
            
            # Tokenize batch with padding
            model_inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                padding_side="left",
                truncation=True,
            ).to(self.device)
            
            # Generate responses in batch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=32768,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # Decode all responses
            responses = []
            for i, generated_sequence in enumerate(generated_ids):
                # Extract only the generated part
                input_length = model_inputs.input_ids[i].shape[0]
                output_ids = generated_sequence[input_length:].tolist()
                
                # Extract response (after thinking tokens)
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
                except ValueError:
                    index = 0
                
                response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
                responses.append(response)
            
            # Cleanup
            del model_inputs, generated_ids
            self.cleanup_memory()
            time.sleep(25)
            return responses
            
        except Exception as e:
            self.cleanup_memory()
            raise Exception(f"Batch generation error: {str(e)}")
    
    def parse_analysis_response(self, response):
        """Parse JSON response with fallback handling"""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['key_contributions', 'introduces_new_model', 'introduces_new_dataset', 
                                 'introduces_new_benchmark', 'introduces_new_architecture', 'contribution']
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = "" if field in ['key_contributions', 'contribution'] else False
                
                return analysis, None
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            # Fallback analysis
            return self.create_fallback_analysis(response), f"JSON parsing failed: {str(e)}"
    
    def create_fallback_analysis(self, response):
        """Create fallback analysis from malformed response"""
        analysis = {
            'key_contributions': response[:200] + "..." if len(response) > 200 else response,
            'introduces_new_model': 'model' in response.lower(),
            'introduces_new_dataset': 'dataset' in response.lower(),
            'introduces_new_benchmark': 'benchmark' in response.lower() or 'evaluation' in response.lower(),
            'introduces_new_architecture': 'architecture' in response.lower() or 'framework' in response.lower(),
            'contribution': "Could not parse contribution from response"
        }
        return analysis
    
    def analyze_papers_batch(self, papers):
        """Analyze a batch of papers efficiently"""
        prompts = [self.create_analysis_prompt(paper) for paper in papers]
        
        try:
            # Generate all responses in one batch
            responses = self.generate_batch_responses(prompts)
            
            # Parse responses
            results = []
            for i, (paper, response) in enumerate(zip(papers, responses)):
                analysis, error = self.parse_analysis_response(response)
                
                # Add analysis to paper
                paper_with_analysis = paper.copy()
                paper_with_analysis.update(analysis)
                
                # Add error info if present
                if error:
                    paper_with_analysis['analysis_error'] = error
                
                results.append(paper_with_analysis)
            
            return results
            
        except Exception as e:
            # Fallback to individual processing
            self.logger.log(f"Batch processing failed, falling back to individual: {str(e)}")
            return self.analyze_papers_individual(papers)
    
    def analyze_papers_individual(self, papers):
        """Fallback individual paper analysis"""
        results = []
        for paper in papers:
            try:
                prompt = self.create_analysis_prompt(paper)
                response = self.generate_batch_responses([prompt])[0]
                analysis, error = self.parse_analysis_response(response)
                
                paper_with_analysis = paper.copy()
                paper_with_analysis.update(analysis)
                
                if error:
                    paper_with_analysis['analysis_error'] = error
                
                results.append(paper_with_analysis)
                
            except Exception as e:
                # Create error result
                paper_with_analysis = paper.copy()
                paper_with_analysis.update({
                    'key_contributions': f"Analysis failed: {str(e)}",
                    'introduces_new_model': False,
                    'introduces_new_dataset': False,
                    'introduces_new_benchmark': False,
                    'introduces_new_architecture': False,
                    'contribution': f"Analysis failed: {str(e)}",
                    'analysis_error': str(e)
                })
                results.append(paper_with_analysis)
        
        return results
    
    def unload_model(self):
        """Clean model unloading"""
        if hasattr(self, 'model') and self.model is not None:
            self.logger.log("Unloading model...")
            del self.model
            del self.tokenizer
            self.cleanup_memory()
            self.logger.log("Model unloaded successfully!")

def analyze_papers_gpu_optimized(input_file, output_file, batch_size=4, model_name="Qwen/Qwen3-4B-Thinking-2507", 
                                log_file=None, checkpoint_interval=50):
    """GPU-optimized paper analysis with batching"""
    # Initialize logger
    logger = Logger(log_file)
    
    # Load papers
    logger.log(f"Loading papers from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logger.log(f"Loaded {len(papers)} papers")
    logger.log(f"Using batch size: {batch_size}")
    
    # Initialize analyzer
    analyzer = PaperAnalyzer(model_name=model_name, logger=logger, batch_size=batch_size)
    
    # Load checkpoint if exists
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint.json"
    analyzed_papers, processed_count = load_checkpoint(checkpoint_file, logger)
    
    # Skip already processed papers
    remaining_papers = papers[processed_count:]
    logger.log(f"Resuming from paper {processed_count + 1}, {len(remaining_papers)} papers remaining")
    
    stats = {'analyzed': 0, 'errors': 0, 'total_time': 0}
    start_time = time.time()
    
    try:
        # Process papers in batches
        for i in range(0, len(remaining_papers), batch_size):
            batch_start_time = time.time()
            batch = remaining_papers[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(remaining_papers) + batch_size - 1) // batch_size
            
            logger.log(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} papers):")
            for j, paper in enumerate(batch):
                title = paper.get('title', 'No title available')
                logger.log(f"  {j+1}. {title[:60]}{'...' if len(title) > 60 else ''}")
            
            # Analyze batch
            try:
                batch_results = analyzer.analyze_papers_batch(batch)
                analyzed_papers.extend(batch_results)
                
                # Update stats
                batch_errors = sum(1 for result in batch_results if 'analysis_error' in result)
                stats['analyzed'] += len(batch_results) - batch_errors
                stats['errors'] += batch_errors
                
                batch_time = time.time() - batch_start_time
                stats['total_time'] += batch_time
                
                logger.log(f"  ✅ Batch completed in {batch_time:.1f}s ({len(batch_results) - batch_errors} successful, {batch_errors} errors)")
                
                # Log GPU usage periodically
                if batch_num % 5 == 0:
                    analyzer._log_gpu_usage()
                
            except Exception as e:
                logger.log(f"  ❌ Batch failed: {str(e)}")
                # Add error results for the batch
                for paper in batch:
                    error_result = paper.copy()
                    error_result.update({
                        'key_contributions': f"Batch analysis failed: {str(e)}",
                        'introduces_new_model': False,
                        'introduces_new_dataset': False,
                        'introduces_new_benchmark': False,
                        'introduces_new_architecture': False,
                        'contribution': f"Batch analysis failed: {str(e)}",
                        'analysis_error': str(e)
                    })
                    analyzed_papers.append(error_result)
                
                stats['errors'] += len(batch)
            
            # Save checkpoint
            if (i + batch_size) % (checkpoint_interval) == 0 or i + batch_size >= len(remaining_papers):
                save_checkpoint(analyzed_papers, processed_count + i + len(batch), checkpoint_file, logger)
                
                # Progress update
                current_progress = processed_count + i + len(batch)
                total_progress = len(papers)
                avg_time_per_paper = stats['total_time'] / max(1, current_progress - processed_count)
                remaining_time = avg_time_per_paper * (total_progress - current_progress)
                
                logger.log(f"\nProgress: {current_progress}/{total_progress} papers ({100*current_progress/total_progress:.1f}%)")
                logger.log(f"Estimated time remaining: {remaining_time/60:.1f} minutes")
                logger.log(f"Average time per paper: {avg_time_per_paper:.2f}s")
    
    finally:
        analyzer.unload_model()
    
    # Save final results
    logger.log(f"\nSaving final results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analyzed_papers, f, indent=2, ensure_ascii=False)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Final statistics
    total_time = time.time() - start_time
    logger.log(f"\n{'='*60}")
    logger.log(f"GPU-OPTIMIZED ANALYSIS COMPLETE")
    logger.log(f"{'='*60}")
    logger.log(f"Total papers: {len(papers)}")
    logger.log(f"Successfully analyzed: {stats['analyzed']}")
    logger.log(f"Errors: {stats['errors']}")
    logger.log(f"Total time: {total_time/60:.1f} minutes")
    logger.log(f"Average time per paper: {total_time/len(papers):.2f} seconds")
    logger.log(f"Throughput: {len(papers)/(total_time/3600):.1f} papers/hour")
    logger.log(f"Batch size used: {batch_size}")
    logger.log(f"Results saved to: {output_file}")
    if log_file:
        logger.log(f"Log saved to: {log_file}")
    logger.log(f"{'='*60}")
    
    return analyzed_papers

def save_checkpoint(analyzed_papers, processed_count, checkpoint_file, logger):
    """Save analysis checkpoint"""
    try:
        checkpoint_data = {
            'analyzed_papers': analyzed_papers,
            'processed_count': processed_count,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.log(f"Checkpoint saved: {len(analyzed_papers)} papers analyzed, {processed_count} processed")
    except Exception as e:
        logger.log(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file, logger):
    """Load analysis checkpoint"""
    if not os.path.exists(checkpoint_file):
        return [], 0
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        analyzed_papers = data.get('analyzed_papers', [])
        processed_count = data.get('processed_count', 0)
        timestamp = data.get('timestamp', 'unknown')
        logger.log(f"Checkpoint loaded from {timestamp}: {len(analyzed_papers)} papers, processed up to #{processed_count}")
        return analyzed_papers, processed_count
    except Exception as e:
        logger.log(f"Failed to load checkpoint: {e}")
        return [], 0

def main():
    parser = argparse.ArgumentParser(
        description="GPU-Optimized Paper Analyzer with Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Optimization Features:
- Batch processing for improved GPU utilization
- Memory-efficient model loading with fp16
- Flash attention support (if available)
- Automatic memory management and cleanup
- Dynamic batch sizing based on GPU memory

Examples:
  %(prog)s papers.json --batch-size 8
  %(prog)s papers.json -o results.json --batch-size 4 --model Qwen/Qwen3-4B-Thinking-2507
        """
    )
    
    parser.add_argument('input_file', help='JSON file containing papers to analyze')
    parser.add_argument('-o', '--output', help='Output JSON file (default: <input>_gpu_analyzed.json)')
    parser.add_argument('--log', help='Log file (default: <output>.log)')
    parser.add_argument('--batch-size', type=int, default=4, 
                       help='Batch size for GPU processing (default: 4, adjust based on GPU memory)')
    parser.add_argument('--model', default="Qwen/Qwen3-4B-Thinking-2507", 
                       help='HuggingFace model to use')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Save checkpoint every N papers (default: 50)')
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_enriched.json"
    
    # Generate log filename
    if args.log:
        log_file = args.log
    else:
        output_base = os.path.splitext(output_file)[0]
        log_file = f"{output_base}.log"
    
    print(f"GPU-Optimized Paper Analyzer")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This optimizer requires GPU.")
        return
    
    analyze_papers_gpu_optimized(
        args.input_file, 
        output_file, 
        args.batch_size, 
        args.model, 
        log_file,
        args.checkpoint_interval
    )
    
    print("✅ GPU-optimized analysis completed!")

if __name__ == "__main__":
    main()