import json
import os
import gc
import argparse
import threading
from datetime import datetime
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

class Logger:
    """Thread-safe logger that writes to both terminal and file."""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.lock = threading.Lock()
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Research Trend Analyzer Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
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

class ResearchTrendAnalyzer:
    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507", logger=None):
        """Initialize with GPU optimizations"""
        self.logger = logger or Logger()
        self.model_name = model_name
        
        # Check GPU availability and memory
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - this optimizer requires GPU")
        
        self.device = torch.device("cuda")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.logger.log(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        
        # Load model with optimizations
        self.logger.log(f"Loading model: {model_name}")
        self._load_model_optimized()
        
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
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Use flash attention if available
        )
        
        # Enable optimizations
        self.model.eval()
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            
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

    def group_papers_by_time_periods(self, papers):
        """Group papers into specific time periods"""
        time_periods = {}
        
        for paper in papers:
            year = paper.get('year')
            month = paper.get('month', 1)
            
            if not year:
                continue
                
            # 2016-2021: One bucket
            if 2016 <= year <= 2021:
                period_key = "2016-2021_Early_Era"
            # 2022: Own bucket
            elif year == 2022:
                period_key = "2022_Growth_Year"
            # 2023+: Quarterly
            elif year >= 2023:
                quarter = ((month - 1) // 3) + 1
                period_key = f"{year}_Q{quarter}"
            else:
                continue
                
            if period_key not in time_periods:
                time_periods[period_key] = []
            time_periods[period_key].append(paper)
        
        # Sort periods chronologically
        sorted_periods = {}
        for key in sorted(time_periods.keys()):
            sorted_periods[key] = time_periods[key]
            
        return sorted_periods
    
    def create_trend_analysis_prompt(self, period_key, papers):
        """Create analysis prompt for research trends in a time period"""
        paper_count = len(papers)
        
        # Extract key contributions and contributions
        contributions_data = []
        for i, paper in enumerate(papers):
            title = paper.get('title', 'No title available')
            key_contrib = paper.get('key_contributions', '')
            contribution = paper.get('contribution', '')
            
            # Include both fields if available
            paper_summary = f"Paper {i+1}: {title}\n"
            if key_contrib:
                paper_summary += f"Key Contributions: {key_contrib}\n"
            if contribution:
                paper_summary += f"Contribution vs Prior Work: {contribution}\n"
            
            contributions_data.append(paper_summary)
        
        papers_text = "\n---\n".join(contributions_data)
        
        return f"""You are an expert research analyst specializing in visual agents, OS agents, and GUI agents research. You are analyzing research trends across time periods to understand the evolution and pulse of the field.

TIME PERIOD: {period_key}
NUMBER OF PAPERS: {paper_count}

PAPERS IN THIS PERIOD:
{papers_text}

Please analyze the research trends in this time period and provide your analysis in the following JSON format:

{{
    "period": "{period_key}",
    "paper_count": {paper_count},
    "dominant_themes": ["list of 3-5 main research themes/areas of focus"],
    "key_innovations": ["list of 3-5 major innovations or breakthroughs"],
    "emerging_trends": ["list of 2-4 emerging trends or new directions"],
    "research_gaps": ["list of 2-3 potential areas needing more research"],
    "methodological_approaches": ["list of common methodologies/approaches used"],
    "evolution_summary": "2-3 sentence summary of how the field evolved in this period",
    "future_directions": "2-3 sentence prediction of where research might head based on these trends"
}}

Focus on:
1. What are researchers primarily working on?
2. What new approaches or methodologies are emerging?
3. What gaps or opportunities exist?
4. How is the field evolving technically and conceptually?

Respond with ONLY the JSON object, no additional text.
"""
    
    def generate_response(self, prompt):
        """Generate response for a single prompt"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
            ).to(self.device)
            
            # Generate response
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
            
            # Extract only the generated part
            input_length = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[0][input_length:].tolist()
            
            # Extract response (after thinking tokens)
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
            except ValueError:
                index = 0
            
            response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            
            # Cleanup
            del model_inputs, generated_ids
            self.cleanup_memory()
            time.sleep(15)  # Brief pause between generations
            
            return response
            
        except Exception as e:
            self.cleanup_memory()
            raise Exception(f"Generation error: {str(e)}")
    
    def parse_trend_analysis_response(self, response, period_key):
        """Parse JSON response with fallback handling"""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['period', 'paper_count', 'dominant_themes', 'key_innovations', 
                                 'emerging_trends', 'research_gaps', 'methodological_approaches',
                                 'evolution_summary', 'future_directions']
                for field in required_fields:
                    if field not in analysis:
                        if field in ['evolution_summary', 'future_directions']:
                            analysis[field] = "Could not extract from response"
                        elif field in ['paper_count']:
                            analysis[field] = 0
                        else:
                            analysis[field] = []
                
                return analysis, None
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            # Fallback analysis
            return self.create_fallback_analysis(response, period_key), f"JSON parsing failed: {str(e)}"
    
    def create_fallback_analysis(self, response, period_key):
        """Create fallback analysis from malformed response"""
        analysis = {
            'period': period_key,
            'paper_count': 0,
            'dominant_themes': ["Could not extract themes from response"],
            'key_innovations': ["Could not extract innovations from response"],
            'emerging_trends': ["Could not extract trends from response"],
            'research_gaps': ["Could not extract gaps from response"],
            'methodological_approaches': ["Could not extract approaches from response"],
            'evolution_summary': response[:300] + "..." if len(response) > 300 else response,
            'future_directions': "Could not extract future directions from response"
        }
        return analysis
    
    def analyze_time_period(self, period_key, papers):
        """Analyze trends for a single time period"""
        self.logger.log(f"\nAnalyzing period: {period_key} ({len(papers)} papers)")
        
        try:
            prompt = self.create_trend_analysis_prompt(period_key, papers)
            response = self.generate_response(prompt)
            analysis, error = self.parse_trend_analysis_response(response, period_key)
            
            if error:
                self.logger.log(f"  ⚠️  Parsing warning: {error}")
            else:
                self.logger.log(f"  ✅ Analysis completed successfully")
            
            return analysis
            
        except Exception as e:
            self.logger.log(f"  ❌ Analysis failed: {str(e)}")
            # Create error result
            return {
                'period': period_key,
                'paper_count': len(papers),
                'dominant_themes': [f"Analysis failed: {str(e)}"],
                'key_innovations': [],
                'emerging_trends': [],
                'research_gaps': [],
                'methodological_approaches': [],
                'evolution_summary': f"Analysis failed: {str(e)}",
                'future_directions': f"Analysis failed: {str(e)}",
                'analysis_error': str(e)
            }
    
    def unload_model(self):
        """Clean model unloading"""
        if hasattr(self, 'model') and self.model is not None:
            self.logger.log("Unloading model...")
            del self.model
            del self.tokenizer
            self.cleanup_memory()
            self.logger.log("Model unloaded successfully!")

def analyze_research_trends(input_file, output_file, model_name="Qwen/Qwen3-4B-Thinking-2507", log_file=None):
    """Analyze research trends across time periods"""
    # Initialize logger
    logger = Logger(log_file)
    
    # Load papers
    logger.log(f"Loading papers from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logger.log(f"Loaded {len(papers)} papers")
    
    # Initialize analyzer
    analyzer = ResearchTrendAnalyzer(model_name=model_name, logger=logger)
    
    # Group papers by time periods
    logger.log("Grouping papers by time periods...")
    time_periods = analyzer.group_papers_by_time_periods(papers)
    
    logger.log(f"Created {len(time_periods)} time periods:")
    for period, period_papers in time_periods.items():
        logger.log(f"  {period}: {len(period_papers)} papers")
    
    start_time = time.time()
    trend_analyses = []
    
    try:
        # Analyze each time period
        for i, (period_key, period_papers) in enumerate(time_periods.items()):
            period_start_time = time.time()
            
            analysis = analyzer.analyze_time_period(period_key, period_papers)
            trend_analyses.append(analysis)
            
            period_time = time.time() - period_start_time
            logger.log(f"  Period completed in {period_time:.1f}s")
            
            # Log progress
            logger.log(f"Progress: {i+1}/{len(time_periods)} periods completed")
            
            # Log GPU usage periodically
            if (i + 1) % 3 == 0:
                analyzer._log_gpu_usage()
    
    finally:
        analyzer.unload_model()
    
    # Save results
    logger.log(f"\nSaving trend analysis to {output_file}...")
    results = {
        'analysis_metadata': {
            'input_file': input_file,
            'total_papers': len(papers),
            'time_periods_analyzed': len(time_periods),
            'analysis_timestamp': datetime.now().isoformat(),
            'model_used': model_name
        },
        'time_period_trends': trend_analyses
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Final statistics
    total_time = time.time() - start_time
    logger.log(f"\n{'='*60}")
    logger.log(f"RESEARCH TREND ANALYSIS COMPLETE")
    logger.log(f"{'='*60}")
    logger.log(f"Total papers analyzed: {len(papers)}")
    logger.log(f"Time periods analyzed: {len(time_periods)}")
    logger.log(f"Total analysis time: {total_time/60:.1f} minutes")
    logger.log(f"Average time per period: {total_time/len(time_periods):.1f} seconds")
    logger.log(f"Results saved to: {output_file}")
    if log_file:
        logger.log(f"Log saved to: {log_file}")
    logger.log(f"{'='*60}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Research Trend Analyzer - Analyze research trends across time periods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Time Period Buckets:
- 2016-2021: Early Era (combined)
- 2022: Growth Year (standalone)  
- 2023+: Quarterly analysis (Q1, Q2, Q3, Q4)

Examples:
  %(prog)s keyword_filtered_enriched_qwen3_8b.json
  %(prog)s papers.json -o trends.json --model Qwen/Qwen3-4B-Thinking-2507
        """
    )
    
    parser.add_argument('input_file', help='JSON file containing enriched papers with key_contributions and contribution fields')
    parser.add_argument('-o', '--output', help='Output JSON file (default: <input>_trend_analysis.json)')
    parser.add_argument('--log', help='Log file (default: <output>.log)')
    parser.add_argument('--model', default="Qwen/Qwen3-4B-Thinking-2507", 
                       help='HuggingFace model to use')
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_trend_analysis.json"
    
    # Generate log filename
    if args.log:
        log_file = args.log
    else:
        output_base = os.path.splitext(output_file)[0]
        log_file = f"{output_base}.log"
    
    print(f"Research Trend Analyzer")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print(f"Model: {args.model}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This analyzer requires GPU.")
        return
    
    analyze_research_trends(
        args.input_file, 
        output_file, 
        args.model, 
        log_file
    )
    
    print("✅ Research trend analysis completed!")

if __name__ == "__main__":
    main()
