import json
import requests
import time
import re
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from urllib.parse import urlparse
import os
from datetime import datetime
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataExtractor:
    def __init__(self, data_dir="data", output_file="processed_papers.json", request_delay=0.1):
        self.data_dir = data_dir
        self.output_file = output_file
        self.request_delay = request_delay
        self.processed_papers = []
        self.processing_log = {"processed": set(), "failed": set(), "skipped": set()}
        
        # NEW: Index of all processed papers by arxiv_id for fast lookup
        self.papers_index = {}  # arxiv_id -> paper_data
        
        # Cache for references to avoid redundant arXiv requests
        self.reference_cache = {}
        self.cache_file = 'reference_cache.json'
        self.load_reference_cache()
        
        # Thread locks for thread-safe operations
        self.cache_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.papers_index_lock = threading.Lock()  # NEW: Lock for papers index
        
        # Platform keywords for detection
        self.platform_keywords = {
            'Web': ['web', 'browser', 'HTML', 'DOM', 'website', 'HTTP', 'javascript', 'CSS'],
            'Mobile': ['mobile', 'smartphone', 'phone', 'app', 'cellular'],
            'Android': ['android', 'APK', 'google play', 'dalvik'],
            'iOS': ['iOS', 'iPhone', 'iPad', 'app store', 'swift', 'objective-c'],
            'Desktop': ['desktop', 'computer', 'PC', 'workstation'],
            'Windows': ['windows', 'win32', 'microsoft', 'DirectX', '.NET'],
            'macOS': ['macOS', 'mac', 'apple', 'cocoa', 'darwin'],
            'Linux': ['linux', 'ubuntu', 'unix', 'bash', 'shell']
        }
        
        # Category keywords for appears_in detection
        self.category_keywords = {
            'survey': ['survey', 'review', 'overview', 'comprehensive analysis', 'systematic review', 'literature review'],
            'dataset': ['dataset', 'corpus', 'collection', 'benchmark data', 'training data', 'evaluation data'],
            'models': ['model', 'architecture', 'neural network', 'transformer', 'framework', 'algorithm'],
            'benchmark': ['benchmark', 'evaluation', 'test suite', 'performance comparison', 'assessment', 'metric']
        }

    def load_and_deduplicate_papers(self):
        """Load all papers from JSON files and deduplicate"""
        logger.info("Loading and deduplicating papers from data directory...")
        
        all_papers = []
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(self.data_dir, filename)
            category = filename.replace('.json', '')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                
                for paper in papers:
                    paper['appears_in'] = [category]  # Track source file
                    all_papers.append(paper)
                
                logger.info(f"Loaded {len(papers)} papers from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        
        # Deduplicate based on Paper_Url
        deduplicated = {}
        for paper in all_papers:
            url = paper.get('Paper_Url', '')
            if url in deduplicated:
                # Merge appears_in lists
                deduplicated[url]['appears_in'].extend(paper['appears_in'])
                deduplicated[url]['appears_in'] = list(set(deduplicated[url]['appears_in']))
            else:
                deduplicated[url] = paper
        
        logger.info(f"Deduplicated {len(all_papers)} papers to {len(deduplicated)} unique papers")
        return list(deduplicated.values())

    def extract_arxiv_id(self, url):
        """Extract arXiv ID from URL"""
        if not url:
            return None
        
        # Common arXiv URL patterns
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',
            r'arxiv\.org/pdf/(\d+\.\d+)',
            r'arxiv\.org/html/(\d+\.\d+)',
            r'arXiv:(\d+\.\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None

    def parse_arxiv_date(self, arxiv_id):
        """Extract year and month from arXiv ID"""
        if not arxiv_id:
            return None, None
        
        try:
            # Format: YYMM.NNNNN or YYYY.NNNNN
            parts = arxiv_id.split('.')
            if len(parts) != 2:
                return None, None
            
            year_month = parts[0]
            
            if len(year_month) == 4:  # YYMM format
                year = 2000 + int(year_month[:2])
                month = int(year_month[2:])
            elif len(year_month) >= 4:  # YYYY format (newer papers)
                if '.' in arxiv_id and len(year_month) >= 6:
                    # This might be YYYYMM format
                    year = int(year_month[:4])
                    month = int(year_month[4:6]) if len(year_month) >= 6 else None
                else:
                    # Just year, try to extract from the decimal part or default
                    year = int(year_month)
                    month = None
            else:
                return None, None
            
            # Validate month
            if month and (month < 1 or month > 12):
                month = None
            
            return year, month
            
        except (ValueError, IndexError):
            return None, None

    def parse_platform_string(self, platform_str):
        """Parse platform string into clean list"""
        if not platform_str:
            return []
        
        # Clean and split the string
        clean_str = platform_str.lower()
        
        # Remove common noise words (using word boundaries to avoid cutting words like "Android")
        import re
        noise_patterns = [
            r'\bplatforms?\b',  # platform/platforms
            r'\bdevices?\b',    # device/devices  
            r'\band\b',         # and (word boundary)
            r'\bthe\b',         # the (word boundary)
            r'\bos\b',          # os (word boundary)
        ]
        for pattern in noise_patterns:
            clean_str = re.sub(pattern, ' ', clean_str, flags=re.IGNORECASE)
        
        # Split on common separators and whitespace
        separators = [',', '&', '+', '/', '|', ' ']
        parts = [clean_str]
        for sep in separators:
            new_parts = []
            for part in parts:
                if sep == ' ':
                    # For space, split on multiple whitespace and filter empty
                    new_parts.extend([p.strip() for p in re.split(r'\s+', part) if p.strip()])
                else:
                    new_parts.extend([p.strip() for p in part.split(sep) if p.strip()])
            parts = new_parts
        
        # Clean and standardize each part
        platforms = []
        standardization_map = {
            'web': 'Web',
            'browser': 'Web',
            'mobile': 'Mobile',
            'android': 'Android',
            'ios': 'iOS',
            'iphone': 'iOS',
            'ipad': 'iOS',
            'desktop': 'Desktop',
            'computer': 'Desktop',
            'pc': 'Desktop',
            'windows': 'Windows',
            'macos': 'macOS',
            'mac': 'macOS',
            'linux': 'Linux',
            'ubuntu': 'Linux',
            'unix': 'Linux'
        }
        
        for part in parts:
            # Clean whitespace and normalize
            part = re.sub(r'\s+', ' ', part.strip().lower())
            if part and len(part) > 1:
                # Try direct mapping first
                if part in standardization_map:
                    platforms.append(standardization_map[part])
                else:
                    # Capitalize first letter for unknown platforms
                    platforms.append(part.capitalize())
        
        return list(set(platforms))  # Remove duplicates

    def detect_platforms_from_content(self, content):
        """Detect platforms from paper content using keywords"""
        if not content:
            return []
        
        content_lower = content.lower()
        detected_platforms = []
        
        for platform, keywords in self.platform_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    detected_platforms.append(platform)
                    break  # Found one keyword for this platform, move to next
        
        return list(set(detected_platforms))

    def detect_categories_from_content(self, content):
        """Detect appears_in categories from paper content using keywords"""
        if not content:
            return []
        
        content_lower = content.lower()
        detected_categories = []
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    detected_categories.append(category)
                    break  # Found one keyword for this category, move to next
        
        return list(set(detected_categories))

    def add_paper_to_index(self, paper_data):
        """Add a paper to the index for fast lookup (thread-safe)"""
        with self.papers_index_lock:
            self.papers_index[paper_data['arxiv_id']] = paper_data
            logger.debug(f"Added paper to index: {paper_data['arxiv_id']}")

    def find_paper_by_arxiv_id(self, arxiv_id):
        """Find a paper by arxiv_id in the index (thread-safe)"""
        with self.papers_index_lock:
            return self.papers_index.get(arxiv_id)

    def update_paper_citations(self, arxiv_id, cited_by_list):
        """Update cited_by relationships for an existing paper (thread-safe)"""
        with self.papers_index_lock:
            if arxiv_id in self.papers_index:
                paper = self.papers_index[arxiv_id]
                if 'cited_by' not in paper:
                    paper['cited_by'] = []
                
                # Add new citations
                paper['cited_by'].extend(cited_by_list)
                # Remove duplicates while preserving order
                paper['cited_by'] = list(dict.fromkeys(paper['cited_by']))
                
                logger.info(f"Updated citations for {arxiv_id}: now cited by {len(paper['cited_by'])} papers")
                return True
        return False

    def get_all_papers_from_index(self):
        """Get all papers from the index as a list (thread-safe)"""
        with self.papers_index_lock:
            return list(self.papers_index.values())

    def update_backward_citations(self, newly_processed_papers):
        """Update backward citation relationships after processing a level (thread-safe)"""
        logger.info(f"Updating backward citations for {len(newly_processed_papers)} newly processed papers...")
        
        updates_made = 0
        
        for citing_paper in newly_processed_papers:
            citing_id = citing_paper['arxiv_id']
            
            # For each reference in this paper
            for ref in citing_paper.get('references', []):
                if ref.get('has_arxiv') and ref.get('arxiv_id'):
                    cited_arxiv_id = ref['arxiv_id']
                    
                    # Check if the cited paper is already in our index (processed)
                    cited_paper = self.find_paper_by_arxiv_id(cited_arxiv_id)
                    if cited_paper:
                        # Update the cited paper's cited_by list
                        with self.papers_index_lock:
                            if 'cited_by' not in cited_paper:
                                cited_paper['cited_by'] = []
                            
                            # Add the citing paper if not already there
                            if citing_id not in cited_paper['cited_by']:
                                cited_paper['cited_by'].append(citing_id)
                                updates_made += 1
                                logger.debug(f"Added backward citation: {cited_arxiv_id} ← {citing_id}")
        
        logger.info(f"✓ Updated {updates_made} backward citation relationships")
        return updates_made

    def extract_title_from_citation(self, citation_text):
        """Extract title from citation using improved pattern matching"""
        if not citation_text:
            return citation_text[:80] if len(citation_text) > 80 else citation_text
        
        # Step 1: Clean the citation by removing problematic suffixes
        clean_text = citation_text
        
        # Remove "arXiv preprint" and everything after it
        arxiv_patterns = [
            r'\s*arXiv preprint.*$',
            r'\s*arXiv:\d+\.\d+.*$',
            r'\s*arXiv\s+\d+\.\d+.*$'
        ]
        
        for pattern in arxiv_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
        
        # Remove URLs and DOIs
        url_patterns = [
            r'\s*https?://[^\s]+',  # HTTP/HTTPS URLs
            r'\s*www\.[^\s]+',      # www URLs
            r'\s*doi:[^\s]+',       # DOI identifiers
            r'\s*DOI:[^\s]+',       # DOI identifiers (caps)
        ]
        
        for pattern in url_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
        
        # Remove other common suffixes that aren't part of the title
        suffix_patterns = [
            r'\s*In\s+Proceedings.*$',
            r'\s*Proceedings\s+of.*$',
            r'\s*Journal\s+of.*$',
            r'\s*Conference\s+on.*$',
            r'\s*Association for.*$',  # "Association for Computational Linguistics"
            r'\s*\d+[-–]\d+\s*$',      # Page numbers like "74-81" or "74–81"
        ]
        
        for pattern in suffix_patterns:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
        
        # Step 2: Now extract title from the cleaned text
        
        # Pattern 1: Title in quotes (most reliable)
        quoted_pattern = r'"([^"]{5,150})"'
        quoted_match = re.search(quoted_pattern, clean_text)
        if quoted_match:
            title = quoted_match.group(1).strip()
            # Clean up trailing years
            title = re.sub(r',?\s*\d{4}\.?$', '', title).strip()
            return title
        
        # Pattern 2: NEW - Handle "Author et al. (YYYY) Full Author Names. YYYY. Title Here" format
        # This is specifically for the Carlini et al. case
        # Look for year followed by period, then title with common academic keywords
        year_dot_pattern = r'(\d{4})\.\s*([A-Z][^.]+(?:Models?|Vision|Learning|Analysis|System|Method|Framework|Study|Evaluation|Understanding|Interface|Agent|Language|Neural|Deep|Machine|Automatic|Reinforcement|Quantifying|Memorization|Approach|Training|Detection|Recognition|Planning)[^.]*)'
        year_dot_match = re.search(year_dot_pattern, clean_text, re.IGNORECASE)
        if year_dot_match:
            potential_title = year_dot_match.group(2).strip()
            # Clean up the title part
            potential_title = re.sub(r'\s+', ' ', potential_title)  # Normalize whitespace
            if (len(potential_title) > 10 and 
                not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+', potential_title) and  # Not "John Smith Brown"
                not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+[A-Z]', potential_title)):  # Not "John Smith, A"
                return potential_title
        
        # Pattern 3: IEEE Format - "Author, Title, Journal/Conference, details"
        # Look for: "J. Smith, Title here, Journal Name, vol. X, 2023"
        # Must have journal/conference indicators like "vol.", "pp.", "IEEE", "ACM", etc.
        ieee_pattern = r'^[A-Z]\.\s*[A-Z][a-z]+(?:\s+et\s+al\.)?[,\s]+([^,]{10,100})[,\s]+[^,]*(?:vol\.|pp\.|IEEE|ACM|Journal|Conference|Proceedings).*\d{4}'
        ieee_match = re.search(ieee_pattern, clean_text, re.IGNORECASE)
        if ieee_match:
            title = ieee_match.group(1).strip()
            # Clean up common IEEE artifacts
            title = re.sub(r'^"([^"]+)"$', r'\1', title)  # Remove quotes
            title = re.sub(r',?\s*\d{4}\.?$', '', title).strip()  # Remove years
            if len(title) > 5 and not re.match(r'^[A-Z]\.\s*[A-Z]', title):  # Not author initials
                return title
        
        # Pattern 4: Title after LAST year - "Author et al. 2021. Title here."
        # Find all years and use the last one to avoid author name confusion
        year_matches = list(re.finditer(r'\b\d{4}\b', clean_text))
        if year_matches:
            last_year_match = year_matches[-1]
            after_last_year = clean_text[last_year_match.end():].strip()
            if after_last_year.startswith('.'):
                after_last_year = after_last_year[1:].strip()
            
            # Look for title pattern after the last year
            # Split by periods and take the first substantial part
            parts = after_last_year.split('.')
            for part in parts:
                part = part.strip()
                if (len(part) > 10 and 
                    not re.match(r'^\d+', part) and  # Not starting with numbers
                    not re.match(r'^[A-Z]\w*,?\s+[A-Z]', part) and  # Not "Author, A"
                    not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+', part)):  # Not "John Smith Brown"
                    # Clean up trailing years
                    title = re.sub(r',?\s*\d{4}\.?$', '', part).strip()
                    # Normalize whitespace
                    title = re.sub(r'\s+', ' ', title)
                    return title
        
        # Pattern 5: Title after (year) - "Author (2021). Title here."
        # Use the last parenthetical year to avoid confusion
        paren_years = list(re.finditer(r'\(\d{4}\)', clean_text))
        if paren_years:
            last_paren_year = paren_years[-1]
            after_paren = clean_text[last_paren_year.end():].strip()
            if after_paren.startswith('.'):
                after_paren = after_paren[1:].strip()
            
            # Extract title after parenthetical year
            parts = after_paren.split('.')
            for part in parts:
                part = part.strip()
                if (len(part) > 10 and 
                    not re.match(r'^\d+', part) and
                    not re.match(r'^[A-Z]\w*,?\s+[A-Z]', part)):  # Not "Author, A"
                    # Clean up trailing years
                    title = re.sub(r',?\s*\d{4}\.?$', '', part).strip()
                    # Normalize whitespace
                    title = re.sub(r'\s+', ' ', title)
                    return title
        
        # Pattern 6: Simple sentence-based title extraction
        sentences = [s.strip() for s in clean_text.split('.') if s.strip()]
        
        # Look for sentences with title keywords (but not author lists)
        title_keywords = [
            'learning', 'model', 'system', 'analysis', 'approach', 'method', 'framework', 
            'study', 'evaluation', 'understanding', 'interface', 'web', 'mobile', 'gui', 
            'agent', 'language', 'vision', 'neural', 'deep', 'machine', 'automatic', 
            'reinforcement', 'using', 'planning', 'training', 'detection', 'recognition',
            'quantifying', 'memorization', 'across'
        ]
        
        for sentence in sentences:
            if (len(sentence) > 10 and 
                not re.match(r'^\(\d{4}\)', sentence) and  # Not starting with (year)
                not re.match(r'^\d{4}', sentence) and  # Not starting with year
                not re.match(r'^et al', sentence, re.IGNORECASE) and  # Not "et al"
                not sentence.lower().startswith('in ') and
                not sentence.lower().startswith('proceedings') and
                # Not author lists (improved heuristics)
                sentence.count(',') <= 2 and  # Not too many commas
                len(sentence.split()) <= 15 and  # Not too long
                not re.match(r'^[A-Z][a-z]+,?\s+[A-Z][a-z]+\s+[A-Z][a-z]+', sentence) and  # Not "John Smith Brown"
                not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+[A-Z]', sentence) and  # Not "John Smith, A"
                # Has title keywords (word boundaries)
                any(re.search(rf'\b{keyword.lower()}\b', sentence.lower()) for keyword in title_keywords)):
                # Clean up trailing years and normalize whitespace
                title = re.sub(r',?\s*\d{4}\.?$', '', sentence).strip()
                title = re.sub(r'\s+', ' ', title)
                return title
        
        # Second pass: Look for other reasonable titles
        for sentence in sentences:
            if (len(sentence) > 15 and 
                not re.match(r'^\(\d{4}\)', sentence) and  # Not starting with (year)
                not re.match(r'^\d{4}', sentence) and  # Not starting with year
                not re.match(r'^et al', sentence, re.IGNORECASE) and  # Not "et al"
                not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+', sentence) and  # Not "John Smith Brown"
                not re.match(r'^[A-Z]\w*,\s+[A-Z]\w+', sentence) and  # Not "Author, Name"
                not sentence.lower().startswith('in ') and
                not sentence.lower().startswith('proceedings') and
                sentence[0].isupper() and  # Starts with capital
                len(sentence.split()) >= 4):  # Reasonably long
                # Clean up trailing years and normalize whitespace
                title = re.sub(r',?\s*\d{4}\.?$', '', sentence).strip()
                title = re.sub(r'\s+', ' ', title)
                return title
        
        # Pattern 7: Look for capitalized phrases that could be titles
        # Find phrases that start with capital and contain title-like words
        title_phrases = re.findall(r'[A-Z][^.]{20,150}', clean_text)
        for phrase in title_phrases:
            if (not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+', phrase) and  # Not "John Smith Brown"
                any(word.lower() in phrase.lower() for word in ['learning', 'model', 'system', 'analysis', 'approach', 'method', 'framework', 'study', 'evaluation', 'understanding', 'interface', 'web', 'mobile', 'gui', 'agent', 'language', 'vision', 'neural', 'deep', 'machine', 'automatic', 'reinforcement', 'using', 'with', 'for', 'on', 'based', 'quantifying', 'memorization'])):
                # Clean up trailing years and normalize whitespace
                title = re.sub(r',?\s*\d{4}\.?$', '', phrase).strip()
                title = re.sub(r'\s+', ' ', title)
                return title
        
        # Ultimate fallback: take a reasonable chunk from the middle
        words = clean_text.split()
        if len(words) > 5:
            # Skip likely author names at the beginning, take middle portion
            start_idx = min(3, len(words) // 3)
            end_idx = min(start_idx + 15, len(words))
            return ' '.join(words[start_idx:end_idx])
        
        return clean_text[:80] if len(clean_text) > 80 else clean_text

    def load_reference_cache(self):
        """Load reference cache from file if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.reference_cache = json.load(f)
                logger.info(f"Loaded {len(self.reference_cache)} cached references from {self.cache_file}")
            else:
                self.reference_cache = {}
                logger.info("No reference cache found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load reference cache: {e}")
            self.reference_cache = {}

    def save_reference_cache(self):
        """Save reference cache to file (thread-safe)"""
        try:
            with self.cache_lock:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.reference_cache, f, indent=2)
                logger.debug(f"Saved {len(self.reference_cache)} references to cache")
        except Exception as e:
            logger.warning(f"Failed to save reference cache: {e}")

    def extract_section_content(self, soup, section_keywords, max_paragraphs=5, max_chars=2000):
        """Extract content from a section using multiple strategies"""
        content_parts = []

        # Strategy 1: Look for headers with specific classes and text
        for keyword in section_keywords:
            patterns = [
                re.compile(rf'\b{keyword}\b', re.IGNORECASE),
                re.compile(rf'{keyword}s?', re.IGNORECASE),
            ]

            for pattern in patterns:
                # Look for headers with ltx classes (common in arXiv HTML)
                headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                       class_=re.compile(r'ltx_title'))
                for header in headers:
                    if header.get_text() and pattern.search(header.get_text()):
                        content_parts = self._extract_content_after_header(header, max_paragraphs)
                        if content_parts:
                            break
                
                if content_parts:
                    break
                
                # Look for span tags with section tags
                spans = soup.find_all('span', class_='ltx_tag ltx_tag_section')
                for span in spans:
                    next_sibling = span.next_sibling
                    if next_sibling and hasattr(next_sibling, 'get_text'):
                        if pattern.search(next_sibling.get_text()):
                            header = span.find_parent(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                            if header:
                                content_parts = self._extract_content_after_header(header, max_paragraphs)
                                if content_parts:
                                    break
                
                if content_parts:
                    break
            
            if content_parts:
                break

        # Strategy 2: Fallback to text search
        if not content_parts and section_keywords:
            main_keyword = section_keywords[0]
            text_pattern = re.compile(rf'\b{main_keyword}\b[.\s]*', re.IGNORECASE)
            
            all_text_elements = soup.find_all(['p', 'div', 'section'])
            for element in all_text_elements:
                text = element.get_text()
                if text_pattern.search(text) and len(text) > 100:
                    content_parts.append(text)
                    
                    # Try to get following paragraphs
                    current = element.next_sibling
                    paragraph_count = 1
                    
                    while current and paragraph_count < max_paragraphs:
                        if hasattr(current, 'name') and current.name in ['p', 'div']:
                            next_text = current.get_text(separator=' ', strip=True)
                            if next_text and len(next_text) > 30:
                                if re.search(r'\b(Method|Result|Experiment|Related|Discussion|Conclusion)\b',
                                           next_text, re.IGNORECASE):
                                    break
                                content_parts.append(next_text)
                                paragraph_count += 1
                        current = current.next_sibling
                    break

        # Join and limit content
        if content_parts:
            full_content = ' '.join(content_parts)
            if len(full_content) > max_chars:
                full_content = full_content[:max_chars] + "..."
            return full_content

        return ""

    def _extract_content_after_header(self, header, max_paragraphs):
        """Helper to extract content after a header"""
        content_parts = []
        current = header.next_sibling
        paragraph_count = 0

        while current and paragraph_count < max_paragraphs:
            if hasattr(current, 'name'):
                if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                elif current.name in ['p', 'div', 'section']:
                    text = current.get_text(separator=' ', strip=True)
                    if text and len(text) > 30:
                        content_parts.append(text)
                        paragraph_count += 1
            current = current.next_sibling

        return content_parts

    def extract_references(self, soup):
        """Extract references from the paper with caching"""
        references = []
        
        # Look for bibliography section
        bib_section = soup.find(['section', 'div'], class_=re.compile(r'ltx_bibliography'))
        if not bib_section:
            # Fallback: look for References header
            ref_headers = soup.find_all(['h1', 'h2', 'h3'], string=re.compile(r'References?', re.IGNORECASE))
            if ref_headers:
                bib_section = ref_headers[0].find_next_sibling()
        
        if bib_section:
            ref_items = bib_section.find_all(['li', 'p'], class_=re.compile(r'ltx_bibitem|reference'))
            if not ref_items:
                # Fallback: find all list items in bibliography section
                ref_items = bib_section.find_all('li')
            
            for item in ref_items:
                ref_text = item.get_text(separator=' ', strip=True)
                if len(ref_text) > 20:  # Only substantial references
                    
                    # Clean up citation numbers like [31], (31), etc.
                    clean_ref_text = re.sub(r'^\s*[\[\(]?\d+[\]\)]?\s*', '', ref_text)
                    clean_ref_text = re.sub(r'^\s*\d+\.\s*', '', clean_ref_text)  # Remove "31. "
                    
                    # Extract title using structured citation format patterns
                    ref_title = self.extract_title_from_citation(clean_ref_text)
                    
                    # Extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
                    year = year_match.group(0) if year_match else ""
                    
                    # Check for arXiv ID (use original text for ID extraction)
                    arxiv_match = re.search(r'arXiv:(\d+\.\d+)', ref_text)
                    arxiv_id = arxiv_match.group(1) if arxiv_match else ""
                    
                    # Check cache first if we have an arXiv ID (thread-safe)
                    if arxiv_id:
                        with self.cache_lock:
                            if arxiv_id in self.reference_cache:
                                logger.debug(f"Using cached reference data for {arxiv_id}")
                                cached_ref = self.reference_cache[arxiv_id].copy()
                                # Update the full_text with current extraction (cleaned)
                                cached_ref['full_text'] = clean_ref_text
                                references.append(cached_ref)
                                continue
                    
                    # Extract month from arXiv ID if available
                    month = ""
                    if arxiv_id:
                        ref_year, ref_month = self.parse_arxiv_date(arxiv_id)
                        if not year:  # If we didn't find year in text, use arXiv year
                            year = str(ref_year) if ref_year else ""
                        month = str(ref_month) if ref_month else ""
                    
                    # Detect platform and appears_in from reference text using keyword matching
                    platform = self.detect_platforms_from_content(clean_ref_text)
                    appears_in = self.detect_categories_from_content(clean_ref_text)
                    
                    ref_data = {
                        'title': ref_title,
                        'year': year,
                        'month': month,
                        'arxiv_id': arxiv_id,
                        'has_arxiv': bool(arxiv_id),
                        'platform': platform,
                        'appears_in': appears_in,
                        'full_text': clean_ref_text
                    }
                    
                    references.append(ref_data)
                    
                    # Cache the reference if it has an arXiv ID (thread-safe)
                    if arxiv_id:
                        with self.cache_lock:
                            self.reference_cache[arxiv_id] = {k: v for k, v in ref_data.items() if k != 'full_text'}
                            logger.debug(f"Cached reference data for {arxiv_id}")
        
        return references

    def extract_references_from_arxiv(self, arxiv_id):
        """Extract references using multiple approaches"""
        references = []
        
        # Method 1: Try to get references from HTML (with better error handling)
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            html_url = f"https://arxiv.org/html/{arxiv_id}"
            response = requests.get(html_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                references = self.extract_references(soup)
                
                if references:
                    logger.info(f"Found {len(references)} references from HTML for {arxiv_id}")
                    return references
        except Exception as e:
            logger.debug(f"HTML reference extraction failed for {arxiv_id}: {e}")
        
        # Method 2: Try to extract from PDF text (simplified approach)
        try:
            # We could add PDF parsing here if needed, but for now return empty
            # This would require additional libraries like PyPDF2 or pdfplumber
            pass
        except Exception as e:
            logger.debug(f"PDF reference extraction failed for {arxiv_id}: {e}")
        
        # Method 3: Could add other reference extraction methods here
        # For now, we rely on HTML extraction from arXiv
        pass
        
        return references  # Return empty list if all methods fail

    def get_paper_data(self, arxiv_id, level=1, cited_by=None):
        """Get comprehensive paper data from arXiv HTML with robust extraction"""
        try:
            logger.info(f"Fetching level-{level} paper: {arxiv_id}")
            
            # Enhanced headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"macOS"',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Get paper metadata from abstract page
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            
            # Add delay to avoid rate limiting
            time.sleep(1.25)
            
            response = requests.get(abs_url, headers=headers, timeout=15)
            if response.status_code == 403:
                logger.warning(f"403 Forbidden for {arxiv_id} - may be rate limited")
            elif response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {arxiv_id}")
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title with multiple fallback strategies
            title = f"Paper {arxiv_id}"  # Default fallback
            
            # Strategy 1: Look for title in various formats
            title_selectors = [
                # Standard arXiv title formats
                ('h1.title', soup.find('h1', class_='title')),
                ('h1.title.mathjax', soup.find('h1', class_='title mathjax')),
                ('h1[class*="title"]', soup.find('h1', attrs={'class': re.compile('title', re.I)})),
                # Meta tag approach
                ('meta[name="citation_title"]', soup.find('meta', attrs={'name': 'citation_title'})),
                ('meta[property="og:title"]', soup.find('meta', attrs={'property': 'og:title'})),
                # Fallback to any h1
                ('h1', soup.find('h1'))
            ]
            
            for selector_name, element in title_selectors:
                if element:
                    if element.name == 'meta':
                        title_text = element.get('content', '')
                    else:
                        title_text = element.get_text().strip()
                    
                    if title_text and len(title_text) > 5 and not any(err in title_text.lower() for err in ['error', 'forbidden', 'not found']):
                        # Clean up the title
                        title = re.sub(r'^Title:\s*', '', title_text, flags=re.IGNORECASE).strip()
                        logger.debug(f"Found title via {selector_name}: {title[:50]}...")
                        break
            
            # Extract abstract with multiple fallback strategies
            abstract = ""
            
            abstract_selectors = [
                # Standard arXiv abstract formats
                ('blockquote.abstract', soup.find('blockquote', class_='abstract')),
                ('blockquote.abstract.mathjax', soup.find('blockquote', class_='abstract mathjax')),
                ('blockquote[class*="abstract"]', soup.find('blockquote', attrs={'class': re.compile('abstract', re.I)})),
                # Meta tag approach
                ('meta[name="citation_abstract"]', soup.find('meta', attrs={'name': 'citation_abstract'})),
                ('meta[name="description"]', soup.find('meta', attrs={'name': 'description'})),
                # Fallback to any blockquote
                ('blockquote', soup.find('blockquote'))
            ]
            
            for selector_name, element in abstract_selectors:
                if element:
                    if element.name == 'meta':
                        abstract_text = element.get('content', '')
                    else:
                        abstract_text = element.get_text().strip()
                    
                    if abstract_text and len(abstract_text) > 50 and not any(err in abstract_text.lower() for err in ['error', 'forbidden', 'not found']):
                        # Clean up the abstract
                        abstract = re.sub(r'^Abstract:\s*', '', abstract_text, flags=re.IGNORECASE).strip()
                        logger.debug(f"Found abstract via {selector_name}: {len(abstract)} chars")
                        break
            
            # Parse date from arXiv ID
            year, month = self.parse_arxiv_date(arxiv_id)
            
            # Initialize sections
            sections = {'abstract': abstract}
            references = []
            
            # Try to get enhanced content from HTML version
            time.sleep(1.25)  # Another delay before HTML request
            
            html_url = f"https://arxiv.org/html/{arxiv_id}"
            try:
                html_response = requests.get(html_url, headers=headers, timeout=15)
                if html_response.status_code == 200:
                    html_soup = BeautifulSoup(html_response.text, 'html.parser')
                    
                    # Extract various sections
                    section_configs = {
                        'introduction': ['Introduction', 'Intro'],
                        'related_work': ['Related Work', 'Related Works', 'Background', 'Prior Work'],
                        'methodology': ['Methodology', 'Method', 'Methods', 'Approach', 'Framework'],
                        'conclusion': ['Conclusion', 'Conclusions', 'Summary', 'Discussion', 'Final Remarks']
                    }
                    
                    for section_name, keywords in section_configs.items():
                        content = self.extract_section_content(html_soup, keywords)
                        if content:
                            sections[section_name] = content
                    
                    # Extract references
                    references = self.extract_references(html_soup)
                    logger.info(f"Found {len(references)} references from HTML")
                else:
                    logger.warning(f"HTML version returned {html_response.status_code} for {arxiv_id}")
            except Exception as e:
                logger.warning(f"HTML extraction failed for {arxiv_id}: {e}")
            
            # Create combined semantic text (clean, without prefixes)
            semantic_parts = []
            for section_name in ['abstract', 'introduction', 'related_work', 'methodology', 'conclusion']:
                if section_name in sections and sections[section_name]:
                    semantic_parts.append(sections[section_name])
            
            combined_semantic_text = ' '.join(semantic_parts)
            
            # For level 2 and level 3 papers, detect platform and categories from content
            platform = []
            appears_in = []
            
            if level >= 2:  # Level 2 and Level 3 papers
                platform = self.detect_platforms_from_content(combined_semantic_text)
                appears_in = self.detect_categories_from_content(combined_semantic_text)
            
            paper_data = {
                'arxiv_id': arxiv_id,
                'title': title,
                'year': year,
                'month': month,
                'paper_url': f"https://arxiv.org/abs/{arxiv_id}",
                'code_url': "",  # Will be empty for level 2 papers
                'platform': platform,
                'highlight': "",  # Will be empty for level 2 papers
                'appears_in': appears_in,
                'extraction_status': 'complete',
                'sections': sections,
                'semantic_text': combined_semantic_text,
                'references': references,
                'level': level,
                'cited_by': cited_by if cited_by else []
            }
            
            logger.info(f"✓ Successfully extracted: {title[:50]}... ({len(references)} refs, {len(sections)} sections)")
            return paper_data
            
        except Exception as e:
            logger.error(f"✗ Failed to extract {arxiv_id} (Level {level}): {type(e).__name__}: {e}")
            return None

    def process_papers(self):
        """Main processing pipeline"""
        logger.info("Starting enhanced data extraction pipeline...")
        
        # Load and deduplicate original papers
        original_papers = self.load_and_deduplicate_papers()
        
        # Process level 1 papers (original dataset)
        logger.info(f"Processing {len(original_papers)} level-1 papers...")
        level1_processed = []
        
        for i, paper in enumerate(original_papers):
            logger.info(f"[{i+1}/{len(original_papers)}] Processing level-1 paper...")
            
            arxiv_id = self.extract_arxiv_id(paper.get('Paper_Url', ''))
            if not arxiv_id:
                logger.warning(f"No arXiv ID found for: {paper.get('Name', 'Unknown')}")
                continue
            
            if arxiv_id in self.processing_log['processed']:
                logger.info(f"Already processed: {arxiv_id}")
                continue
            
            # Get enhanced data
            enhanced_data = self.get_paper_data(arxiv_id, level=1)
            if enhanced_data:
                # Merge with original data
                enhanced_data.update({
                    'code_url': paper.get('Code_Url', ''),
                    'highlight': paper.get('Highlight', ''),
                    'platform': self.parse_platform_string(paper.get('Platform', '')),
                    'appears_in': paper.get('appears_in', [])
                })
                
                level1_processed.append(enhanced_data)
                self.processing_log['processed'].add(arxiv_id)
                
                # NEW: Add to papers index for cross-citation tracking
                self.add_paper_to_index(enhanced_data)
            else:
                self.processing_log['failed'].add(arxiv_id)
            
            # Save progress periodically
            if (i + 1) % 5 == 0:
                self.save_progress(level1_processed)
                self.save_reference_cache()
        
        # FIXED: Update backward citations after Level-1 processing
        if level1_processed:
            self.update_backward_citations(level1_processed)
        
        # Collect arXiv references for level 2 processing
        logger.info("Collecting arXiv references for level-2 processing...")
        arxiv_references = {}
        
        for paper in level1_processed:
            for ref in paper['references']:
                if ref['has_arxiv'] and ref['arxiv_id']:
                    if ref['arxiv_id'] not in arxiv_references:
                        arxiv_references[ref['arxiv_id']] = {
                            'title': ref['title'],
                            'cited_by': []
                        }
                    arxiv_references[ref['arxiv_id']]['cited_by'].append(paper['arxiv_id'])
        
        logger.info(f"Found {len(arxiv_references)} unique arXiv references for level-2 processing")
        
        # Process level 2 papers
        level2_processed = []
        for i, (arxiv_id, ref_info) in enumerate(arxiv_references.items()):
            logger.info(f"[{i+1}/{len(arxiv_references)}] Processing level-2 paper: {arxiv_id}")
            
            if arxiv_id in self.processing_log['processed']:
                # FIXED: Don't skip! Update citation relationships for already processed papers
                logger.info(f"Already processed: {arxiv_id} - updating citation relationships")
                self.update_paper_citations(arxiv_id, ref_info['cited_by'])
                continue
            
            enhanced_data = self.get_paper_data(arxiv_id, level=2, cited_by=ref_info['cited_by'])
            if enhanced_data:
                level2_processed.append(enhanced_data)
                self.processing_log['processed'].add(arxiv_id)
                
                # NEW: Add to papers index for cross-citation tracking
                self.add_paper_to_index(enhanced_data)
            else:
                self.processing_log['failed'].add(arxiv_id)
            
            # Save progress periodically
            if (i + 1) % 10 == 0:
                self.save_progress(level1_processed + level2_processed)
                self.save_reference_cache()
        
        # FIXED: Update backward citations after Level-2 processing
        if level2_processed:
            self.update_backward_citations(level2_processed)
        
        # Collect arXiv references for level 3 processing
        logger.info("Collecting arXiv references for level-3 processing...")
        level3_arxiv_references = {}
        
        # FIXED: Collect Level-3 candidates from BOTH Level-1 AND Level-2 papers
        all_citing_papers = level1_processed + level2_processed
        
        for paper in all_citing_papers:
            for ref in paper['references']:
                if ref['has_arxiv'] and ref['arxiv_id']:
                    # Skip if already processed as Level-1 or Level-2
                    if ref['arxiv_id'] not in self.processing_log['processed']:
                        if ref['arxiv_id'] not in level3_arxiv_references:
                            level3_arxiv_references[ref['arxiv_id']] = {
                                'title': ref['title'],
                                'cited_by': []
                            }
                        level3_arxiv_references[ref['arxiv_id']]['cited_by'].append(paper['arxiv_id'])
        
        logger.info(f"Found {len(level3_arxiv_references)} unique arXiv references for level-3 processing")
        logger.info(f"  - Collected from {len(level1_processed)} Level-1 papers and {len(level2_processed)} Level-2 papers")
        
        # Process level 3 papers
        level3_processed = []
        for i, (arxiv_id, ref_info) in enumerate(level3_arxiv_references.items()):
            logger.info(f"[{i+1}/{len(level3_arxiv_references)}] Processing level-3 paper: {arxiv_id}")
            
            if arxiv_id in self.processing_log['processed']:
                # FIXED: Don't skip! Update citation relationships for already processed papers
                logger.info(f"Already processed: {arxiv_id} - updating citation relationships")
                self.update_paper_citations(arxiv_id, ref_info['cited_by'])
                continue
            
            enhanced_data = self.get_paper_data(arxiv_id, level=3, cited_by=ref_info['cited_by'])
            if enhanced_data:
                level3_processed.append(enhanced_data)
                self.processing_log['processed'].add(arxiv_id)
                
                # NEW: Add to papers index for cross-citation tracking
                self.add_paper_to_index(enhanced_data)
            else:
                self.processing_log['failed'].add(arxiv_id)
            
            # Save progress periodically
            if (i + 1) % 20 == 0:
                self.save_progress(level1_processed + level2_processed + level3_processed)
                self.save_reference_cache()
        
        # FIXED: Update backward citations after Level-3 processing
        if level3_processed:
            self.update_backward_citations(level3_processed)
        
        # Combine all processed papers
        all_processed = level1_processed + level2_processed + level3_processed
        
        # IMPORTANT: Get the final list of papers from the index to ensure all citation updates are included
        final_papers_list = self.get_all_papers_from_index()
        
        # Final save
        self.save_final_results(final_papers_list)
        self.save_reference_cache()
        
        logger.info(f"Completed processing: {len(level1_processed)} level-1, {len(level2_processed)} level-2, {len(level3_processed)} level-3 papers")
        
        return final_papers_list

    def process_paper_parallel(self, paper_info):
        """Process a single paper (for parallel execution)"""
        paper, level, cited_by = paper_info
        
        if level == 1:
            arxiv_id = self.extract_arxiv_id(paper.get('Paper_Url', ''))
            if not arxiv_id:
                logger.warning(f"No arXiv ID found for: {paper.get('Name', 'Unknown')}")
                return None
            
            # Check if already processed
            with self.processing_lock:
                if arxiv_id in self.processing_log['processed']:
                    logger.info(f"Already processed: {arxiv_id}")
                    return None
                
                # Mark as being processed
                self.processing_log['processed'].add(arxiv_id)
            
            # Get enhanced data
            enhanced_data = self.get_paper_data(arxiv_id, level=1)
            if enhanced_data:
                # Merge with original data
                enhanced_data.update({
                    'code_url': paper.get('Code_Url', ''),
                    'highlight': paper.get('Highlight', ''),
                    'platform': self.parse_platform_string(paper.get('Platform', '')),
                    'appears_in': paper.get('appears_in', [])
                })
                
                # NEW: Add to papers index for cross-citation tracking
                self.add_paper_to_index(enhanced_data)
                return enhanced_data
            else:
                with self.processing_lock:
                    self.processing_log['failed'].add(arxiv_id)
                return None
                
        else:  # level == 2 or level == 3
            arxiv_id = paper  # For level 2/3, paper is just the arxiv_id
            
            # Check if already processed
            with self.processing_lock:
                if arxiv_id in self.processing_log['processed']:
                    logger.info(f"Already processed (Level {level}): {arxiv_id} - updating citation relationships")
                    # FIXED: Don't return None! Update citation relationships
                    self.update_paper_citations(arxiv_id, cited_by)
                    return None  # Still return None to indicate no new paper was created
                
                # Mark as being processed
                self.processing_log['processed'].add(arxiv_id)
            
            enhanced_data = self.get_paper_data(arxiv_id, level=level, cited_by=cited_by)
            if enhanced_data:
                # NEW: Add to papers index for cross-citation tracking
                self.add_paper_to_index(enhanced_data)
                logger.debug(f"Successfully extracted Level {level} paper: {arxiv_id}")
                return enhanced_data
            else:
                with self.processing_lock:
                    self.processing_log['failed'].add(arxiv_id)
                logger.warning(f"Failed to extract Level {level} paper: {arxiv_id}")
                return None

    def process_papers_parallel(self, max_workers=4):
        """Main processing pipeline with parallel execution"""
        logger.info("Starting enhanced data extraction pipeline (parallel)...")
        
        # Load and deduplicate papers
        papers = self.load_and_deduplicate_papers()
        
        # Process level 1 papers in parallel
        logger.info(f"Processing {len(papers)} level-1 papers with {max_workers} workers...")
        level1_processed = []
        
        paper_infos = [(paper, 1, None) for paper in papers]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_paper = {executor.submit(self.process_paper_parallel, paper_info): paper_info 
                              for paper_info in paper_infos}
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_paper):
                completed += 1
                paper_info = future_to_paper[future]
                
                try:
                    result = future.result()
                    paper_info = future_to_paper[future]
                    paper = paper_info[0]  # paper_info is (paper, level, cited_by)
                    arxiv_id = self.extract_arxiv_id(paper.get('Paper_Url', ''))
                    paper_name = paper.get('Name', 'Unknown')
                    
                    if result:
                        level1_processed.append(result)
                        logger.info(f"[{completed}/{len(papers)}] ✓ Level-1: {result['title'][:50]}... ({arxiv_id})")
                    else:
                        # Check why it failed/was skipped
                        if not arxiv_id:
                            reason = f"no arXiv ID found in {paper.get('Paper_Url', 'N/A')}"
                        else:
                            with self.processing_lock:
                                if arxiv_id in self.processing_log['processed']:
                                    reason = "already processed"
                                elif arxiv_id in self.processing_log['failed']:
                                    reason = "extraction failed"
                                else:
                                    reason = "unknown reason"
                        logger.info(f"[{completed}/{len(papers)}] ✗ Level-1 {reason}: {paper_name} ({arxiv_id or 'no ID'})")
                        
                    # Save progress periodically
                    if completed % 10 == 0:
                        self.save_progress(level1_processed)
                        self.save_reference_cache()
                        
                except Exception as e:
                        paper_info = future_to_paper[future]
                        paper = paper_info[0]
                        paper_name = paper.get('Name', 'Unknown')
                        logger.error(f"Error processing Level-1 paper {paper_name}: {e}")
        
        # FIXED: Update backward citations after Level-1 processing
        if level1_processed:
            self.update_backward_citations(level1_processed)
        
        # Collect arXiv references for level 2 processing
        logger.info("Collecting arXiv references for level-2 processing...")
        arxiv_references = {}
        
        for paper in level1_processed:
            for ref in paper['references']:
                if ref['has_arxiv'] and ref['arxiv_id']:
                    if ref['arxiv_id'] not in arxiv_references:
                        arxiv_references[ref['arxiv_id']] = {
                            'title': ref['title'],
                            'cited_by': []
                        }
                    arxiv_references[ref['arxiv_id']]['cited_by'].append(paper['arxiv_id'])
        
        logger.info(f"Found {len(arxiv_references)} unique arXiv references for level-2 processing")
        
        # Process level 2 papers in parallel
        level2_processed = []
        
        if arxiv_references:
            paper_infos_l2 = [(arxiv_id, 2, ref_info['cited_by']) 
                             for arxiv_id, ref_info in arxiv_references.items()]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all level-2 tasks
                future_to_paper = {executor.submit(self.process_paper_parallel, paper_info): paper_info 
                                  for paper_info in paper_infos_l2}
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_paper):
                    completed += 1
                    paper_info = future_to_paper[future]
                    
                    try:
                        result = future.result()
                        paper_info = future_to_paper[future]
                        arxiv_id = paper_info[0]  # paper_info is (arxiv_id, level, cited_by)
                        
                        if result:
                            level2_processed.append(result)
                            logger.info(f"[{completed}/{len(arxiv_references)}] ✓ Level-2: {result['title'][:50]}... ({arxiv_id})")
                        else:
                            # Check why it failed/was skipped
                            with self.processing_lock:
                                if arxiv_id in self.processing_log['processed']:
                                    reason = "already processed"
                                elif arxiv_id in self.processing_log['failed']:
                                    reason = "extraction failed"
                                else:
                                    reason = "unknown reason"
                            logger.info(f"[{completed}/{len(arxiv_references)}] ✗ Level-2 {reason}: {arxiv_id}")
                            
                        # Save progress periodically
                        if completed % 20 == 0:
                            self.save_progress(level1_processed + level2_processed)
                            self.save_reference_cache()
                            
                    except Exception as e:
                        paper_info = future_to_paper[future]
                        arxiv_id = paper_info[0]
                        logger.error(f"Error processing level-2 paper {arxiv_id}: {e}")
        
        # FIXED: Update backward citations after Level-2 processing
        if level2_processed:
            self.update_backward_citations(level2_processed)
        
        # Collect arXiv references for level 3 processing
        logger.info("Collecting arXiv references for level-3 processing...")
        level3_arxiv_references = {}
        
        # FIXED: Collect Level-3 candidates from BOTH Level-1 AND Level-2 papers
        all_citing_papers = level1_processed + level2_processed
        
        for paper in all_citing_papers:
            for ref in paper['references']:
                if ref['has_arxiv'] and ref['arxiv_id']:
                    # Skip if already processed as Level-1 or Level-2
                    if ref['arxiv_id'] not in self.processing_log['processed']:
                        if ref['arxiv_id'] not in level3_arxiv_references:
                            level3_arxiv_references[ref['arxiv_id']] = {
                                'title': ref['title'],
                                'cited_by': []
                            }
                        level3_arxiv_references[ref['arxiv_id']]['cited_by'].append(paper['arxiv_id'])
        
        logger.info(f"Found {len(level3_arxiv_references)} unique arXiv references for level-3 processing")
        logger.info(f"  - Collected from {len(level1_processed)} Level-1 papers and {len(level2_processed)} Level-2 papers")
        
        # Process level 3 papers in parallel
        level3_processed = []
        
        if level3_arxiv_references:
            paper_infos_l3 = [(arxiv_id, 3, ref_info['cited_by']) 
                             for arxiv_id, ref_info in level3_arxiv_references.items()]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all level-3 tasks
                future_to_paper = {executor.submit(self.process_paper_parallel, paper_info): paper_info 
                                  for paper_info in paper_infos_l3}
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_paper):
                    completed += 1
                    paper_info = future_to_paper[future]
                    
                    try:
                        result = future.result()
                        paper_info = future_to_paper[future]
                        arxiv_id = paper_info[0]  # paper_info is (arxiv_id, level, cited_by)
                        
                        if result:
                            level3_processed.append(result)
                            logger.info(f"[{completed}/{len(level3_arxiv_references)}] ✓ Level-3: {result['title'][:50]}... ({arxiv_id})")
                        else:
                            # Check why it failed/was skipped
                            with self.processing_lock:
                                if arxiv_id in self.processing_log['processed']:
                                    reason = "already processed"
                                elif arxiv_id in self.processing_log['failed']:
                                    reason = "extraction failed"
                                else:
                                    reason = "unknown reason"
                            logger.info(f"[{completed}/{len(level3_arxiv_references)}] ✗ Level-3 {reason}: {arxiv_id}")
                            
                        # Save progress periodically
                        if completed % 50 == 0:
                            self.save_progress(level1_processed + level2_processed + level3_processed)
                            self.save_reference_cache()
                            
                    except Exception as e:
                        paper_info = future_to_paper[future]
                        arxiv_id = paper_info[0]
                        logger.error(f"Error processing level-3 paper {arxiv_id}: {e}")
        
        # FIXED: Update backward citations after Level-3 processing
        if level3_processed:
            self.update_backward_citations(level3_processed)
        
        # Combine all processed papers
        all_processed = level1_processed + level2_processed + level3_processed
        
        # IMPORTANT: Get the final list of papers from the index to ensure all citation updates are included
        final_papers_list = self.get_all_papers_from_index()
        
        # Final save
        self.save_final_results(final_papers_list)
        self.save_reference_cache()
        
        logger.info(f"Completed processing: {len(level1_processed)} level-1, {len(level2_processed)} level-2, {len(level3_processed)} level-3 papers")
        logger.info(f"Final output contains: {len(final_papers_list)} papers with updated citation relationships")
        
        return final_papers_list

    def save_progress(self, papers):
        """Save progress to avoid losing work"""
        temp_file = f"temp_{self.output_file}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        logger.info(f"Progress saved to {temp_file}")

    def save_final_results(self, papers):
        """Save final results"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        # Save processing log
        log_file = self.output_file.replace('.json', '_log.json')
        log_data = {
            'processed_count': len(self.processing_log['processed']),
            'failed_count': len(self.processing_log['failed']),
            'processed_papers': list(self.processing_log['processed']),
            'failed_papers': list(self.processing_log['failed'])
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"✓ Final results saved to {self.output_file}")
        logger.info(f"✓ Processing log saved to {log_file}")
        logger.info(f"✓ Successfully processed {len(papers)} papers total")
        logger.info(f"  - Level 1 papers: {len([p for p in papers if p['level'] == 1])}")
        logger.info(f"  - Level 2 papers: {len([p for p in papers if p['level'] == 2])}")
        logger.info(f"  - Level 3 papers: {len([p for p in papers if p['level'] == 3])}")


def main():
    """Run the enhanced data extraction"""
    parser = argparse.ArgumentParser(
        description="📄 Enhanced Data Extractor for LLM-Powered GUI Agents Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_extractor.py
  python data_extractor.py --parallel
  python data_extractor.py --parallel --workers 8
  python data_extractor.py --data-dir custom_data --output custom_output.json
  python data_extractor.py --parallel --delay 0.5 --workers 2
        """
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run in parallel mode (faster but more concurrent requests to arXiv)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: %(default)s, only used with --parallel)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing input JSON files (default: %(default)s)'
    )
    
    parser.add_argument(
        '--output',
        default='processed_papers.json',
        help='Output JSON file name (default: %(default)s)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=0.25,
        help='Request delay in seconds (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    extractor = EnhancedDataExtractor(
        data_dir=args.data_dir,
        output_file=args.output,
        request_delay=args.delay
    )
    
    try:
        if args.parallel:
            print(f"Running in PARALLEL mode with {args.workers} workers...")
            print("This will be much faster but will make more concurrent requests to arXiv")
            processed_papers = extractor.process_papers_parallel(max_workers=args.workers)
        else:
            print("Running in SEQUENTIAL mode...")
            print("Use --parallel flag for faster processing")
            processed_papers = extractor.process_papers()
            
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total papers processed: {len(processed_papers)}")
        print(f"Level 1 papers: {len([p for p in processed_papers if p['level'] == 1])}")
        print(f"Level 2 papers: {len([p for p in processed_papers if p['level'] == 2])}")
        print(f"Level 3 papers: {len([p for p in processed_papers if p['level'] == 3])}")
        print(f"Output saved to: {extractor.output_file}")
        print(f"Reference cache: {len(extractor.reference_cache)} entries")
        
    except KeyboardInterrupt:
        print("\nExtraction interrupted. Progress has been saved.")
    except Exception as e:
        print(f"Extraction failed: {e}")
        logger.error(f"Extraction failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()