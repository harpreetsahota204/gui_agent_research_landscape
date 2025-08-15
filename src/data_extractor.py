import json
import requests
import time
import re
import argparse
import os
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Set
import unicodedata
from difflib import SequenceMatcher

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Special papers not on arXiv but widely cited
SPECIAL_PAPERS = {
    'rico': {
        'title': 'Rico: A Mobile App Dataset for Building Data-Driven Design Applications',
        'year': 2017,
        'month':10,
        'platforms': ['Mobile', 'Android'],
        'cited_by': [],
        "sections": {
        "abstract": "Rico is presented, the largest repository of mobile app designs to date, created to support five classes of data-driven applications: design search, UI layout generation, UI code generation, user interaction modeling, and user perception prediction. ",
        "introduction": """Data-driven models of design can scaffold the creation of mobile apps. Having access to relevant examples helps designers understand best practices and trends. In the future, data-driven models will enable systems that can predict whether a design will achieve its specified goals before it is deployed to millions of people, and scale the creation of personalized designs that automatically adapt to diverse users and contexts. To build these models, researchers require design datasets which expose the details of mobile app designs at scale. This paper presents Rico, the largest repository of mobile app designs to date, comprising visual, textual, structural and interactive properties of UIs. These properties can be combined in different ways to support five classes of datadriven applications: design search, UI layout generation, UI code generation, user interaction modeling, and user perception prediction. Rico was built by mining Android apps at runtime via humanpowered and programmatic exploration. Like its predecessor ERICA, Rico’s app mining infrastructure requires no access to — or modification of — an app’s source code. Apps are downloaded from the Google Play Store and served to crowd workers through a web interface. When crowd workers use an app, the system records a user interaction trace that captures the UIs visited and the interactions performed on them. Then, an automated agent replays the trace to warm up a new copy of the app, and continues the exploration programmatically. By combining crowdsourcing and automation, Rico can achieve higher coverage over an app’s UI states than either crawling strategy alone. The Rico dataset contains design and interaction data for
72, 219 UIs from 9, 772 apps, spanning 27 Google Play categories. For each app, Rico presents a collection of individual user interaction traces, as well as a collection of unique UIs determined by a novel content-agnostic similarity heuristic.
Additionally, since the Rico dataset is large enough to support deep-learning applications, each UI is annotated with a low-dimensional vector produced by training an autoencoder for UI layout similarity, which can be used to cluster and retrieve similar UIs from different apps.""",
        "conclusion": "The are a number of opportunities to extend and improve the Rico dataset. New models could be trained to annotate Rico’s design components with richer labels, like classifiers that describe the semantic function of elements and screens (e.g., search, login). Similarly, researchers could crowdsource additional perceptual annotations (e.g., first impressions) over design components such as screenshots and animations, and use them to train newer types of perception-based predictive models. Unlike static research datasets such as ImageNet [12], Rico will become outdated over time if new apps are not continually crawled and their entries updated in the database. Therefore, another important avenue for future work is to explore ways to make app mining more sustainable. One potential path to sustainability is to create a platform where designers can use apps and contribute their traces to the repository for the entire community’s benefit."
        },
        'title_variants': [
            'rico a mobile app aataset for building data driven design applications',
        ]
    },
    'erica': {
        'title': 'ERICA: Interaction Mining Mobile Apps',
        'year': 2016,
        'month':10,
        'platforms': ['Mobile', 'Android'],
        'cited_by': [],
        "sections": {
        "abstract": """Design plays an important role in the adoption of apps. App design, however, is a complex process with multiple design activities. To enable data-driven app design applications, we present interaction mining – a method for capturing both static components (UI layouts, visual details) and dynamic components (user flows, motion details) of an app's design.
We present ERICA, a system that takes a scalable, human-computer approach to interaction mining existing Android apps without the need to modify them in any way. As users interact with apps through ERICA, the system detects UI changes, seamlessly records multiple data streams in the background, and unifies them into a comprehensive user interaction trace.
Using ERICA, we collected interaction traces from over a thousand popular Android apps. Leveraging this trace data, we built machine learning classifiers to detect elements and layouts indicative of 23 common user flows. User flows are an important component of user experience (UX) design and consist of a sequence of UI states that represent semantically meaningful tasks such as searching or composing.
With these classifiers, we identified and indexed more than 3,000 flow examples, and released the largest online search engine of user flows in Android apps. This research represents a significant advancement in understanding mobile app design patterns through automated interaction mining, providing designers and developers with unprecedented access to real-world app interaction data.""",
        "introduction": """Design plays an important role in the adoption of apps. App design, however, is a complex process comprised of multiple design activities: researchers, designers, and developers must all work together to identify user needs, create user flows (UX design), determine the proper layout of UI elements (UI design), and define their visual (visual design) and interactive (interaction design) properties. To create durable and engaging applications, app builders must consider hundreds of solutions from a vast space of design possibilities, prototype the most promising ones, and evaluate their effectiveness heuristically and through user testing.
To help navigate this complex process, this paper introduces interaction mining: capturing and analyzing both static (UI layouts, visual details) and dynamic (user flows, motion details) components of an application's design. Mined at scale, the data produced by interaction mining enables tools that scaffold the app design process: finding examples for design inspiration, understanding successful patterns and trends, generating new designs, and evaluating alternatives.
This paper takes a human-computer approach to interaction mining, using people to understand and interact with UIs, and machines to capture the UI states they explore. This approach is manifest in ERICA, a system for interaction mining Android apps which demonstrates, for the first time, a scalable way to mine the dynamic components of digital design. ERICA provides a web-based interface through which users interact with apps installed on Android devices. As a user navigates an app's interface, ERICA detects UI changes, seamlessly records screenshots, view hierarchies, and user events, and combines them into a unified representation called an interaction trace. ERICA requires no modifications to an app's source code or binary, making interaction mining possible for any Android app.
We used ERICA to collect user interaction traces from more than one thousand popular apps from the Google Play Store. These traces contain more than 18,000 unique UI screens, 50,000 user interactions, and half a million interactive elements. Leveraging this trace data, we built machine learning classifiers to detect elements and layouts indicative of 23 common user flows. User flows are important components of UX design, comprising sequences of UI states that represent semantically meaningful tasks such as searching or composing. With these classifiers, we identified and indexed more than 3,000 flow examples, and released the largest online search engine of user flows in mobile apps.""",
        "conclusion": """This paper demonstrates — for the first time — the possibility of mining user interactions and dynamic design data from mobile apps. Our system, ERICA, uses a web-based interface to allow crowdsourced data collection over the Internet. One important avenue for future work is to mine a more substantial portion of the available mobile apps, perhaps using crowd workers on platforms such as Amazon Mechanical Turk.
Another way to improve the scale of ERICA's repository would be to increase the coverage of UI states within each app. One way this might be accomplished is by combining multiple user traces. The research illustrates how coverage increased for three apps of varying complexity (measured by the number of Android activities found in their APK files) as multiple traces were combined. The observations show that, like in heuristic evaluation, 5-8 users appear to provide optimal coverage. For truly complex apps, coverage may remain low even after aggregating many user traces, since many UI states exist that humans do not visit during regular usage. Future work could explore augmenting ERICA with automated exploration strategies to visit these states.
While this paper focused on an application in UX design, interaction mining can be useful for building data-driven design tools targeted at other app design activities. The data produced by interaction mining can help designers understand UI layout patterns and motion details in existing apps. For example, heatmaps of common semantic element placements in UI layouts and animation curves of sliding drawer menus can be inferred using motion detection techniques on the changing images of the UI.
The UI data captured by interaction mining can also support learning probabilistic generative models of mobile designs. Such models could enable automated mobile UI generation — useful for building personalized interfaces and retargeting UIs across form factors. The UI data produced by ERICA even has enough information about elements and layouts that it can be used to reverse engineer the source code of existing app UIs, including rendering interfaces in different form factors.
Another important application area for interaction mining is usability testing. Interaction mining can help designers discover usability bugs, such as users mistaking a UI screen to be scrollable. Interaction data collected from a sufficiently large number of users could also enable summative usability testing for mobile apps without the need for source code modifications.
Outside of design, semantic understanding of apps enabled by interaction mining could improve the current metadata-based approaches to indexing and searching in online app stores. For example, learning a similarity metric over apps based on the types of user flows they expose could enable more accurate labeling and clustering. In addition, improved semantic understanding of mobile apps could enable automated identification of useful target states for deep-linking.
With Google's recent foray into web-based Android app streaming, mobile app usage through a web interface may become mainstream. In such a world, an approach like ERICA could enable interaction mining at truly massive scale."""
        },
        'title_variants': [
            'erica interaction mining mobile apps',
        ]
    }
}

class DataExtractor:
    """Enhanced data extractor with simplified architecture and improved performance."""
    
    def __init__(self, data_dir="data", output_file="processed_papers.json", request_delay=0.1):
        self.data_dir = data_dir
        self.output_file = output_file
        self.request_delay = request_delay
        
        # Simplified data structures
        self.papers_index = {}  # arxiv_id -> paper_data
        self.processed_ids = set()
        self.failed_ids = set()
        
        # Simplified caching
        self.reference_cache = {}
        self.cache_file = 'reference_cache.json'
        self.load_cache()
        
        # Performance tracking
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'start_time': time.time()
        }
        
        # Simplified platform keywords (consolidated)
        self.platforms = {
            'Web': ['web', 'browser', 'html', 'dom', 'website', 'javascript', 'css'],
            'Mobile': ['mobile', 'smartphone', 'phone', 'app'],
            'Android': ['android', 'apk', 'google play'],
            'iOS': ['ios', 'iphone', 'ipad', 'swift', 'objective-c'],
            'Desktop': ['desktop', 'computer', 'pc', 'workstation'],
            'Windows': ['windows', 'win32', 'microsoft'],
            'macOS': ['macos', 'mac', 'cocoa', 'darwin'],
            'Linux': ['linux', 'ubuntu', 'unix', 'bash']
        }
        
        # Simplified category detection
        self.categories = {
            'survey': ['survey', 'review', 'overview'],
            'dataset': ['dataset', 'corpus', 'collection', 'benchmark data'],
            'models': ['model', 'architecture', 'neural network', 'transformer'],
            'benchmark': ['benchmark', 'evaluation', 'test suite', 'metric']
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison - remove special chars, lowercase, etc."""
        if not text:
            return ""
        # Remove accents
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        # Lowercase and remove special characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def load_cache(self):
        """Load reference cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.reference_cache = json.load(f)
                logger.info(f"Loaded {len(self.reference_cache)} cached references")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.reference_cache = {}

    def save_cache(self):
        """Save reference cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.reference_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def save_progress(self):
        """Save current progress to avoid data loss."""
        papers = list(self.papers_index.values())
        temp_file = f"temp_{self.output_file}"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            logger.debug(f"Progress saved: {len(papers)} papers")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from URL or text - preserves version if present."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Single comprehensive pattern for all arXiv formats (including version)
        pattern = r'(?:arxiv[:\.]?\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)'
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
        
        # Old format (pre-2007) with optional version
        pattern_old = r'(?:arxiv[:\.]?\s*)?([a-z-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?)'
        match = re.search(pattern_old, text_lower)
        if match:
            return match.group(1)
        
        return None
    
    def get_arxiv_base_id(self, arxiv_id: str) -> str:
        """Get the base arXiv ID without version suffix."""
        if not arxiv_id:
            return arxiv_id
        return re.sub(r'v\d+$', '', arxiv_id)

    def parse_arxiv_date(self, arxiv_id: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract year and month from arXiv ID - simplified."""
        if not arxiv_id or '.' not in arxiv_id:
            return None, None
        
        try:
            # Remove version suffix if present before parsing date
            arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)
            parts = arxiv_id_clean.split('.')
            year_month = parts[0]
            
            # Handle YYMM format (most common)
            if len(year_month) == 4 and year_month.isdigit():
                year = 2000 + int(year_month[:2])
                month = int(year_month[2:])
                if 1 <= month <= 12:
                    return year, month
            
            return None, None
        except:
            return None, None

    def detect_platforms(self, text: str) -> List[str]:
        """Simplified platform detection."""
        if not text:
            return []
        
        text_lower = text.lower()
        detected = set()
        
        for platform, keywords in self.platforms.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.add(platform)
        
        return list(detected)

    def detect_categories(self, text: str) -> List[str]:
        """Simplified category detection."""
        if not text:
            return []
        
        text_lower = text.lower()
        detected = set()
        
        for category, keywords in self.categories.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.add(category)
        
        return list(detected)

    def extract_title_from_citation(self, citation: str) -> str:
        """Simplified title extraction - focus on the most effective patterns."""
        if not citation:
            return ""
        
        # Remove common suffixes that aren't part of titles
        citation = re.sub(r'\s*arXiv[:\s]+\d+\.\d+.*$', '', citation, flags=re.IGNORECASE)
        citation = re.sub(r'\s*https?://.*$', '', citation)
        citation = re.sub(r'\s*doi:.*$', '', citation, flags=re.IGNORECASE)
        citation = re.sub(r'\s*In\s+Proceedings.*$', '', citation, flags=re.IGNORECASE)
        citation = re.sub(r'\s*\d+[-–]\d+\s*$', '', citation)  # Page numbers
        
        # Pattern 1: Quoted title (most reliable)
        match = re.search(r'"([^"]{10,200})"', citation)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: After year with period
        match = re.search(r'\b\d{4}\.\s+([A-Z][^.]{10,150})', citation)
        if match:
            title = match.group(1).strip()
            # Basic validation - should contain some keywords
            if any(word in title.lower() for word in ['learning', 'model', 'system', 'vision', 
                                                       'analysis', 'method', 'framework', 'neural',
                                                       'detection', 'recognition', 'interface', 'agent']):
                return title
        
        # Pattern 3: After parenthetical year
        match = re.search(r'\(\d{4}\)[.\s]+([A-Z][^.]{10,150})', citation)
        if match:
            return match.group(1).strip()
        
        # Pattern 4: First reasonable sentence
        sentences = citation.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                sentence[0].isupper() and
                not re.match(r'^[A-Z][a-z]+,?\s+[A-Z]', sentence)):  # Not author names
                return sentence
        
        # Fallback: return first 100 chars
        return citation[:100].strip()

    def check_rico_citation(self, reference_text: str) -> bool:
        """Check if a reference is citing the Rico paper using fuzzy string matching."""
        normalized_ref = self.normalize_text(reference_text)
        
        # Get the actual Rico title and normalize it
        rico_title = self.normalize_text(SPECIAL_PAPERS['rico']['title'])
        
        # Use fuzzy string matching to check similarity
        similarity = SequenceMatcher(None, rico_title, normalized_ref).ratio()
        
        # If similarity is high enough, it's a match
        if similarity >= 0.6:  # 60% similarity threshold
            return True
        
        # Also check if the normalized title appears as a substring
        if rico_title in normalized_ref:
            return True
        
        return False
    
    def check_erica_citation(self, reference_text: str) -> bool:
        """Check if a reference is citing the ERICA paper using fuzzy string matching."""
        normalized_ref = self.normalize_text(reference_text)
        
        # Get the actual ERICA title and normalize it
        erica_title = self.normalize_text(SPECIAL_PAPERS['erica']['title'])
        
        # Use fuzzy string matching to check similarity
        similarity = SequenceMatcher(None, erica_title, normalized_ref).ratio()
        
        # If similarity is high enough, it's a match
        if similarity >= 0.6:  # 60% similarity threshold
            return True
        
        # Also check if the normalized title appears as a substring
        if erica_title in normalized_ref:
            return True
        
        return False


    def extract_references(self, soup: BeautifulSoup, citing_paper_id: str) -> List[Dict]:
        """Extract references from HTML - simplified and with Rico detection."""
        references = []
        
        # Find bibliography section
        bib_section = soup.find(['section', 'div'], class_=re.compile(r'ltx_bibliography'))
        if not bib_section:
            ref_headers = soup.find_all(['h1', 'h2', 'h3'], string=re.compile(r'References?', re.IGNORECASE))
            if ref_headers:
                bib_section = ref_headers[0].find_next_sibling()
        
        if not bib_section:
            return references
        
        # Extract reference items
        ref_items = bib_section.find_all(['li', 'p'])
        
        for item in ref_items:
            ref_text = item.get_text(separator=' ', strip=True)
            if len(ref_text) < 20:
                continue
            
            # Clean reference text
            ref_text = re.sub(r'^\s*[\[\(]?\d+[\]\)]?\s*', '', ref_text)
            
            # Check if this cites the Rico paper
            if self.check_rico_citation(ref_text):
                # Add citation to Rico paper
                if citing_paper_id not in SPECIAL_PAPERS['rico']['cited_by']:
                    SPECIAL_PAPERS['rico']['cited_by'].append(citing_paper_id)
                    logger.info(f"Found Rico citation in {citing_paper_id}")

            # Check if this cites the ERICA paper
            if self.check_erica_citation(ref_text):
                # Add citation to ERICA paper
                if citing_paper_id not in SPECIAL_PAPERS['erica']['cited_by']:
                    SPECIAL_PAPERS['erica']['cited_by'].append(citing_paper_id)
                    logger.info(f"Found ERICA citation in {citing_paper_id}")
            
            # Extract standard reference data
            arxiv_id = self.extract_arxiv_id(ref_text)
            
            # Use cache if available
            if arxiv_id and arxiv_id in self.reference_cache:
                self.stats['cache_hits'] += 1
                references.append(self.reference_cache[arxiv_id].copy())
                continue
            
            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
            year = year_match.group(0) if year_match else ""
            
            # Parse arXiv date if available
            month = ""
            if arxiv_id:
                ref_year, ref_month = self.parse_arxiv_date(arxiv_id)
                if not year and ref_year:
                    year = str(ref_year)
                if ref_month:
                    month = str(ref_month)
            
            ref_data = {
                'title': self.extract_title_from_citation(ref_text),
                'year': year,
                'month': month,
                'arxiv_id': arxiv_id or "",
                'has_arxiv': bool(arxiv_id),
                'platform': self.detect_platforms(ref_text),
                'appears_in': self.detect_categories(ref_text),
                'full_text': ref_text[:500]  # Limit stored text
            }
            
            references.append(ref_data)
            
            # Cache if has arXiv ID
            if arxiv_id:
                self.reference_cache[arxiv_id] = {k: v for k, v in ref_data.items() if k != 'full_text'}
        
        return references

    def extract_section_content(self, soup: BeautifulSoup, section_keywords: List[str], 
                               max_paragraphs: int = 5, max_chars: int = 2000) -> str:
        """Extract content from a section using multiple strategies."""
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

    def _extract_content_after_header(self, header, max_paragraphs: int) -> List[str]:
        """Helper to extract content after a header."""
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

    def extract_authors(self, soup: BeautifulSoup) -> List[str]:
        """Extract authors from the paper."""
        authors = []
        
        # Try multiple strategies to find authors
        # Strategy 1: Look for author meta tags
        author_metas = soup.find_all('meta', attrs={'name': 'citation_author'})
        if author_metas:
            for meta in author_metas:
                author_name = meta.get('content', '').strip()
                if author_name and author_name not in authors:
                    authors.append(author_name)
        
        # Strategy 2: Look for author divs with class
        if not authors:
            author_divs = soup.find_all('div', class_=re.compile(r'authors?', re.IGNORECASE))
            for div in author_divs:
                author_text = div.get_text(separator=', ', strip=True)
                # Split by common separators
                potential_authors = re.split(r'[,;]|\band\b', author_text)
                for author in potential_authors:
                    author = author.strip()
                    # Basic validation - should look like a name
                    if author and len(author) > 2 and not author.isdigit():
                        # Remove affiliations (numbers in parentheses)
                        author = re.sub(r'\([^)]*\)', '', author).strip()
                        if author and author not in authors:
                            authors.append(author)
        
        return authors

    def fetch_arxiv_paper(self, arxiv_id: str, level: int = 1) -> Optional[Dict]:
        """Fetch paper data from arXiv - enhanced with all essential information.
        
        Args:
            arxiv_id: The arXiv ID to fetch (may include version like 2412.10840v2)
            level: The processing level (1, 2, or 3)
        """
        
        # For deduplication, use base ID without version
        arxiv_id_base = self.get_arxiv_base_id(arxiv_id)
        
        # Check if already processed (using base ID for deduplication)
        if arxiv_id_base in self.processed_ids:
            return self.papers_index.get(arxiv_id_base)
        
        if arxiv_id_base in self.failed_ids:
            return None
        
        try:
            logger.info(f"Fetching L{level} paper: {arxiv_id}")
            self.stats['api_calls'] += 1
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            # Enhanced headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Cache-Control': 'no-cache'
            }
            
            # Get abstract page
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            response = requests.get(abs_url, headers=headers, timeout=10)
            
            if response.status_code == 403:
                logger.warning(f"Rate limited for {arxiv_id}, waiting...")
                time.sleep(25)
                response = requests.get(abs_url, headers=headers, timeout=10)
            
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title with multiple fallback strategies
            title = f"Paper {arxiv_id}"  # Default fallback
            
            title_selectors = [
                ('h1.title', soup.find('h1', class_='title')),
                ('h1.title.mathjax', soup.find('h1', class_='title mathjax')),
                ('h1[class*="title"]', soup.find('h1', attrs={'class': re.compile('title', re.I)})),
                ('meta[name="citation_title"]', soup.find('meta', attrs={'name': 'citation_title'})),
                ('meta[property="og:title"]', soup.find('meta', attrs={'property': 'og:title'})),
                ('h1', soup.find('h1'))
            ]
            
            for selector_name, element in title_selectors:
                if element:
                    if element.name == 'meta':
                        title_text = element.get('content', '')
                    else:
                        title_text = element.get_text().strip()
                    
                    if title_text and len(title_text) > 5:
                        title = re.sub(r'^Title:\s*', '', title_text, flags=re.IGNORECASE).strip()
                        break
            
            # Extract abstract with multiple fallback strategies
            abstract = ""
            
            abstract_selectors = [
                ('blockquote.abstract', soup.find('blockquote', class_='abstract')),
                ('blockquote.abstract.mathjax', soup.find('blockquote', class_='abstract mathjax')),
                ('blockquote[class*="abstract"]', soup.find('blockquote', attrs={'class': re.compile('abstract', re.I)})),
                ('meta[name="citation_abstract"]', soup.find('meta', attrs={'name': 'citation_abstract'})),
                ('meta[name="description"]', soup.find('meta', attrs={'name': 'description'})),
                ('blockquote', soup.find('blockquote'))
            ]
            
            for selector_name, element in abstract_selectors:
                if element:
                    if element.name == 'meta':
                        abstract_text = element.get('content', '')
                    else:
                        abstract_text = element.get_text().strip()
                    
                    if abstract_text and len(abstract_text) > 50:
                        abstract = re.sub(r'^Abstract:\s*', '', abstract_text, flags=re.IGNORECASE).strip()
                        break
            
            # Extract authors
            authors = self.extract_authors(soup)
            
            # Parse date
            year, month = self.parse_arxiv_date(arxiv_id)
            
            # Initialize sections with abstract
            sections = {'abstract': abstract}
            references = []
            
            # Initialize paper data
            paper_data = {
                'arxiv_id': arxiv_id,  # Store the actual ID (with version if present)
                'title': title,
                'authors': authors,
                'year': year,
                'month': month,
                'paper_url': abs_url,
                'code_url': '',  # Will be filled for level 1 papers
                'sections': sections,
                'references': [],
                'cited_by': [],
                'level': level,
                'platform': [],
                'appears_in': [],
                'highlight': '',  # Will be filled for level 1 papers
                'extraction_status': 'partial'
            }
            
            # Try to get HTML version for enhanced content
            time.sleep(self.request_delay)
            html_url = f"https://arxiv.org/html/{arxiv_id}"
            
            try:
                html_response = requests.get(html_url, headers=headers, timeout=10)
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
                    
                    # Update sections in paper data
                    paper_data['sections'] = sections
                    
                    # Extract references
                    paper_data['references'] = self.extract_references(html_soup, arxiv_id)
                    paper_data['extraction_status'] = 'complete'
                    
                    # Create combined text for platform/category detection
                    semantic_parts = []
                    for section_name in ['abstract', 'introduction', 'related_work', 'methodology', 'conclusion']:
                        if section_name in sections and sections[section_name]:
                            semantic_parts.append(sections[section_name])
                    
                    combined_text = ' '.join(semantic_parts)
                    
                    # Detect platforms and categories from combined text
                    paper_data['platform'] = self.detect_platforms(combined_text)
                    paper_data['appears_in'] = self.detect_categories(combined_text)
                    
                    logger.info(f"✓ Extracted full content: {len(sections)} sections, {len(paper_data['references'])} refs")
            except Exception as e:
                logger.debug(f"HTML extraction failed for {arxiv_id}: {e}")
                # Still use what we have from abstract page
                paper_data['platform'] = self.detect_platforms(abstract + ' ' + title)
                paper_data['appears_in'] = self.detect_categories(abstract + ' ' + title)
            
            # Mark as processed and add to index (using base ID for deduplication)
            self.processed_ids.add(arxiv_id_base)
            self.papers_index[arxiv_id_base] = paper_data
            
            logger.info(f"✓ Extracted: {title[:50]}... ({len(paper_data['references'])} refs, {len(authors)} authors)")
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to extract {arxiv_id}: {e}")
            self.failed_ids.add(arxiv_id_base)
            return None

    def load_papers(self) -> List[Dict]:
        """Load and deduplicate papers from JSON files."""
        logger.info("Loading papers from data directory...")
        
        papers_by_url = {}
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.data_dir, filename)
            category = filename.replace('.json', '')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                
                for paper in papers:
                    url = paper.get('Paper_Url', '')
                    if url not in papers_by_url:
                        paper['appears_in'] = [category]
                        papers_by_url[url] = paper
                    else:
                        papers_by_url[url]['appears_in'].append(category)
                
                logger.info(f"Loaded {len(papers)} papers from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        
        papers = list(papers_by_url.values())
        logger.info(f"Total unique papers: {len(papers)}")
        return papers

    def update_backward_citations(self, newly_processed_papers: List[Dict]) -> int:
        """Update backward citation relationships after processing a batch of papers."""
        logger.info(f"Updating backward citations for {len(newly_processed_papers)} newly processed papers...")
        
        updates_made = 0
        
        for citing_paper in newly_processed_papers:
            citing_id = citing_paper.get('arxiv_id')
            if not citing_id:
                continue
            
            # For each reference in this paper
            for ref in citing_paper.get('references', []):
                if ref.get('has_arxiv') and ref.get('arxiv_id'):
                    cited_arxiv_id = self.get_arxiv_base_id(ref['arxiv_id'])  # Use base ID for lookup
                    
                    # Check if the cited paper is already in our index (processed)
                    if cited_arxiv_id in self.papers_index:
                        cited_paper = self.papers_index[cited_arxiv_id]
                        
                        # Initialize cited_by if not present
                        if 'cited_by' not in cited_paper:
                            cited_paper['cited_by'] = []
                        
                        # Add the citing paper if not already there
                        if citing_id not in cited_paper['cited_by']:
                            cited_paper['cited_by'].append(citing_id)
                            updates_made += 1
                            logger.debug(f"Added backward citation: {cited_arxiv_id} ← {citing_id}")
        
        logger.info(f"✓ Updated {updates_made} backward citation relationships")
        return updates_made

    def update_citations(self):
        """Update all citation relationships - comprehensive version."""
        logger.info("Updating all citation relationships...")
        updates = 0
        
        # First pass: update forward citations and collect all relationships
        for paper in self.papers_index.values():
            citing_id = paper.get('arxiv_id')
            if not citing_id:
                continue  # Skip papers without arxiv_id (like Rico/ERICA)
            
            for ref in paper.get('references', []):
                if ref.get('has_arxiv') and ref['arxiv_id']:
                    cited_id = self.get_arxiv_base_id(ref['arxiv_id'])  # Use base ID for lookup
                    
                    # Update backward citation if cited paper is in our index
                    if cited_id in self.papers_index:
                        cited_paper = self.papers_index[cited_id]
                        if 'cited_by' not in cited_paper:
                            cited_paper['cited_by'] = []
                        
                        if citing_id not in cited_paper['cited_by']:
                            cited_paper['cited_by'].append(citing_id)
                            updates += 1
        
        # Special handling for Rico and ERICA papers
        for special_key in ['rico_2017', 'erica_2016']:
            if special_key in self.papers_index:
                special_paper = self.papers_index[special_key]
                if 'cited_by' not in special_paper:
                    special_paper['cited_by'] = []
                
                # Get citations from the SPECIAL_PAPERS data
                paper_type = special_key.split('_')[0]
                if paper_type in SPECIAL_PAPERS:
                    special_citations = SPECIAL_PAPERS[paper_type]['cited_by']
                    for citing_id in special_citations:
                        if citing_id not in special_paper['cited_by']:
                            special_paper['cited_by'].append(citing_id)
                            updates += 1
        
        logger.info(f"Updated {updates} total citation relationships")

    def process_papers(self, max_workers: int = 4) -> List[Dict]:
        """Main processing pipeline - simplified and unified."""
        logger.info("Starting data extraction pipeline...")
        start_time = time.time()
        
        # Load original papers
        original_papers = self.load_papers()
        
        # Add special papers (Rico and ERICA) to index
        for paper_key, paper_data in SPECIAL_PAPERS.items():
            special_paper = paper_data.copy()
            special_paper['arxiv_id'] = None
            special_paper['level'] = 1  # Special level for external papers
            special_paper['references'] = []
            special_paper['extraction_status'] = 'external'
            special_paper['paper_url'] = ''
            special_paper['code_url'] = ''
            special_paper['authors'] = []
            special_paper['highlight'] = ''
            special_paper['appears_in'] = ['dataset'] if paper_key == 'rico' else ['models']
            self.papers_index[f'{paper_key}_{special_paper["year"]}'] = special_paper
        
        # Process Level 1 papers
        logger.info(f"Processing {len(original_papers)} Level-1 papers...")
        
        level1_refs = defaultdict(list)  # arxiv_id -> citing_papers
        level1_processed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for paper in original_papers:
                arxiv_id = self.extract_arxiv_id(paper.get('Paper_Url', ''))
                if arxiv_id:
                    future = executor.submit(self.fetch_arxiv_paper, arxiv_id, 1)
                    futures.append((future, paper, arxiv_id))
            
            for i, (future, orig_paper, arxiv_id) in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        # Merge with original data - ensure all fields are preserved
                        result['code_url'] = orig_paper.get('Code_Url', '')
                        result['highlight'] = orig_paper.get('Highlight', '')
                        
                        # For appears_in, merge with detected categories
                        original_appears_in = orig_paper.get('appears_in', [])
                        if original_appears_in:
                            result['appears_in'] = list(set(result.get('appears_in', []) + original_appears_in))
                        
                        # For platform, prefer original if available, otherwise use detected
                        original_platform = orig_paper.get('Platform', '')
                        if original_platform:
                            # Parse the platform string from original data
                            parsed_platforms = []
                            for p in original_platform.split(','):
                                p = p.strip()
                                if p:
                                    parsed_platforms.append(p)
                            if parsed_platforms:
                                result['platform'] = list(set(parsed_platforms + result.get('platform', [])))
                        
                        level1_processed.append(result)
                        
                        # Collect Level-2 candidates
                        for ref in result['references']:
                            if ref.get('has_arxiv') and ref['arxiv_id']:
                                level1_refs[ref['arxiv_id']].append(arxiv_id)
                    
                    # Save progress periodically
                    if (i + 1) % 20 == 0:
                        self.save_progress()
                        self.save_cache()
                        logger.info(f"Progress: {i+1}/{len(futures)} Level-1 papers")
                        
                except Exception as e:
                    logger.error(f"Error processing {arxiv_id}: {e}")
        
        # Update backward citations after Level-1 processing
        if level1_processed:
            self.update_backward_citations(level1_processed)
        
        # Process Level 2 papers
        logger.info(f"Processing {len(level1_refs)} Level-2 papers...")
        
        level2_refs = defaultdict(list)
        level2_processed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for arxiv_id, citing_papers in level1_refs.items():
                if self.get_arxiv_base_id(arxiv_id) not in self.processed_ids:
                    future = executor.submit(self.fetch_arxiv_paper, arxiv_id, 2)
                    futures.append((future, arxiv_id, citing_papers))
                else:
                    # Update citation relationships for already processed papers
                    base_id = self.get_arxiv_base_id(arxiv_id)
                    if base_id in self.papers_index:
                        existing_paper = self.papers_index[base_id]
                        if 'cited_by' not in existing_paper:
                            existing_paper['cited_by'] = []
                        for citing_id in citing_papers:
                            if citing_id not in existing_paper['cited_by']:
                                existing_paper['cited_by'].append(citing_id)
            
            for i, (future, arxiv_id, citing_papers) in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        result['cited_by'] = citing_papers
                        level2_processed.append(result)
                        
                        # Collect Level-3 candidates
                        for ref in result['references']:
                            if ref.get('has_arxiv') and ref['arxiv_id']:
                                level2_refs[ref['arxiv_id']].append(arxiv_id)
                    
                    # Save progress periodically
                    if (i + 1) % 50 == 0:
                        self.save_progress()
                        self.save_cache()
                        logger.info(f"Progress: {i+1}/{len(futures)} Level-2 papers")
                        
                except Exception as e:
                    logger.error(f"Error processing L2 {arxiv_id}: {e}")
        
        # Update backward citations after Level-2 processing
        if level2_processed:
            self.update_backward_citations(level2_processed)
        
        # Process Level 3 papers
        logger.info(f"Processing {len(level2_refs)} Level-3 papers...")
        
        level3_processed = []
        level3_candidates = dict(list(level2_refs.items()))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for arxiv_id, citing_papers in level3_candidates.items():
                if self.get_arxiv_base_id(arxiv_id) not in self.processed_ids:
                    future = executor.submit(self.fetch_arxiv_paper, arxiv_id, 3)
                    futures.append((future, arxiv_id, citing_papers))
                else:
                    # Update citation relationships for already processed papers
                    base_id = self.get_arxiv_base_id(arxiv_id)
                    if base_id in self.papers_index:
                        existing_paper = self.papers_index[base_id]
                        if 'cited_by' not in existing_paper:
                            existing_paper['cited_by'] = []
                        for citing_id in citing_papers:
                            if citing_id not in existing_paper['cited_by']:
                                existing_paper['cited_by'].append(citing_id)
            
            for i, (future, arxiv_id, citing_papers) in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        result['cited_by'] = citing_papers
                        level3_processed.append(result)
                    
                    # Save progress periodically
                    if (i + 1) % 100 == 0:
                        self.save_progress()
                        self.save_cache()
                        logger.info(f"Progress: {i+1}/{len(futures)} Level-3 papers")
                        
                except Exception as e:
                    logger.error(f"Error processing L3 {arxiv_id}: {e}")
        
        # Update backward citations after Level-3 processing
        if level3_processed:
            self.update_backward_citations(level3_processed)
        
        # Update all citation relationships
        self.update_citations()
        
        # Get final results
        final_papers = list(self.papers_index.values())
        
        # Save final results
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_papers, f, indent=2, ensure_ascii=False)
        
        # Save cache
        self.save_cache()
        
        # Log statistics
        elapsed = time.time() - start_time
        rico_citations = len(SPECIAL_PAPERS['rico']['cited_by'])
        erica_citations = len(SPECIAL_PAPERS['erica']['cited_by'])
        
        logger.info(f"""
        ========== Extraction Complete ==========
        Total papers: {len(final_papers)}
        - Level 0 (external): {len([p for p in final_papers if p.get('level') == 0])}
        - Level 1: {len([p for p in final_papers if p['level'] == 1])}
        - Level 2: {len([p for p in final_papers if p['level'] == 2])}
        - Level 3: {len([p for p in final_papers if p['level'] == 3])}
        
        Special paper citations:
        - Rico citations found: {rico_citations}
        - ERICA citations found: {erica_citations}
        - Total special citations: {rico_citations + erica_citations}
        
        Performance:
        - API calls: {self.stats['api_calls']}
        - Cache hits: {self.stats['cache_hits']}
        - Time elapsed: {elapsed:.1f}s
        - Papers/minute: {len(final_papers) * 60 / elapsed:.1f}
        
        Output: {self.output_file}
        ========================================
        """)
        
        return final_papers


def main():
    """Run the enhanced data extraction."""
    parser = argparse.ArgumentParser(
        description="Enhanced Data Extractor for Research Papers"
    )
    
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing input JSON files')
    parser.add_argument('--output', default='processed_papers.json',
                       help='Output JSON file name')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Request delay in seconds')
    
    args = parser.parse_args()
    
    extractor = DataExtractor(
        data_dir=args.data_dir,
        output_file=args.output,
        request_delay=args.delay
    )
    
    try:
        processed_papers = extractor.process_papers(max_workers=args.workers)
        print(f"\n✓ Successfully processed {len(processed_papers)} papers")
        print(f"✓ Output saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nExtraction interrupted. Progress has been saved.")
    except Exception as e:
        print(f"Extraction failed: {e}")
        logger.error(f"Extraction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
