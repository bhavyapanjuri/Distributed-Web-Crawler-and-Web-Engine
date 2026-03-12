import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import defaultdict, deque
import hashlib
import math
import re
from concurrent.futures import ThreadPoolExecutor
import time

class WebCrawlerSearchEngine:
    def __init__(self, max_pages=50, max_workers=5):
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.visited_urls = set()
        self.url_queue = deque()
        self.pages = {}  # url -> {title, content, links, hash}
        self.content_hashes = set()
        self.inverted_index = defaultdict(set)
        self.pagerank_scores = {}
        self.link_graph = defaultdict(set)
        
    def normalize_url(self, url):
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
    
    def hash_content(self, content):
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def crawl_page(self, url):
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content
            title = soup.title.string if soup.title else ""
            for script in soup(["script", "style"]):
                script.decompose()
            content = soup.get_text(separator=' ', strip=True)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(url, link['href'])
                normalized = self.normalize_url(absolute_url)
                if normalized.startswith('http'):
                    links.append(normalized)
            
            content_hash = self.hash_content(content)
            
            return {
                'title': title,
                'content': content,
                'links': links,
                'hash': content_hash
            }
        except Exception as e:
            return None
    
    def crawl(self, seed_urls):
        print(f"🚀 Starting crawl with {self.max_workers} workers...")
        self.url_queue.extend(seed_urls)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while self.url_queue and len(self.pages) < self.max_pages:
                batch = []
                for _ in range(min(self.max_workers, len(self.url_queue))):
                    if self.url_queue:
                        url = self.url_queue.popleft()
                        normalized = self.normalize_url(url)
                        if normalized not in self.visited_urls:
                            self.visited_urls.add(normalized)
                            batch.append(normalized)
                
                if not batch:
                    break
                
                futures = {executor.submit(self.crawl_page, url): url for url in batch}
                
                for future in futures:
                    url = futures[future]
                    result = future.result()
                    
                    if result and result['hash'] not in self.content_hashes:
                        self.content_hashes.add(result['hash'])
                        self.pages[url] = result
                        
                        # Build link graph
                        for link in result['links']:
                            self.link_graph[url].add(link)
                            if link not in self.visited_urls and len(self.pages) < self.max_pages:
                                self.url_queue.append(link)
                        
                        print(f"✓ Crawled: {url[:60]}... ({len(self.pages)}/{self.max_pages})")
        
        print(f"\n✅ Crawling complete! Total unique pages: {len(self.pages)}\n")
    
    def build_index(self):
        print("📚 Building inverted index...")
        for url, page in self.pages.items():
            words = re.findall(r'\w+', page['content'].lower())
            for word in set(words):
                if len(word) > 2:
                    self.inverted_index[word].add(url)
        print(f"✅ Index built with {len(self.inverted_index)} terms\n")
    
    def calculate_tf_idf(self, term, url):
        page = self.pages[url]
        words = re.findall(r'\w+', page['content'].lower())
        
        tf = words.count(term) / len(words) if words else 0
        idf = math.log(len(self.pages) / len(self.inverted_index[term])) if self.inverted_index[term] else 0
        
        return tf * idf
    
    def calculate_pagerank(self, iterations=10, damping=0.85):
        print("📊 Calculating PageRank...")
        num_pages = len(self.pages)
        if num_pages == 0:
            return
        
        # Initialize
        for url in self.pages:
            self.pagerank_scores[url] = 1.0 / num_pages
        
        # Iterate
        for _ in range(iterations):
            new_scores = {}
            for url in self.pages:
                rank_sum = 0
                for other_url in self.pages:
                    if url in self.link_graph[other_url]:
                        outlinks = len(self.link_graph[other_url])
                        if outlinks > 0:
                            rank_sum += self.pagerank_scores[other_url] / outlinks
                
                new_scores[url] = (1 - damping) / num_pages + damping * rank_sum
            
            self.pagerank_scores = new_scores
        
        print(f"✅ PageRank calculated\n")
    
    def search(self, query, top_k=5):
        print(f"🔍 Searching for: '{query}'")
        terms = re.findall(r'\w+', query.lower())
        
        # Find matching documents
        matching_docs = set()
        for term in terms:
            if term in self.inverted_index:
                matching_docs.update(self.inverted_index[term])
        
        if not matching_docs:
            print("❌ No results found\n")
            return []
        
        # Calculate scores
        results = []
        for url in matching_docs:
            tfidf_score = sum(self.calculate_tf_idf(term, url) for term in terms)
            pagerank_score = self.pagerank_scores.get(url, 0)
            combined_score = tfidf_score * 0.7 + pagerank_score * 0.3
            
            results.append({
                'url': url,
                'title': self.pages[url]['title'],
                'score': combined_score,
                'tfidf': tfidf_score,
                'pagerank': pagerank_score
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✅ Found {len(results)} results\n")
        return results[:top_k]


def main():
    # Initialize crawler
    crawler = WebCrawlerSearchEngine(max_pages=30, max_workers=5)
    
    # Seed URLs
    seed_urls = [
        'https://en.wikipedia.org/wiki/Python_(programming_language)',
        'https://en.wikipedia.org/wiki/Web_crawler',
    ]
    
    # Step 1: Crawl websites
    crawler.crawl(seed_urls)
    
    # Step 2: Build search index
    crawler.build_index()
    
    # Step 3: Calculate PageRank
    crawler.calculate_pagerank()
    
    # Step 4: Search API
    print("="*70)
    print("🔎 SEARCH ENGINE READY")
    print("="*70 + "\n")
    
    # Example searches
    queries = ["python programming", "web crawler", "computer science"]
    
    for query in queries:
        results = crawler.search(query, top_k=5)
        
        print(f"📄 Top Results for '{query}':")
        print("-" * 70)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title'][:60]}")
            print(f"   URL: {result['url'][:65]}")
            print(f"   Score: {result['score']:.4f} (TF-IDF: {result['tfidf']:.4f}, PageRank: {result['pagerank']:.6f})")
            print()
        print()


if __name__ == "__main__":
    main()
