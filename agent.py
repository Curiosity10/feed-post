import requests
import feedparser
from datetime import datetime, timedelta
import google.generativeai as genai
from openai import OpenAI
import os
import argparse
from dotenv import load_dotenv
import warnings
import logging
import platform
import sys
import urllib3
import json

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load configuration from config.json
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    RSS_SOURCES = config.get('rss_sources', [])
    KEYWORDS = config.get('keywords', [])
except FileNotFoundError:
    logging.error("config.json not found. Please create it with 'rss_sources' and 'keywords'.")
    RSS_SOURCES = []
    KEYWORDS = []
except json.JSONDecodeError:
    logging.error("Error decoding config.json. Please ensure it is valid JSON.")
    RSS_SOURCES = []
    KEYWORDS = []

# Windows-specific SSL handling
if platform.system() == 'Windows':
    # Try to find and use system certificates if available
    import ssl
    
    # Log Windows and Python version
    logging.info(f"Running on {platform.system()} {platform.version()}")
    logging.info(f"Python version: {sys.version}")
    
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        logging.info(f"Using certifi certificates from: {certifi.where()}")
    except ImportError:
        logging.warning("certifi package not found. Consider installing it for better SSL support.")
        pass
        
# Custom function to patch Google Generative AI library for SSL verification issues
def patch_genai_for_ssl_verification(verify=True):
    try:
        # This is a slightly invasive approach but necessary to fix SSL issues with the Gemini API
        import httpcore
        import httpx
        
        # If we want to disable verification, create a custom transport
        if not verify:
            # Create a custom HTTP transport for httpx that ignores SSL verification
            transport = httpcore.AsyncHTTPTransport(verify=False)
            
            # Patch the httpx client used by google.generativeai
            # Find where genai stores its client
            for attr_name in dir(genai):
                attr = getattr(genai, attr_name)
                if hasattr(attr, "_client"):
                    # Found a module with a client, let's patch it
                    logging.info(f"Patching genai.{attr_name}._client for SSL verification")
                    # Create a new client with our transport
                    old_client = attr._client
                    if hasattr(old_client, "_base_url"):
                        new_client = httpx.AsyncClient(
                            base_url=old_client._base_url,
                            headers=old_client._headers,
                            transport=transport
                        )
                        attr._client = new_client
            
            logging.info("Successfully patched google.generativeai for disabled SSL verification")
    except ImportError as e:
        logging.warning(f"Couldn't patch genai for SSL: {e}")
    except Exception as e:
        logging.warning(f"Error while patching genai for SSL: {e}")

def fetch_articles(verify_ssl=True):
    # RSS feeds are loaded from config.json.
    rss_sources = RSS_SOURCES
    
    if not verify_ssl:
        warnings.warn("SSL verification is disabled. This is not secure and should only be used for testing.")
    
    all_articles = []
    for source in rss_sources:
        try:
            response = requests.get(source, timeout=10, verify=verify_ssl)
            feed = feedparser.parse(response.content)
            all_articles.extend(feed.entries)
            print(f"Successfully fetched: {source}")
        except Exception as e:
            # Print simple message to console
            print(f"Could not fetch: {source}")
            if verify_ssl:
                print("  If this is an SSL error, try using --no-verify-ssl option")
            
            # Log detailed error to log
            logging.error(f"Error fetching {source}: {str(e)}")
            
    return all_articles

def filter_articles(articles, days=7, top_n=15, max_per_source=None):
    """
    Filter and diversify articles using a two-stage approach:
    1. Score articles by keyword relevance
    2. Apply diversification algorithm to ensure variety of sources
    
    Diversification rules:
    - Maximum top_n // 3 articles per source (but at least 1)
    - Fallback: if not enough articles, add remaining ones
    """
    # Keywords are loaded from config.json.
    keywords = KEYWORDS
    # Filter articles by last 7 days.
    one_week_ago = datetime.now() - timedelta(days=days)
    
    # Filter by date
    recent_articles = []
    for article in articles:
        if hasattr(article, 'published_parsed') and article.published_parsed:
            published_date = datetime(*article.published_parsed[:6])
        else:
            published_date = datetime.now()
        if published_date >= one_week_ago:
            recent_articles.append(article)
            
    # Score by keywords
    scored_articles = []
    for article in recent_articles:
        score = 0
        found_keywords = []
        title = article.title.lower() if hasattr(article, 'title') else ''
        summary = article.summary.lower() if hasattr(article, 'summary') else ''
        
        for keyword in keywords:
            if keyword in title or keyword in summary:
                score += 1
                found_keywords.append(keyword)
        
        if score > 0:
            # Extract source domain for diversification
            source_domain = ""
            if hasattr(article, 'link') and article.link:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(article.link)
                    source_domain = parsed_url.netloc
                except:
                    source_domain = "unknown"
            else:
                source_domain = "unknown"
            
            scored_articles.append({
                'article': article, 
                'score': score, 
                'source_domain': source_domain,
                'found_keywords': found_keywords
            })
            
    # Sort by relevance first
    sorted_articles = sorted(scored_articles, key=lambda x: x['score'], reverse=True)
    
    # Apply diversification algorithm
    if max_per_source is None:
        max_per_source = max(1, top_n // 3)  # Default: maximum articles per source (but at least 1)
    else:
        max_per_source = max(1, max_per_source)  # Ensure at least 1 article per source
    diversified_articles = []
    source_counts = {}
    
    # First pass: add articles respecting source limits
    for item in sorted_articles:
        source = item['source_domain']
        if source not in source_counts:
            source_counts[source] = 0
        
        if source_counts[source] < max_per_source:
            diversified_articles.append(item['article'])
            source_counts[source] += 1
            
            if len(diversified_articles) >= top_n:
                break
    
    # Fallback: if we don't have enough articles, add remaining ones
    if len(diversified_articles) < top_n:
        remaining_articles = []
        for item in sorted_articles:
            if item['article'] not in diversified_articles:
                remaining_articles.append(item['article'])
        
        # Add remaining articles until we reach top_n
        for article in remaining_articles:
            if len(diversified_articles) >= top_n:
                break
            diversified_articles.append(article)
    
    # Log diversification results
    logging.info(f"Diversification applied: max {max_per_source} articles per source")
    logging.info(f"Source distribution: {source_counts}")
    logging.info(f"Final article count: {len(diversified_articles)}")
    
    # Return both articles and statistics for better reporting
    stats = {
        'total_articles': len(articles),
        'recent_articles': len(recent_articles),
        'scored_articles': len(scored_articles),
        'final_articles': len(diversified_articles),
        'source_distribution': source_counts,
        'max_per_source': max_per_source,
        'scored_articles_data': scored_articles  # Include the full scored articles data
    }
    
    return diversified_articles, stats

def generate_post_gemini(articles, api_key, verify_ssl=True, timeout=300, model_name='gemini-2.5-flash', language='English'):
    try:
        # Set environment variable to disable SSL verification if needed
        if not verify_ssl:
            os.environ['GOOGLE_API_USE_CLIENT_CERTIFICATE'] = 'false'
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['SSL_CERT_FILE'] = ''
            logging.info("SSL verification for Gemini API disabled")
            
            # Apply our custom patch for SSL verification
            patch_genai_for_ssl_verification(verify=False)
        
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # You can change model to gemini-2.5-flash or gemini-2.5-pro/other gemini models.
        model = genai.GenerativeModel(model_name)
        
        # You can change the style and language output here.
        prompt = f"""
You are a friendly and enthusiastic tech assistant. Your task is to create a weekly news digest for our software development team. The tone should be informative and easy-to-read. Do not use complex words. The post should be in {language}. 
Do not use exclamation marks in the text.

Create a post with a fun, engaging title. For each article below, present it as a numbered list item in the following format:
1. emoji [Article Title](link) 
   3-sentence summary.

Here are the articles:
"""

        for i, article in enumerate(articles):
            title = getattr(article, 'title', 'No Title')
            link = getattr(article, 'link', '#')
            summary = getattr(article, 'summary', 'No summary available.')
            prompt += f"\n{i+1}. [{title}]({link})\n   {summary}\n"

        print(f"Generating content with {model_name} API...")
        
        # Use a cross-platform timeout approach with threading
        import threading
        import time
        
        result = {"response": None, "error": None, "completed": False}
        
        def generate_with_timeout():
            try:
                result["response"] = model.generate_content(prompt)
                result["completed"] = True
            except Exception as e:
                result["error"] = e
                result["completed"] = True
        
        # Start the generation in a separate thread
        generation_thread = threading.Thread(target=generate_with_timeout)
        generation_thread.daemon = True
        generation_thread.start()
        
        # Wait for the generation to complete or timeout
        start_time = time.time()
        while not result["completed"] and (time.time() - start_time) < timeout:
            time.sleep(0.5)  # Check every half second
            
        if result["completed"]:
            if result["error"]:
                raise result["error"]
            print("Content generation successful!")
            return result["response"].text
        else:
            return f"Error: API call timed out after {timeout} seconds. Try again or use a different model."
        
    except Exception as e:
        error_msg = f"Error generating content: {e}"
        logging.error(error_msg)
        return error_msg

def generate_post_fallback(articles, api_key, model_name='gemini-2.5-flash', language='English'):
    """Fallback function that uses direct REST API calls with requests when the standard API doesn't work."""
    try:
        print(f"Trying fallback approach with direct API requests to {model_name}...")
        
        # Construct the prompt
        prompt = f"""
You are a friendly and enthusiastic tech assistant. Your task is to create a weekly news digest for our software development team. The tone should be informative and easy-to-read. Do not use complex words. The post should be in {language}. 
Do not use exclamation marks in the text.

Create a post with a fun, engaging title. For each article below, present it as a numbered list item in the following format:
1. emoji [Article Title](link) 
   3-sentence summary.

Here are the articles:
"""

        for i, article in enumerate(articles):
            title = getattr(article, 'title', 'No Title')
            link = getattr(article, 'link', '#')
            summary = getattr(article, 'summary', 'No summary available.')
            prompt += f"\n{i+1}. [{title}]({link})\n   {summary}\n"

        # Direct API call using requests
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        payload = {
            "contents": [{"parts":[{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        # Make the request with SSL verification disabled
        response = requests.post(
            api_url, 
            json=payload, 
            headers=headers, 
            verify=False  # Disable SSL verification
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract the text from the response
            text = result['candidates'][0]['content']['parts'][0]['text']
            print("Fallback content generation successful!")
            return text
        else:
            error_msg = f"Fallback API call failed with status code {response.status_code}: {response.text}"
            logging.error(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Error in fallback content generation: {e}"
        logging.error(error_msg)
        return error_msg

def generate_post_openai(articles, api_key, model_name='gpt-5-mini', language='English', timeout=300):
    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
You are a friendly and enthusiastic tech assistant. Your task is to create a weekly news digest for our software development team. The tone should be informative and easy-to-read. Do not use complex words. The post should be in {language}. 
Do not use exclamation marks in the text.

Create a post with a fun, engaging title. For each article below, present it as a numbered list item in the following format:
1. emoji [Article Title](link) 
   3-sentence summary.

Here are the articles:
"""
        for i, article in enumerate(articles):
            title = getattr(article, 'title', 'No Title')
            link = getattr(article, 'link', '#')
            summary = getattr(article, 'summary', 'No summary available.')
            prompt += f"\n{i+1}. [{title}]({link})\n   {summary}\n"

        print(f"Generating content with {model_name} API...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            timeout=timeout
        )
        
        print("Content generation successful!")
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = f"Error generating content with OpenAI: {e}"
        logging.error(error_msg)
        return error_msg

def save_post_as_md(post_content, filename="feed_post.md"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(post_content)
        print(f"Post saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving post to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a tech news digest.')
    parser.add_argument('--provider', type=str, default='gemini', choices=['gemini', 'openai'], help='The provider to use for content generation (default: gemini).')
    parser.add_argument('--gemini-api-key', type=str, help='Your Google Gemini API key (overrides .env file).')
    parser.add_argument('--openai-api-key', type=str, help='Your OpenAI API key (overrides .env file).')
    parser.add_argument('--no-verify-ssl', action='store_true', help='Disable SSL certificate verification (not secure, use only when needed).')
    parser.add_argument('--log-file', type=str, help='Path to a log file to record detailed errors and info.')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for API calls (default: 300)')
    parser.add_argument('--model', type=str, help='The model to use for the selected provider.')
    parser.add_argument('--use-fallback', action='store_true', help='Use fallback direct API approach for Gemini instead of the standard library.')
    parser.add_argument('--language', type=str, default='English',
                        help='Language for the generated content (default: English). Examples: Russian, German, French, Spanish, etc.')
    parser.add_argument('--max-per-source', type=int, default=None,
                        help='Maximum articles per source (default: top_n // 3). Set to 1 for maximum diversity.')
    args = parser.parse_args()
    
    # Set up file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info("Starting feed-post agent with file logging")
    
    verify_ssl = not args.no_verify_ssl
    
    all_articles = fetch_articles(verify_ssl=verify_ssl)
    print(f"Fetched {len(all_articles)} articles.")
    
    # Show source distribution before filtering
    print("\n--- Source Distribution Before Filtering ---")
    source_counts_before = {}
    for article in all_articles:
        if hasattr(article, 'link') and article.link:
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(article.link)
                source = parsed_url.netloc
                source_counts_before[source] = source_counts_before.get(source, 0) + 1
            except:
                pass
    
    for source, count in sorted(source_counts_before.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} articles")
    print(f"Total sources: {len(source_counts_before)}")
    print("--- End Source Distribution ---\n")
    
    filtered_articles, filter_stats = filter_articles(all_articles, max_per_source=args.max_per_source)
    print(f"Filtered down to {len(filtered_articles)} articles.")
    
    # Show detailed filtering statistics
    print("\n--- Filtering Statistics ---")
    print(f"Total articles fetched: {filter_stats['total_articles']}")
    print(f"Articles within {7} days: {filter_stats['recent_articles']}")
    print(f"Articles with keywords: {filter_stats['scored_articles']}")
    print(f"Final articles after diversification: {filter_stats['final_articles']}")
    print(f"Maximum articles per source: {filter_stats['max_per_source']}")
    if args.max_per_source:
        print(f"Custom max-per-source setting: {args.max_per_source}")
    else:
        print("Using default max-per-source setting (top_n // 3)")
    print("--- End Filtering Statistics ---\n")
    
    # Show diversification results
    if filtered_articles:
        print("\n--- Diversification Results ---")
        source_counts = filter_stats['source_distribution']
        
        print("Articles per source after diversification:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} articles")
        print(f"Total sources: {len(source_counts)}")
        print("--- End Diversification Results ---\n")
        
        # Show keyword analysis for top articles
        print("\n--- Keyword Analysis for Top Articles ---")
        scored_articles_data = filter_stats['scored_articles_data']
        for i, item in enumerate(scored_articles_data[:10]):  # Show top 10 scored articles
            if item['article'] in filtered_articles:
                title = getattr(item['article'], 'title', 'No Title')[:60] + "..." if len(getattr(item['article'], 'title', 'No Title')) > 60 else getattr(item['article'], 'title', 'No Title')
                source = item['source_domain']
                score = item['score']
                keywords = item['found_keywords']
                print(f"  {i+1}. Score: {score} | Source: {source}")
                print(f"     Title: {title}")
                print(f"     Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
                print()
        print("--- End Keyword Analysis ---\n")
    
    if not filtered_articles:
        print("No articles to generate a post for.")
        sys.exit(0)

    generated_post = ""
    if args.provider == 'gemini':
        api_key = args.gemini_api_key or os.getenv('GEMINI_KEY')
        model_name = args.model or 'gemini-2.5-flash'
        if not api_key:
            print("Error: No Gemini API key found. Please provide it using --gemini-api-key or set GEMINI_KEY in .env file.")
            sys.exit(1)
            
        print(f"Starting Feed Post generator with Gemini...")
        print(f"Using model: {model_name}")
        
        if args.use_fallback:
            generated_post = generate_post_fallback(
                filtered_articles, api_key, model_name=model_name, language=args.language
            )
        else:
            generated_post = generate_post_gemini(
                filtered_articles, api_key, verify_ssl=verify_ssl, timeout=args.timeout, model_name=model_name, language=args.language
            )
            if "Error:" in generated_post:
                print("\nStandard API method failed. Trying fallback approach...")
                generated_post = generate_post_fallback(
                    filtered_articles, api_key, model_name=model_name, language=args.language
                )

    elif args.provider == 'openai':
        api_key = args.openai_api_key or os.getenv('OPENAI_KEY')
        model_name = args.model or 'gpt-5-mini'
        if not api_key:
            print("Error: No OpenAI API key found. Please provide it using --openai-api-key or set OPENAI_KEY in .env file.")
            sys.exit(1)
            
        print(f"Starting Feed Post generator with OpenAI...")
        print(f"Using model: {model_name}")
        
        generated_post = generate_post_openai(
            filtered_articles, api_key, model_name=model_name, language=args.language, timeout=args.timeout
        )

    print("\n--- Generated Post ---")
    print(generated_post)
    save_post_as_md(generated_post)
