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

def ai_select_best_articles(articles, api_key, provider='gemini', model_name='gemini-2.5-flash', 
                           target_count=8, language='English', verify_ssl=True, timeout=300):
    """
    Use AI to select the best articles from the filtered list.
    AI will analyze articles and choose the most interesting/relevant ones.
    """
    if not articles:
        return []
    
    if len(articles) <= target_count:
        print(f"Only {len(articles)} articles available, returning all.")
        return articles
    
    try:
        # Prepare article summaries for AI analysis
        articles_summary = []
        for i, article in enumerate(articles):
            title = getattr(article, 'title', 'No Title')
            summary = getattr(article, 'summary', 'No summary available.')
            link = getattr(article, 'link', '#')
            
            # Extract source domain
            source_domain = ""
            if hasattr(article, 'link') and article.link:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(article.link)
                    source_domain = parsed_url.netloc
                except:
                    source_domain = "unknown"
            
            articles_summary.append({
                'index': i,
                'title': title,
                'summary': summary,
                'source': source_domain,
                'link': link
            })
        
        # Create prompt for AI article selection
        selection_prompt = f"""
You are a tech news curator. Your task is to select the {target_count} most interesting and relevant articles from the following list for a software development team's weekly digest. The articles should be in {language}.

Consider these factors when selecting:
1. Technical relevance and innovation
2. Practical value for developers
3. Current trends and important updates
4. Diversity of topics and sources
5. Overall interest and engagement potential

Please analyze the following {len(articles)} articles and select exactly {target_count} of them by providing their indices in order of preference (most interesting first).

Articles to choose from:
"""
        
        for article in articles_summary:
            selection_prompt += f"""
{article['index']}. {article['title']}
   Source: {article['source']}
   Summary: {article['summary'][:200]}{'...' if len(article['summary']) > 200 else ''}
"""
        
        selection_prompt += f"""

Please respond with ONLY the indices of the {target_count} selected articles, separated by commas, in order of preference (most interesting first).
Example: 0, 3, 7, 2, 5, 1, 4, 6

Selected article indices:"""

        print(f"AI is analyzing {len(articles)} articles to select the best {target_count}...")
        
        if provider == 'gemini':
            return _ai_select_with_gemini(selection_prompt, api_key, model_name, articles, 
                                        target_count, language, verify_ssl, timeout)
        elif provider == 'openai':
            return _ai_select_with_openai(selection_prompt, api_key, model_name, articles, 
                                        target_count, language, timeout)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        print(f"Error in AI article selection: {e}")
        logging.error(f"AI article selection failed: {e}")
        # Fallback: return first target_count articles
        print(f"Falling back to first {target_count} articles.")
        return articles[:target_count]

def _ai_select_with_gemini(prompt, api_key, model_name, articles, target_count, language, verify_ssl, timeout):
    """Use Gemini API for article selection"""
    try:
        if not verify_ssl:
            os.environ['GOOGLE_API_USE_CLIENT_CERTIFICATE'] = 'false'
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['SSL_CERT_FILE'] = ''
            logging.info("SSL verification for Gemini API disabled")
            patch_genai_for_ssl_verification(verify=False)
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Use threading for timeout
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
        
        generation_thread = threading.Thread(target=generate_with_timeout)
        generation_thread.daemon = True
        generation_thread.start()
        
        start_time = time.time()
        while not result["completed"] and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            
        if result["completed"]:
            if result["error"]:
                raise result["error"]
            
            response_text = result["response"].text.strip()
            print(f"AI selection response: {response_text}")
            
            # Parse the response to get article indices
            selected_indices = _parse_ai_selection(response_text, len(articles), target_count)
            selected_articles = [articles[i] for i in selected_indices if i < len(articles)]
            
            print(f"AI selected {len(selected_articles)} articles from indices: {selected_indices}")
            return selected_articles
        else:
            raise Exception(f"AI selection timed out after {timeout} seconds")
            
    except Exception as e:
        print(f"Gemini AI selection failed: {e}")
        logging.error(f"Gemini AI selection error: {e}")
        return articles[:target_count]

def _ai_select_with_openai(prompt, api_key, model_name, articles, target_count, language, timeout):
    """Use OpenAI API for article selection"""
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a tech news curator. Respond with only the requested indices."},
                {"role": "user", "content": prompt}
            ],
            timeout=timeout
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"AI selection response: {response_text}")
        
        # Parse the response to get article indices
        selected_indices = _parse_ai_selection(response_text, len(articles), target_count)
        selected_articles = [articles[i] for i in selected_indices if i < len(articles)]
        
        print(f"AI selected {len(selected_articles)} articles from indices: {selected_indices}")
        return selected_articles
        
    except Exception as e:
        print(f"OpenAI AI selection failed: {e}")
        logging.error(f"OpenAI AI selection error: {e}")
        return articles[:target_count]

def _parse_ai_selection(response_text, max_index, target_count):
    """Parse AI response to extract article indices"""
    try:
        # Extract numbers from the response
        import re
        numbers = re.findall(r'\d+', response_text)
        
        if not numbers:
            raise ValueError("No numbers found in AI response")
        
        # Convert to integers and filter valid indices
        indices = []
        for num in numbers:
            idx = int(num)
            if 0 <= idx < max_index and idx not in indices:
                indices.append(idx)
                if len(indices) >= target_count:
                    break
        
        if len(indices) < target_count:
            print(f"Warning: AI only selected {len(indices)} articles, expected {target_count}")
        
        return indices[:target_count]
        
    except Exception as e:
        print(f"Error parsing AI selection: {e}")
        # Fallback: return first target_count indices
        return list(range(min(target_count, max_index)))

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
    parser.add_argument('--days', type=int, default=7,
                        help='The number of days to look back for recent articles (default: 7).')
    parser.add_argument('--top-n', type=int, default=15,
                        help='The total number of articles to include in the post (default: 15).')
    parser.add_argument('--ai-select', action='store_true',
                        help='Use AI to select the best articles from filtered list instead of using all filtered articles.')
    parser.add_argument('--ai-select-count', type=int, default=8,
                        help='Number of articles for AI to select (default: 8). Only used with --ai-select.')
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
    
    filtered_articles, filter_stats = filter_articles(
        all_articles, 
        days=args.days, 
        top_n=args.top_n, 
        max_per_source=args.max_per_source
    )
    print(f"Filtered down to {len(filtered_articles)} articles.")
    
    # Show detailed filtering statistics
    print("\n--- Filtering Statistics ---")
    print(f"Total articles fetched: {filter_stats['total_articles']}")
    print(f"Articles within {args.days} days: {filter_stats['recent_articles']}")
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

    # AI Article Selection (if enabled)
    final_articles = filtered_articles
    if args.ai_select and len(filtered_articles) > args.ai_select_count:
        print(f"\n--- AI Article Selection ---")
        print(f"AI will select {args.ai_select_count} best articles from {len(filtered_articles)} filtered articles.")
        
        # Get API key for AI selection
        if args.provider == 'gemini':
            ai_api_key = args.gemini_api_key or os.getenv('GEMINI_KEY')
            if not ai_api_key:
                print("Error: No Gemini API key found for AI selection. Please provide it using --gemini-api-key or set GEMINI_KEY in .env file.")
                sys.exit(1)
        elif args.provider == 'openai':
            ai_api_key = args.openai_api_key or os.getenv('OPENAI_KEY')
            if not ai_api_key:
                print("Error: No OpenAI API key found for AI selection. Please provide it using --openai-api-key or set OPENAI_KEY in .env file.")
                sys.exit(1)
        
        try:
            final_articles = ai_select_best_articles(
                filtered_articles, 
                ai_api_key, 
                provider=args.provider,
                model_name=args.model or ('gemini-2.5-flash' if args.provider == 'gemini' else 'gpt-4o-mini'),
                target_count=args.ai_select_count,
                language=args.language,
                verify_ssl=verify_ssl,
                timeout=args.timeout
            )
            
            print(f"AI selected {len(final_articles)} articles for final post generation.")
            print("--- End AI Article Selection ---\n")
            
        except Exception as e:
            print(f"AI selection failed: {e}")
            print("Falling back to using all filtered articles.")
            final_articles = filtered_articles
    elif args.ai_select:
        print(f"Only {len(filtered_articles)} articles available, skipping AI selection.")

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
                final_articles, api_key, model_name=model_name, language=args.language
            )
        else:
            generated_post = generate_post_gemini(
                final_articles, api_key, verify_ssl=verify_ssl, timeout=args.timeout, model_name=model_name, language=args.language
            )
            if "Error:" in generated_post:
                print("\nStandard API method failed. Trying fallback approach...")
                generated_post = generate_post_fallback(
                    final_articles, api_key, model_name=model_name, language=args.language
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
            final_articles, api_key, model_name=model_name, language=args.language, timeout=args.timeout
        )

    print("\n--- Generated Post ---")
    print(generated_post)
    save_post_as_md(generated_post)
