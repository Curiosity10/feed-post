import requests
import feedparser
from datetime import datetime, timedelta
import google.generativeai as genai
import os
import argparse
from dotenv import load_dotenv
import warnings
import logging
import platform
import sys
import urllib3

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    # You can add more RSS feeds here.
    rss_sources = [
        "https://thezvi.substack.com/feed",
        "https://simonwillison.net/atom/everything/",
        "https://www.artificial-intelligence.blog/ai-news?format=rss",
        "https://research.facebook.com/feed/",
        "https://www.wired.com/feed/tag/ai/latest/rss",
        "https://magazine.sebastianraschka.com/feed",
        "https://www.artificialintelligence-news.com/feed/",
        "https://www.404media.co/rss/",
        "https://hackernoon.com/tagged/ai/feed",
        "https://blogs.windows.com/feed/",
        "https://visualstudiomagazine.com/rss-feeds/news.aspx"
    ]
    
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

def filter_articles(articles, days=14, top_n=15):
    # You can add the most important for you keywords here.
    keywords = [
        # Core JavaScript & Web
        'javascript', 'typescript', 'node.js', 'npm', 'yarn', 'webpack', 'vite',
        'react', 'next.js', 'vue', 'angular', 'svelte', 'tailwind', 
        'web components', 'pwa', 'microfrontend', 'css', 'html',
        'accessibility', 'a11y', 'seo', 'performance', 'optimization',
        'web vitals', 'lighthouse', 
        
        # AI/ML Focus
        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
        'tensorflow.js', 'langchain', 'llama',
        'openai', 'anthropic', 'gemini', 'chatgpt', 'copilot',
        'vector database', 'embeddings', 'rag', 'fine-tuning', 'prompt engineering',
        'ai tool', 'ai agent', 'mcp', 'llm', 'large language model',
        'ollama', 'lm studio',
        
        # Development Tools
        'cursor', 'vs code', 'ide', 'windsurf',
        'visual studio', 

        # Testing
        'jest', 'vitest', 'cypress', 'testing', 'playwright',
        'unit test', 'integration test', 'e2e test', 'test coverage',
        'test automation', 'test framework', 'test runner',
        
        # Backend & Data
        'express', 'mongodb', 'postgresql', 'mysql', 'redis',
        'graphql', 'rest api', 'microservices', 'serverless',
        
        # Infrastructure
        'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform',
        
        # Emerging Tech
        'web3', 'blockchain', 'deno', 'bun',

        # Companies/Products
        'microsoft', 'amazon', 'google', 'meta', 'apple',
        'netflix', 'spotify', 'tiktok', 'instagram', 'facebook',
        'twitter', 'youtube', 'linkedin', 'reddit', 'pinterest',
        'snapchat', 'twitch', 'discord', 'telegram', 'whatsapp',
        'slack', 'zoom', 'github', 'gitlab', 'bitbucket',
        'vercel', 'cloudflare', 'figma',
    ]
    # Filter articles by last 14 days.
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
        title = article.title.lower() if hasattr(article, 'title') else ''
        summary = article.summary.lower() if hasattr(article, 'summary') else ''
        
        for keyword in keywords:
            if keyword in title or keyword in summary:
                score += 1
        
        if score > 0:
            scored_articles.append({'article': article, 'score': score})
            
    # Sort and get top N
    sorted_articles = sorted(scored_articles, key=lambda x: x['score'], reverse=True)
    
    return [item['article'] for item in sorted_articles[:top_n]]

def generate_post(articles, api_key, verify_ssl=True, timeout=300, model_name='gemini-2.5-flash', language='Russian'):
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

def generate_post_fallback(articles, api_key, model_name='gemini-2.5-flash', language='Russian'):
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

def save_post_as_md(post_content, filename="feed_post.md"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(post_content)
        print(f"Post saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving post to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a tech news digest.')
    parser.add_argument('--api_key', type=str, help='Your Google Gemini API key (overrides .env file).')
    parser.add_argument('--no-verify-ssl', action='store_true', help='Disable SSL certificate verification (not secure, use only when needed).')
    parser.add_argument('--log-file', type=str, help='Path to a log file to record detailed errors and info.')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for API calls (default: 300)')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', 
                        help='Gemini model to use (default: gemini-2.5-flash). Options include gemini-2.5-pro, gemini-1.5-pro.')
    parser.add_argument('--use-fallback', action='store_true', help='Use fallback direct API approach instead of the standard library.')
    parser.add_argument('--language', type=str, default='Russian', 
                        help='Language for the generated content (default: Russian). Examples: English, German, French, Spanish, etc.')
    args = parser.parse_args()
    
    # Set up file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info("Starting feed-post agent with file logging")
    
    # Use API key from command line if provided, otherwise use from .env
    api_key = args.api_key or os.getenv('GEMINI_KEY')
    verify_ssl = not args.no_verify_ssl

    if not api_key:
        print("Error: No API key found. Please either provide it using --api_key argument or set GEMINI_KEY in .env file.")
    else:
        print(f"Starting Feed Post generator...")
        print(f"SSL verification: {'Disabled' if not verify_ssl else 'Enabled'}")
        print(f"Using model: {args.model}")
        print(f"Output language: {args.language}")
        print(f"API timeout: {args.timeout} seconds")
        
        all_articles = fetch_articles(verify_ssl=verify_ssl)
        print(f"Fetched {len(all_articles)} articles.")
        
        filtered_articles = filter_articles(all_articles)
        print(f"Filtered down to {len(filtered_articles)} articles.")
        
        if filtered_articles:
            if args.use_fallback:
                # Use the fallback method directly if requested
                generated_post = generate_post_fallback(
                    filtered_articles,
                    api_key,
                    model_name=args.model,
                    language=args.language
                )
            else:
                # Try the standard method first
                generated_post = generate_post(
                    filtered_articles, 
                    api_key, 
                    verify_ssl=verify_ssl, 
                    timeout=args.timeout,
                    model_name=args.model,
                    language=args.language
                )
                
                # If it contains an error, try the fallback method
                if generated_post.startswith("Error:"):
                    print("\nStandard API method failed. Trying fallback approach...")
                    generated_post = generate_post_fallback(
                        filtered_articles,
                        api_key,
                        model_name=args.model,
                        language=args.language
                    )
            
            print("\n--- Generated Post ---")
            print(generated_post)
            save_post_as_md(generated_post)
        else:
            print("No articles to generate a post for.")
