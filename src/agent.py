import requests
import feedparser
from datetime import datetime, timedelta
import google.generativeai as genai
import os
import argparse

def fetch_articles():
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
    
    all_articles = []
    for source in rss_sources:
        try:
            response = requests.get(source, timeout=10)
            feed = feedparser.parse(response.content)
            all_articles.extend(feed.entries)
        except Exception as e:
            print(f"Could not fetch or parse {source}: {e}")
            
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

def generate_post(articles, api_key):
    genai.configure(api_key=api_key)
    # You can change model to gemini-2.5-flash or gemini-2.5-pro/other gemini models.
    model = genai.GenerativeModel('gemini-2.5-flash')
    # You can change the style and language output here.
    prompt = """
You are a friendly and enthusiastic tech assistant. Your task is to create a weekly news digest for our software development team. The tone should be informative and easy-to-read. Do not use complex words. The post should be in Russian. 
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

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"

def save_post_as_md(post_content, filename="feed_post.md"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(post_content)
        print(f"Post saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving post to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a tech news digest.')
    parser.add_argument('--api_key', type=str, help='Your Google Gemini API key.')
    args = parser.parse_args()

    if not args.api_key:
        print("Please provide your Google Gemini API key using the --api_key argument.")
    else:
        all_articles = fetch_articles()
        print(f"Fetched {len(all_articles)} articles.")
        
        filtered_articles = filter_articles(all_articles)
        print(f"Filtered down to {len(filtered_articles)} articles.")
        
        if filtered_articles:
            generated_post = generate_post(filtered_articles, args.api_key)
            print("\n--- Generated Post ---")
            print(generated_post)
            save_post_as_md(generated_post)
        else:
            print("No articles to generate a post for.")
