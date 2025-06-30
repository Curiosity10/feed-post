# Feed Post AI Agent

This project is an AI agent that automates the creation of a tech news digest for a software development team. It fetches articles from various RSS feeds, filters them for relevant topics, and uses Google's Gemini models to generate a friendly, easy-to-read post.

## Features

- Fetches articles from a predefined list of RSS sources.
- Filters articles from the last week based on keywords related to AI, software development, and major tech companies.
- Selects the top 15 most relevant articles.
- Uses Google's Gemini models to generate a formatted post with a title, emojis, and summaries for each article.
- Saves the final post as a Markdown file (`feed_post.md`).

## Requirements

- Python 3.x
- The packages listed in `requirements.txt`

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd feed-post
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Get a Google Gemini API Key:**
    You'll need an API key from [Google AI Studio](https://aistudio.google.com/apikey).

2.  **Set up the API key:**
    You have two options to provide the API key:
   
    **Option 1:** Create a `.env` file in the root directory with your API key:
    ```
    GEMINI_KEY=YOUR_API_KEY
    ```

    **Option 2:** Pass the API key as a command-line argument when running the script.

3.  **Run the script:**
    Execute the agent from the project's root directory using one of the following commands:

    ```bash
    # Using the API key from .env file
    python src/agent.py

    # OR, override the .env file by providing the API key directly
    python src/agent.py --api_key YOUR_API_KEY
    
    # If you encounter SSL certificate errors, you can disable SSL verification (less secure)
    python src/agent.py --no-verify-ssl
    
    # To log detailed error messages to a file
    python src/agent.py --log-file errors.log
    
    # To set a custom timeout for API calls (default is 300 seconds)
    python src/agent.py --timeout 600
    
    # To use a different Gemini model (default is gemini-2.5-flash)
    python src/agent.py --model gemini-2.5-pro
    
    # To change the output language (default is Russian)
    python src/agent.py --language English
    
    # If you encounter persistent SSL issues, use the fallback method
    python src/agent.py --no-verify-ssl --use-fallback
    
    # You can combine options as needed
    python src/agent.py --no-verify-ssl --log-file errors.log --timeout 600 --model gemini-1.5-pro --language English
    ```

4.  **Find the output:**
    The script will generate a `feed_post.md` file in the root of the project directory containing the news digest.
