# Feed Post AI Agent

This project is an AI agent that automates the creation of a tech news digest for a software development team. It fetches articles from various RSS feeds, filters them for relevant topics, and uses AI models from Gemini or OpenAI to generate a friendly, easy-to-read post.

## Features

- Fetches articles from a predefined list of RSS sources.
- Filters articles from the last week based on keywords related to AI, software development, and major tech companies.
- **NEW: Smart diversification algorithm** that ensures variety of sources while maintaining relevance.
- **NEW: AI-powered article selection** that intelligently curates the best articles from filtered results.
- Selects the top 15 most relevant articles with balanced source distribution.
- Uses Google's Gemini or OpenAI's GPT models to generate a formatted post with a title, emojis, and summaries for each article.
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

1.  **Get your API Keys:**
    - **For Gemini:** You'll need an API key from [Google AI Studio](https://aistudio.google.com/apikey).
    - **For OpenAI:** You'll need an API key from the [OpenAI Platform](https://platform.openai.com/api-keys).

2.  **Set up the API keys:**
    You have two options to provide the API keys:
   
    **Option 1:** Create a `.env` file in the root directory with your API keys:
    ```
    GEMINI_KEY=YOUR_GEMINI_API_KEY
    OPENAI_KEY=YOUR_OPENAI_API_KEY
    ```

    **Option 2:** Pass the API keys as command-line arguments when running the script.

3.  **Run the script:**
    Execute the agent from the project's root directory.

    **Using Gemini (default):**
    ```bash
    # Using the API key from .env file
    python agent.py

    # OR, provide the API key directly
    python agent.py --gemini-api-key YOUR_GEMINI_API_KEY
    
    # To use a different Gemini model (default: gemini-2.5-flash)
    python agent.py --model gemini-2.5-pro
    ```

    **Using OpenAI:**
    ```bash
    # Using the API key from .env file
    python agent.py --provider openai

    # OR, provide the API key directly
    python agent.py --provider openai --openai-api-key YOUR_OPENAI_API_KEY

    # To use a different OpenAI model (default: gpt-5-mini)
    python agent.py --provider openai --model gpt-4o-mini
    ```

    **Additional Options:**
    ```bash
    # If you encounter SSL certificate errors, you can disable SSL verification (less secure)
    python agent.py --no-verify-ssl
    
    # To log detailed error messages to a file
    python agent.py --log-file errors.log
    
    # To set a custom timeout for API calls (default is 300 seconds)
    python agent.py --timeout 600
    
    # To change the output language (default is English)
    python agent.py --language Russian
    
    # For Gemini, if you encounter persistent SSL issues, use the fallback method
    python agent.py --no-verify-ssl --use-fallback
    
    # You can combine options as needed
    python agent.py --provider openai --language Russian --model gpt-4o --no-verify-ssl 
    
    # Control source diversification (default: top_n // 3 articles per source)
    python agent.py --max-per-source 1 --no-verify-ssl  # Maximum diversity: only 1 article per source
    python agent.py --max-per-source 3 --no-verify-ssl  # Allow up to 3 articles per source

    # Control the number of articles and time window
    python agent.py --days 3                # Look back for articles from the last 3 days (default: 7)
    python agent.py --top-n 10              # Generate a post with the top 10 articles (default: 15)
    ```
    
    # Use AI to select the best articles from filtered list
    python agent.py --ai-select --ai-select-count 8 --no-verify-ssl  # AI selects 8 best articles (Gemini)
    python agent.py --ai-select --ai-select-count 5 --no-verify-ssl  # AI selects 5 best articles (Gemini)
    python agent.py --provider openai --ai-select --ai-select-count 6 --no-verify-ssl  # AI selects 6 best articles (OpenAI)
        ```

4.  **Find the output:**
    The script will generate a `feed_post.md` file in the root of the project directory containing the news digest.
