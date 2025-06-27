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
    You'll need an API key from Google AI Studio.

2.  **Run the script:**
    Execute the agent from the project's root directory using the following command. Replace `YOUR_API_KEY` with your actual Google Gemini API key.

    ```bash
    python src/agent.py --api_key YOUR_API_KEY
    ```

3.  **Find the output:**
    The script will generate a `feed_post.md` file in the root of the project directory containing the news digest.
