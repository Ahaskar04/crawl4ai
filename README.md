# AI-Powered Web Scraper & Query System

## Overview
This project is an AI-powered web scraper that extracts data from a specified URL, processes the extracted content into structured JSON format, and allows users to query the data using an AI-powered retrieval and response system.

## Features
- **Automated Web Scraping**: Uses `crawl4ai` to extract structured data from a web page.
- **Data Structuring**: Extracted data is saved in JSON format.
- **Embedding Generation**: Converts JSON data into vector embeddings using `sentence-transformers`.
- **FAISS Vector Search**: Enables fast and efficient similarity-based querying of the extracted data.
- **AI-Powered Responses**: Utilizes OpenAI's language model to answer user queries based on extracted data.
- **Interactive Terminal Interface**: Allows users to interact with the AI via text-based queries.

## Requirements
Ensure you have the following installed before running the project:

### Dependencies
- Python 3.8+
- `crawl4ai`
- `pydantic`
- `asyncio`
- `json`
- `langchain_openai`
- `sentence-transformers`
- `faiss`
- `numpy`

### Environment Variables
Ensure the following environment variables are set:
- `DEEPSEEK_API`: API token for DeepSeek AI.
- `OPENAI_API_KEY`: API key for OpenAI.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/web-scraper-ai.git
   cd web-scraper-ai
   ```
2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Step 1: Run the Web Scraper
Run the script to scrape data from the target website:
```sh
python scrape.py
```
This will extract structured data and save it in `scraped_data.json`.

### Step 2: Query the Extracted Data
Run the interactive query system:
```sh
python query.py
```
You can enter queries related to the extracted data. Type `exit` to quit.

## Code Structure
- `scrape.py`: Extracts data from the target website and saves it as JSON.
- `query.py`: Loads the extracted data, generates embeddings, and allows querying.
- `scraped_data.json`: Stores the structured extracted data.

## Troubleshooting
- Ensure API keys are correctly set in your environment.
- If the scraper fails, check if the target website has changed its structure.
- If embeddings fail, verify that `sentence-transformers` is installed and properly configured.

## License
This project is licensed under the MIT License.

## Contributors
- Your Name (@your-github)

## Contact
For any issues, feel free to create an issue in the GitHub repository or reach out via email at `your-email@example.com`.

