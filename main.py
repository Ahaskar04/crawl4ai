import asyncio
import os
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# URL to scrape
URL_TO_SCRAPE = "https://web.lmarena.ai/leaderboard"

# Instruction to the LLM for data extraction
INSTRUCTION_TO_LLM = "Extract all rows from the main table as objects with 'rank', 'model', 'arena score', '95% CI', and other relevant fields."

# Define the schema for extracted data
class Product(BaseModel):
    rank: str
    model: str
    arena_score: str
    confidence_interval: str

# Main asynchronous function
async def main():
    # Configure the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        provider="deepseek/deepseek-chat",
        api_token=os.getenv("DEEPSEEK_API"),  # Ensure DEEPSEEK_API is set in your environment
        schema=Product.model_json_schema(),  # Use the Product schema for structuring the data
        extraction_type="schema",
        instruction=INSTRUCTION_TO_LLM,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 800},
    )

    # Configure the crawler
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        exclude_external_links=True,
    )

    # Configure the browser
    browser_cfg = BrowserConfig(headless=True, verbose=True)

    # Use the AsyncWebCrawler for crawling
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=URL_TO_SCRAPE, config=crawl_config)

    # Check if the crawling and extraction succeeded
    if result.success:
        data = json.loads(result.extracted_content)  # Parse the extracted content
        print("Extracted items:", data)  # Print extracted data
        
        # Save the extracted data to a JSON file
        output_file = "scraped_data.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {output_file}")

        llm_strategy.show_usage()  # Show LLM usage stats
    else:
        print("Error:", result.error_message)  # Print error message if extraction fails

# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())
