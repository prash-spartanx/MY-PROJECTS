# Market Analysis

This project scrapes real-time market data using Selenium, processes it, and applies machine learning for insights and visualization.

## Features
- **Real-time Data Scraping:**
  - Uses Selenium to scrape market data (fields: date, open, high, close, etc.) based on a user-provided topic.
- **Data Processing:**
  - Scraped data is returned as a list, converted to a DataFrame, and saved as a CSV file.
- **Machine Learning & Analysis:**
  - The CSV file is used by `model.py` to train a model, generate insights, and produce graphs and charts.

## Project Structure
- `Market_Analysis/Datasetgenerator.py`: Scrapes and processes data.
- `Market_Analysis/model.py`: Trains the model and generates insights/visualizations.

## Requirements
- Python 3.x
- Selenium
- pandas
- (Other libraries as needed for modeling and plotting)

## Usage
1. Run `Datasetgenerator.py` to scrape and save data.
2. Run `model.py` to analyze the data and view results.

