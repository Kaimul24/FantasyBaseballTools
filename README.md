# Fantasy Baseball Points Prediction Project

## Overview

This project predicts fantasy baseball points for MLB players using historical statistics and machine learning techniques. It features a comprehensive pipeline including data retrieval from Fangraphs, data processing, model training, and prediction generation for three main player categories: Batters, Starting Pitchers (SP), and Relief Pitchers (RP).

## Project Structure

The codebase is organized into several modules, each handling a specific aspect of the prediction pipeline:

-   **FangraphsScraper**: Retrieves player statistics from Fangraphs.
-   **DataProcessing**: Handles data cleaning, transformation, and feature engineering.
-   **ModelTraining**: Trains models for points and individual stats, and then predicts future points and stats.
-   **tests**: Contains unit tests for various components.

## Key Components

### 1. Data Scraping (`FangraphsScraper`)

-   `fangraphsScraper.py`: Scrapes player statistics from Fangraphs.
    -   Implements the `PositionCategory` enum to categorize players as Batters, SP, or RP.
    -   Uses `requests` and `lxml` to fetch and parse HTML content from Fangraphs.
    -   Extracts player data from embedded JSON in the page's HTML.
    -   Caches scraped data to pickle files for improved efficiency.
    -   Supports data collection across multiple seasons.

### 2. Data Processing (`DataProcessing`)

-   `DataProcessing.py`: Abstract base class defining the data processing pipeline.
    -   Handles data filtering, reshaping, and missing value imputation.
    -   Defines data structures for dataset management.
-   Position-specific implementations:
    -   `BatterDataProcessing.py`: Processing for batting statistics.
    -   `StarterDataProcessing.py`: Processing for starting pitcher statistics.
    -   `RelieverDataProcessing.py`: Processing for relief pitcher statistics.
-   `DataPipelines.py`: Creates training and prediction pipelines.
    -   `TrainingDataPrep`: Creates train/validation/test splits.
    -   `PredictionDataPrep`: Prepares data for future predictions.

### 3. Model Training and Prediction (`ModelPredictions`)

-   `ModelTraining.py`: Main script for training prediction models.
    -   Uses XGBoost for regression modeling
    -   Performs hyperparameter tuning.
    -   Calculates performance metrics (RMSE, MAE, R²).
    -   Saves trained models for later use.
-   `PredictFutureYear.py`: Generates predictions for upcoming seasons.
    -   Loads pre-trained models.
    -   Processes current player data.
    -   Outputs predictions in CSV format.
-   `Model`: Core class implementing training, evaluation, and prediction functionality.


## Usage

### 1. Data Scraping

To scrape data from Fangraphs:

```python
from src.FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory

# Example: Scrape data for starting pitchers from 2019 to 2024
scraper = FangraphsScraper(PositionCategory.SP, start_year=2019, end_year=2024)
data = scraper.get_data()
print(data.head())
```

### 2. Model Training

To train prediction models:

```bash
python3 -m src.ModelPrediction/ModelTraining
```

This will:
-   Load and process data for all player categories
-   Train XGBoost models with optimized hyperparameters for individual stats and points
-   Save models to the `models` directory
-   Output performance metrics and detailed results

### 3. Making Predictions

To generate predictions for future seasons:

```bash
python3 -m src.ModelPrediction/PredictFutureYear
```

This will:
-   Load trained models
-   Prepare current player data
-   Generate and save predictions as CSV files for individual stats and points

## Dependencies

-   pandas
-   numpy
-   scikit-learn
-   xgboost
-   lxml
-   requests

To install all dependencies:

```bash
pip install -e .
```

## Tests

Run the test suite using pytest:

```bash
pytest tests
```

The test suite includes unit tests for the FangraphsScraper and other components.