# Fantasy Baseball Points Prediction Project

## Overview

This project aims to predict fantasy baseball points for MLB players using historical statistics and machine learning techniques. It includes data scraping, processing, model training, and prediction components, focusing on three main player categories: Batters, Starting Pitchers (SP), and Relief Pitchers (RP).

## Project Structure

The codebase is organized into several modules, each responsible for a specific aspect of the prediction pipeline:

-   **PointsPredictors**: Contains the main scripts for training models and making predictions.
-   **DataProcessing**: Includes modules for data scraping, cleaning, and feature engineering.
-   **FangraphsScraper**: Provides functionality to scrape data from Fangraphs.
-   **StatCategoryPredictions**: Includes scripts for predicting stat categories.
-   **tests**: Contains unit tests for various modules.

## Key Components

### 1. Data Scraping (`FangraphsScraper`)

-   `fangraphsScraper.py`: Scrapes player statistics from Fangraphs based on position category (Batter, SP, RP) and year range.
    -   Uses `requests` and `lxml` to fetch and parse HTML content.
    -   Caches scraped data to pickle files for efficiency.
    -   Defines `PositionCategory` enum to specify player positions.

### 2. Data Processing (`DataProcessing`)

-   `DataProcessing.py`: Abstract base class for data processing, defining common methods and abstract methods for position-specific subclasses.
    -   Handles data filtering, reshaping, and imputation of missing values using KNN imputation.
    -   Defines data structures for dataset splits (`DatasetSplit`, `WeightedDatasetSplit`, `PredictionDataset`).
-   `BatterDataProcessing.py`, `StarterDataProcessing.py`, `RelieverDataProcessing.py`: Concrete implementations for each position category, inheriting from `DataProcessing`.
    -   Implement abstract methods for data filtering, fantasy points calculation, and counting stats retrieval.
    -   `BatterDataProcessing` includes plate discipline score calculation using PCA.
-   `DataPipelines.py`: Defines data pipelines for training and prediction, including dataset creation, weighting, and formatting.
    -   `TrainingDataPrep` creates rolling train/validation/test splits.
    -   `PredictionDataPrep` prepares data for future predictions.

### 3. Model Training and Prediction (`PointsPredictors`)

-   `PointsPredictor3.py`: Main script for training and evaluating prediction models.
    -   Loads and preprocesses data using `DataProcessing` modules.
    -   Trains XGBoost models for each position category.
    -   Performs hyperparameter tuning using `GridSearchCV`.
    -   Calculates and saves model performance metrics (RMSE, MAE, R², percent difference).
    -   Saves trained models to JSON files.
-   `Predict2025.py`: Script for making predictions for the 2025 season using pre-trained models.
    -   Loads trained XGBoost models.
    -   Prepares prediction data using `PredictionDataPrep`.
    -   Generates and saves predictions to CSV files.
-   `Model`: Class that handles model training, hyperparameter tuning, prediction, and evaluation.
    -   Uses `xgboost` for model training.
    -   Calculates performance metrics such as RMSE, MAE, and R².
    -   Saves trained models for future use.

### 4. Stat Category Predictions (`StatCategoryPredictions`)

-   `CategoryPredictions.py`: Script for predicting specific stat categories (e.g., HR) using a similar pipeline as `PointsPredictor3.py`.
    -   Trains and evaluates models for predicting stat categories instead of total points.

## Usage

### 1. Data Scraping

To scrape data from Fangraphs, use the `FangraphsScraper` class:

```python
from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory

# Example: Scraping data for starting pitchers from 2019 to 2024
scraper = FangraphsScraper(PositionCategory.SP, start_year=2019, end_year=2024)
data = scraper.get_data()
print(data.head())
```

### 2. Model Training

To train a model, run the `PointsPredictor3.py` script:

```bash
python PointsPredictors/PointsPredictor3.py
```

This script will:

-   Load and preprocess data for Batters, Starting Pitchers, and Relief Pitchers.
-   Train XGBoost models for each position category.
-   Save the trained models to the `models` directory.
-   Output performance metrics to the console and save detailed results to the `training_results` directory.

### 3. Making Predictions

To make predictions for the 2025 season, run the `Predict2025.py` script:

```bash
python PointsPredictors/Predict2025.py
```

This script will:

-   Load the trained models from the `models` directory.
-   Prepare prediction data for 2025.
-   Generate predictions for each position category.
-   Save the predictions to CSV files in the `predictions_2025` directory.

### 4. Stat Category Predictions TODO

To predict specific stat categories, run the `CategoryPredictions.py` script:

```bash
python StatCategoryPredictions/CategoryPredictions.py
```

This script will:

-   Load and preprocess data for Batters.
-   Train an XGBoost model for predicting a specific stat category (e.g., HR).
-   Save the trained model and output performance metrics.

## Dependencies

-   pandas
-   numpy
-   xgboost
-   scikit-learn
-   lxml
-   requests

To install the dependencies, use the following command:

```bash
pip install pandas numpy xgboost scikit-learn lxml requests
```

## Tests

The `tests` directory contains unit tests for various modules. To run the tests, use `pytest`:

```bash
pytest tests
```
