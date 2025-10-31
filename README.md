# Housing Price Prediction with MLflow

This lab demonstrates predicting housing prices using XGBoost and MLflow for experiment tracking and model deployment.

## Dataset
California Housing dataset (built-in to scikit-learn)

## Prerequisites
- Python 3.8+
- Virtual environment (recommended)

## Setup

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

4. **Open `starter.ipynb` and run all cells sequentially**

5. **Launch MLflow UI (in a separate terminal):**
```bash
mlflow ui
```
Then open: `http://localhost:5000`

## Lab Steps

1. **Import and Load Data** - Load California Housing dataset
2. **Explore Data** - Basic statistics and data overview
3. **Feature Engineering** - Create ratio-based features
4. **Data Visualization** - Distribution plots and correlation heatmap
5. **Check Missing Data** - Verify data quality
6. **Data Splitting** - Split into train/validation/test sets (60/20/20)
7. **Build Baseline Model** - Train XGBoost with default parameters, track with MLflow
8. **Feature Importance** - Analyze which features matter most
9. **Hyperparameter Tuning** - Grid search over 27 parameter combinations
10. **Train Final Model** - Train best model on full training data
11. **Model Registration** - Register model in MLflow Model Registry
12. **Transition to Production** - Move model to Production stage
13. **Batch Inference** - Load production model and make predictions
14. **Residual Analysis** - Visualize prediction errors
15. **Model Comparison** - Compare baseline vs tuned model
16. **Serve Model** - Deploy model for real-time inference
17. **Real-Time Inference** - Test API endpoint with sample data

## Model Serving

To serve the model for real-time predictions:
```bash
mlflow models serve --env-manager=local -m models:/housing_price_predictor/production -h 0.0.0.0 -p 5001
```

## What to Check in MLflow UI

- **Experiments tab**: View all 28+ runs (baseline + 27 tuning runs + final)
- **Compare runs**: Select multiple runs to compare metrics
- **Models tab**: Check registered model in Production stage
- **Metrics**: RMSE, MAE, R² scores for each run
- **Parameters**: Hyperparameters used in each experiment

(Note: This is a modification of the original lab from "ML with Ramim" where the work was on iris dataset - here the work is done on the wine dataset) 
