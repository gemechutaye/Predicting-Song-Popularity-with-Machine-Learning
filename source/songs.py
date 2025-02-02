"""
Author: CJ Phillips, Dylan Krim, and Gemechu Taye 
Professor: Dr. Korpusik
Description: Machine Learning utilities for predicting song popularity using various regression models.
"""

# =============================================================================
# Import Statements
# =============================================================================
# Importing necessary standard libraries
import math
import csv
from util import *

import xgboost as xgb

from sklearn.model_selection import GridSearchCV  # added

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Function Definitions
# =============================================================================

def create_feature_plots_with_trend(X, y, Xnames, save_plots=False):
    """
    Create scatter plots with trend lines for each feature vs popularity.

    Parameters:
    - X: numpy.ndarray
        Feature matrix.
    - y: numpy.ndarray
        Target variable (popularity).
    - Xnames: list
        List of feature names.
    - save_plots: bool
        Whether to save the plots as PNG files.

    Returns:
    - None
    """
    plt.style.use('seaborn')
    
    for i in range(len(Xnames)):
        plt.figure(figsize=(10, 6))
        selected_x = X[:,i]
        
        # Create scatter plot
        plt.scatter(selected_x, y, alpha=0.3, c='#2ecc71', s=10, label='Songs')
        
        # Calculate and plot trend line
        z = np.polyfit(selected_x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(selected_x), max(selected_x), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Trend Line')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(selected_x, y)[0,1]
        
        # Add labels and title
        plt.xlabel(Xnames[i], fontsize=12)
        plt.ylabel('Popularity', fontsize=12)
        plt.title(f'Relationship between {Xnames[i]} and Song Popularity\nCorrelation: {correlation:.3f}', 
                 fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'plots/{Xnames[i]}_vs_popularity.png', dpi=300, bbox_inches='tight')
        plt.show(block=True)


def train_baseline_model(X, y, Xnames):
    """
    Train and evaluate a baseline Decision Tree classifier model.

    Parameters:
    - X: numpy.ndarray
        Feature matrix.
    - y: numpy.ndarray
        Target variable.
    - Xnames: list
        List of feature names.

    Returns:
    - dt_model: trained DecisionTreeClassifier
        The trained Decision Tree model.
    - scaler: StandardScaler
        The fitted scaler object.
    - feature_importance: list of tuples
        Feature names paired with their importance scores.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train decision tree model
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = list(zip(Xnames, dt_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    print("\nTop 5 Most Important Features:")
    for feature, importance in feature_importance[:5]:
        print(f"{feature}: {importance:.3f}")
    
    return dt_model, scaler, feature_importance


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main function to execute the machine learning pipeline:
    - Load data
    - Train and evaluate multiple regression models
    - Display performance metrics and plots
    """
    import time
    total_start_time = time.time()
    
    # Load Spotify dataset
    songs = load_data('spotify_songs_part1.csv', header=1, predict_col=0)
    X = songs.X
    Xnames = songs.Xnames
    y = songs.y
    yname = songs.yname
    n, d = X.shape  # n = number of examples, d = number of features

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)
    lr_train_mae = metrics.mean_absolute_error(y_train, lr_train_pred)
    lr_test_mae = metrics.mean_absolute_error(y_test, lr_test_pred)
    lr_train_r2 = r2_score(y_train, lr_train_pred)
    lr_test_r2 = r2_score(y_test, lr_test_pred)
    
    # Train and evaluate Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=10, max_features='sqrt', min_samples_split=20, min_samples_leaf=10)
    dt_model.fit(X_train, y_train)
    dt_train_pred = dt_model.predict(X_train)
    dt_test_pred = dt_model.predict(X_test)
    dt_train_mae = metrics.mean_absolute_error(y_train, dt_train_pred)
    dt_test_mae = metrics.mean_absolute_error(y_test, dt_test_pred)
    dt_train_r2 = r2_score(y_train, dt_train_pred)
    dt_test_r2 = r2_score(y_test, dt_test_pred)

    
    ###################################################
    # Code for testing Random Forest parameters (WARNING estimated Runtime 80 minutes)
    ''' 
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],  
        'max_depth': [5, 10, 20, None],  
        'min_samples_split': [2, 5, 10, 20],  
        'min_samples_leaf': [1, 2, 4],  
        'max_features': ['auto', 'sqrt', 'log2', None],  
    }

    rf_model = RandomForestRegressor(random_state=42)

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                            cv=5, scoring='neg_mean_absolute_error', 
                            verbose=2, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")

    best_rf_model = grid_search.best_estimator_

    rf_train_pred = best_rf_model.predict(X_train)
    rf_test_pred = best_rf_model.predict(X_test)

    # Calculate MAE for the best model
    rf_train_mae = metrics.mean_absolute_error(y_train, rf_train_pred)
    rf_test_mae = metrics.mean_absolute_error(y_test, rf_test_pred)

    print("\nRandom Forest Model Performance (Best Hyperparameters):")
    print(f"Training MAE: {rf_train_mae:.2f}")
    print(f"Test MAE: {rf_test_mae:.2f}")

    ###########################################
    '''
    # Train and evaluate Random Forest
    rf_model = RandomForestRegressor(random_state=42, max_depth=10, max_features='sqrt', n_estimators=500, min_samples_split=10, min_samples_leaf=5)
    rf_model.fit(X_train, y_train)
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    rf_train_mae = metrics.mean_absolute_error(y_train, rf_train_pred)
    rf_test_mae = metrics.mean_absolute_error(y_test, rf_test_pred)

    # Calculate R²
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)

    # Define XGBoost model and parameters
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20)

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Predict on training and test sets
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)

    # Evaluate the model
    xgb_train_mae = metrics.mean_absolute_error(y_train, xgb_train_pred)
    xgb_test_mae = metrics.mean_absolute_error(y_test, xgb_test_pred)
    xgb_train_r2 = r2_score(y_train, xgb_train_pred)
    xgb_test_r2 = r2_score(y_test, xgb_test_pred)

    # Baseline: Predict the mean popularity for all samples
    baseline_pred = np.mean(y_train)  # Calculate mean from training data
    y_baseline_test = np.full_like(y_test, baseline_pred)  # Create an array of mean predictions for the test set

    # Calculate Baseline MAE
    baseline_mae = metrics.mean_absolute_error(y_test, y_baseline_test)

    # Print baseline performance
    print(f"\nBaseline Model Performance:")
    print(f"Baseline Mean Absolute Error (MAE): {baseline_mae:.2f}")

    print(f"\nModel Fit:")
    print(f"Linear Regresion Train r2: {lr_train_r2:.2f}")
    print(f"Linear Regresion Test r2: {lr_test_r2:.2f}")
    print(f"Decision Tree Train r2: {dt_train_r2:.2f}")
    print(f"Decision Tree Test r2: {dt_test_r2:.2f}")
    print(f"Random Forest Train r2: {rf_train_r2:.2f}")
    print(f"Random Forest Test r2: {rf_test_r2:.2f}")
    print(f"XGBoost Train r2: {xgb_train_r2:.2f}")
    print(f"XGBoost Test r2: {xgb_test_r2:.2f}")

    # Print results table
    print("\nModel Performance Results:")
    print("Model\t\t\tTraining MAE\tTest MAE\tSamples (Train/Test)")
    print("-" * 70)
    print(f"Linear Regression\t{lr_train_mae:.2f}\t\t{lr_test_mae:.2f}\t\t{len(y_train)}/{len(y_test)}")
    print(f"Decision Tree\t\t{dt_train_mae:.2f}\t\t{dt_test_mae:.2f}\t\t{len(y_train)}/{len(y_test)}")
    print(f"Random Forest\t\t{rf_train_mae:.2f}\t\t{rf_test_mae:.2f}\t\t{len(y_train)}/{len(y_test)}")
    print(f"XGBoost\t\t\t\t{xgb_train_mae:.2f}\t\t{xgb_test_mae:.2f}\t\t{len(y_train)}/{len(y_test)}")

    # Analyze the range and average of the target variable
    y_min = y.min()
    y_max = y.max()
    y_mean = y.mean()

    print(f"\nTarget variable range: {y_min} to {y_max}")
    print(f"Mean of target variable: {y_mean:.2f}")

    # Compare MAE to 10% of the mean
    baseline_error_threshold = 0.1 * y_mean

    # For each model, compare its MAE to the baseline
    print("\nModel Evaluation (MAE vs. 10% of Mean):")
    print(f"Baseline MAE threshold (10% of mean): {baseline_error_threshold:.2f}")
    print(f"Linear Regression Test MAE: {lr_test_mae:.2f} {'(Good)' if lr_test_mae < baseline_error_threshold else '(Needs Improvement)'}")
    print(f"Decision Tree Test MAE: {dt_test_mae:.2f} {'(Good)' if dt_test_mae < baseline_error_threshold else '(Needs Improvement)'}")
    print(f"Random Forest Test MAE: {rf_test_mae:.2f} {'(Good)' if rf_test_mae < baseline_error_threshold else '(Needs Improvement)'}")
    print(f"XGBoost Test MAE: {xgb_test_mae:.2f} {'(Good)' if xgb_test_mae < baseline_error_threshold else '(Needs Improvement)'}")

    # Original visualization code
    for i in range(12):
        selected_x = X[:,i]
        plt.scatter(selected_x, y, alpha=0.2, c='green', s=0.5)
        plt.show(block=True)
    
    print("\n=== Starting Model Training ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10, max_features='sqrt', min_samples_split=20, min_samples_leaf=10),
        'Random Forest': RandomForestRegressor(random_state=42, max_depth=10, max_features='sqrt', n_estimators=500, min_samples_split=10, min_samples_leaf=5),
        'xgb_model': xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print results
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        # Plot predictions vs actual for each model
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Popularity')
        plt.ylabel('Predicted Popularity')
        plt.title(f'{name}: Predicted vs Actual Song Popularity')
        plt.tight_layout()
        plt.show(block=True)
        
        # Show feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.title(f'Feature Importance ({name})')
            plt.bar(range(X.shape[1]), importance[indices])
            plt.xticks(range(X.shape[1]), [Xnames[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.show(block=True)


if __name__ == '__main__':
    main()