## Predicting Song Popularity with Machine Learning

By CJ Phillips, Gemechu Taye, and Dylan Krim

### Overview

These files will run multiple machine learning models that will try to guess the popularity of songs.

### Paper Abstract

This paper explores the use of machine learning to predict song popularity using Spotify's dataset. 
By analyzing song features such as duration, loudness, and energy, we apply and compare models like Linear Regression, Decision Trees, Random Forests, and XGBoost. 
While XGBoost achieves the best results with the lowest Mean Absolute Error and highest R² score, overfitting remains a challenge. 
Insights from this study suggest instrumentalness, duration, and energy significantly influence popularity, whereas features like key and mode are less impactful.


### Instructions for running

1. Download the files
2. If you have an application like Visual Studio Code you can open the files in there and run the Songs.py file to see the results
3. OR if you do not you can run it through the terminal with Python Songs.py after navigating the folder

If for some reason it mentions it can not find the data you might need to edit a file path in util.py

### Acknowledgments

https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs - Spotify song dataset made by Joakim Arvidsson 

This work was made possible by the following Python libraries:
Numpy: For efficient numerical computations and array manipulations.
Matplotlib: For creating visualizations and graphical representations of data.
Scikit-learn (sklearn): For machine learning algorithms and model evaluation.
XGBoost: For implementing a powerful gradient boosting algorithm, enabling improved performance on complex datasets.

