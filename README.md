# Football Matches Prediction ⚽

## Project Overview

<div style=display: flex; justify-content: center;>
    <img width=400px height=auto src=https://images.unsplash.com/photo-1489944440615-453fc2b6a9a9?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1482&q=80 />
</div>

This project focuses on the analysis of football matches based on a dataset obtained from [Kaggle on Football Games](https://www.kaggle.com/datasets/prajitdatta/ultimate-25k-matches-football-database-european). The dataset comprises various tables related to football teams, players, leagues and matches.

### Objectives

In this project, we aim to accomplish the following objectives:

- Perform exploratory data analysis (EDA) on the dataset and its tables to understand the relationships between various features.
- Conduct statistical inference to determine if teams playing at home score more than teams playing away.
- Apply linear machine learning models to predict the final outcome of a match.
- Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.
- Create a dashboard using a BI tool to visualize the key metrics and insights from the models.

### Dataset Overview

Database for data analysis and machine learning on football matches.

- +25,000 matches
- +10,000 players
- 11 European Countries with their lead championship
- Seasons 2008 to 2016
- Players and Teams' attributes* sourced from EA Sports' FIFA video game series, including the weekly updates - - Team line up with squad formation (X, Y coordinates)
- Betting odds from up to 10 providers
- Detailed match events (goal types, possession, corner, cross, fouls, cards etc…) for +10,000 matches

## Table of Contents

- [Data Extraction and Cleaning](#Data-Extraction-and-Cleaning)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Statistical Inference](#Statistical-Inference)
- [Data pre-processing](#Data-pre-processing)
- [Linear Model predictions](#Linear-Model-predictions)
- [Bonus Challenge](#Bonus-Challenge)
- [Conclusions](#conclusions)

## Exploratory Data Analysis

In this section, we explore the football matches dataset by performing the following:

- Data summary and statistics
- Correlation analysis between variables
- Detection of outliers
- Data visualization to identify key trends

## Statistical Inference

We conduct statistical inference to answer a specific question: Do teams playing at home score more than teams playing away? We explore this question using non-parametric tests.

- Hypothesis Testing: We use Mann-Whitney U test to determine the difference.

## Data pre-processing

Before fitting linear machine learning models, we prepare the dataset by:

- Removing unused columns.
- Handling multicollinearity using interaction terms.
- Removing outliers.

## Linear Model Predictions for Wine Quality

We use a linear logistic regression model to predict the matches outocme, aiming to understand which features are significant.

## Looker Studio Report

## Conclusions

In this section, we summarize our findings, discuss potential improvements for the analysis, and suggest ways to further enhance the accuracy and usefulness of the models for predicting 

## Getting Started

To reproduce this analysis, follow these steps:

**Dependencies**:
- Python 3.x
- Required libraries listed in the requirements.txt file

**Usage**:
- Clone this repository.
- Install the required dependencies using `pip install -r requirements.txt`.
- Download the dataset from Kaggle place it in the 'data/raw_data' folder.
- Run the provided Python script.

## License

This project is licensed under the [MIT License](LICENSE).