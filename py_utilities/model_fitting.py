from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
import logging
from sklearn.ensemble import RandomForestClassifier

class ModelFitter:
    def __init__(self):
        """
        Initialize the ModelFitter with a DataFrame.

        Parameters:
            df (pandas.DataFrame): The input DataFrame.
        """
        pass

    def rescale_data(self, X, scaler:str="standard"):
        """
        Rescales the given data using the specified scaler.

        Parameters:
            X (DataFrame): The input data to be rescaled.
            scaler (str, optional): The type of scaler to be used. Defaults to "standard".

        Returns:
            DataFrame: The rescaled data.
        """

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if scaler == 'standard':
            scaler = StandardScaler()
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
            
        X_rescaled = X.copy()
        X_rescaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X_rescaled
    
    def split_data(self, X, y, test_size=0.25):
        """
        Splits the data into training and testing sets.

        Parameters:
            X (array-like): The input features.
            y (array-like): The target variable.
            test_size (float): The proportion of the data to be used for testing. Default is 0.25.

        Returns:
            X_train (array-like): The training features.
            X_test (array-like): The testing features.
            y_train (array-like): The training target variable.
            y_test (array-like): The testing target variable.
        """
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=42)
        
        return X_train, X_test, y_train, y_test

    def classification_model(self, X_train, y_train, smote=False):
        """
        This function builds a classification model using logistic regression.

        Args:
            X_train (array-like): The training data.
            y_train (array-like): The target variable for the training data.
            smote (bool, optional): Whether to apply SMOTE for oversampling. Defaults to False.

        Returns:
            tuple: A tuple containing the grid search object and the coefficients of the best estimator.
        """
        
        # Define the pipeline
        if smote == False:
                pipeline = imbpipeline(steps = [['classifier', LogisticRegression(random_state=42)]])
        else:
            smote = SMOTE(random_state = 42, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            pipeline = Pipeline(steps = [['classifier', LogisticRegression(random_state=42)]])
        
        stratified_kfold = KFold(n_splits=5,
                                        shuffle=True,
                                        random_state=42)
        
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__solver': [ 'newton-cg', 'sag', 'saga'],
            'classifier__max_iter': [10, 100, 500, 3000, 5000, 7000],
            'classifier__tol': [0.001, 0.01, 0.1, 1],
            'classifier__fit_intercept': [True, False]
        }

        grid_search = GridSearchCV(estimator=pipeline,
                                param_grid=param_grid,
                                scoring='f1_samples',
                                cv=stratified_kfold,
                                n_jobs=-1)

        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        coefs = grid_search.best_estimator_.named_steps.classifier.coef_

        return grid_search
    
    def evaluate_classification_model(self, model, X, y, data_type='test'):
        """Create a classification report for metrics."""
        if data_type == 'train':
            y_pred = model.predict(X)
        else:
            y_pred = model.predict(X)
            
        class_report = classification_report(y, y_pred, output_dict=True)
        logging.info(f"{data_type.capitalize()} Classification Report: {class_report}")
        
        return class_report, y_pred
    
    def evaluate_classification_confusion(self, model, X_train, y_train, X_test, y_test, use_train_predictions=True):
        """
        Evaluate a classification model using the provided data and predictions.
    
        Parameters:
            model: The trained classification model.
            X_train: Feature variables for training data.
            y_train: True labels for training data.
            X_test: Feature variables for testing data.
            y_test: True labels for testing data.
            use_train_predictions: If True, use training data predictions for the confusion matrix. Otherwise, use testing data predictions.
    
        Returns:
            dict: A dictionary containing confusion matrix.
        """
        if use_train_predictions:
            y_pred = model.predict(X_train)
        else:
            y_pred = model.predict(X_test)
    
        conf_matrix = confusion_matrix(y_train if use_train_predictions else y_test, y_pred)
    
        return conf_matrix
