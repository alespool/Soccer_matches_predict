import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from py_utilities.statistics_inference import test_normality_and_qq
from py_utilities.model_fitting import ModelFitter
from plotly.subplots import make_subplots
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import poisson


def create_qq_hist(data, column_name: str):
    """
    Test the normality of a column using the Anderson-Darling test and create a QQ plot.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to test for normality.

    Returns:
        tuple: A tuple containing the QQ plot figure and the histogram figure.
    """
    # QQ Plot using Plotly
    column_data = data[column_name].dropna()

    qq_fig = px.scatter(
        x=stats.probplot(column_data, dist="norm", fit=False)[0],
        y=stats.probplot(column_data, dist="norm", fit=False)[1],
        labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
        title="QQ Plot for " + column_name,
    )

    qq_fig.add_trace(go.Scatter(
        x=qq_fig.data[0].x,
        y=qq_fig.data[0].x, 
        mode='lines',
        name='Theoretical Quantiles',
        line=dict(color='red', dash='dash')
    ))

    # Histogram using Plotly
    hist_fig = px.histogram(column_data, nbins=20, title="Histogram for " + column_name)

    qq_fig.update_layout(
            width=800,
            height=600
    )

    hist_fig.update_layout(
            width=800,
            height=600
    )

    return qq_fig, hist_fig

class Plotter:
    def __init__(self):
        pass

    def plotly_histogram(self, df, x_col: str):
        """
        Generates a histogram plot using Plotly based on the quality column of a given DataFrame.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data.
            x_col (str): The name of the column to use as the x-axis.

        Returns:
            None
        """
        fig = px.histogram(df, x=x_col, nbins=10, title="Quality Histogram")
        fig.update_traces(marker_color="lightseagreen")
        fig.update_layout(
            xaxis_title="Quality",
            yaxis_title="Count",
            showlegend=False,
            width=800,
            height=600,
            bargap=0.1,
            bargroupgap=0.2,
        )

        fig.show()

    def plotly_correlation_heatmap(self, df,show_values=True):
        """
        Generates a correlation heatmap using Plotly based on the given DataFrame.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data for which the correlation heatmap is to be generated.

        Returns:
            None
        """
        corr = df.corr(numeric_only=True)
    
        fig = px.imshow(corr, color_continuous_scale='rdbu_r', labels=dict(color="Correlation"))
        fig.update_layout(
            title="Correlation Heatmap",
            width=800,
            height=600,
            xaxis=dict(tickangle=-45),
            yaxis=dict(title=''),
            # xaxis=dict(tickvals=[]),  # Hide x-axis labels
            # yaxis=dict(tickvals=[]),  # Hide y-axis labels
            coloraxis_colorbar=dict(title="Correlation"),
        )

        if show_values:
            for i in range(len(corr.columns)):
                for j in range(len(corr.index)):
                    fig.add_annotation(
                        x=i, y=j,
                        text=str(round(corr.iloc[j, i], 2)),
                        showarrow=False,
                    )

        fig.show()


    def seaborn_box_plots(self, df, x:str):
        """
        Generates box plots for attributes in the given DataFrame using Seaborn.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data for which box plots are to be generated.

        Returns:
            None
        """
        n_cols = 3  # Adjust the number of columns
        plot_width = 8  # Adjust the width of each plot
        plot_height = 5  # Adjust the height of each plot

        n_rows = (len(df.columns) - 1) // n_cols + 1  # Calculate the number of rows

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(plot_width * n_cols, plot_height * n_rows)
        )
        axes = axes.flatten()

        for i, column in enumerate(df.columns[:-1]):  # Exclude the 'quality' column
            ax = axes[i]
            sns.boxplot(data=df, x=x, y=column, ax=ax)
            ax.set_title(column)

        for j in range(i + 1, n_cols * n_rows):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def seaborn_distribution_plots(self, df):
        """
        Generates distribution plots for attributes in the given DataFrame using Seaborn.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data for which distribution plots are to be generated.

        Returns:
            None
        """
        n_cols = 4  # Adjust the number of columns
        plot_width = 6  # Adjust the width of each plot
        plot_height = 4  # Adjust the height of each plot

        n_rows = (len(df.columns) // n_cols) + 1  # Calculate the number of rows

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(plot_width * n_cols, plot_height * n_rows)
        )
        axes = axes.flatten()

        for i, column in enumerate(df.columns):
            ax = axes[i]
            sns.histplot(data=df, x=column, ax=ax)
            ax.set_title(column)

            ax.set_xticklabels([])
            ax.set_xticks([])

        for j in range(i + 1, n_cols * n_rows):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_qq_hist(self, group, column_name: str):
        """
        Generate a QQ plot and histogram for a given group and test if the data follows a normal
        distribution.

        Parameters:
            group (DataFrame): The group to analyze.
            column_name (str): The column name to make plots over.

        Returns:
            fig (plot): The QQ plot of the data.
            hist_fig (plot): The histogram of the data.
        """
        is_normal, p = test_normality_and_qq(group, column_name=column_name)

        fig, hist_fig = create_qq_hist(group, column_name=column_name)

        if is_normal:
            print("The data follows a normal distribution.")
            print(f"P-value: {p}")
        else:
            print("The data does not follow a normal distribution.")
            print(f"P-value: {p}")

        return fig, hist_fig

    def create_histogram(self, dataframe, label):
        # Extract metrics for individual classes (excluding non-class labels)
        class_metrics_df = dataframe.loc[["3", "4", "5", "6", "7", "8"]]

        # Create an empty list to store the subplots
        subplot_list = []

        for class_label, class_data in class_metrics_df.iterrows():
            if "support" in class_data:
                class_data = class_data.drop("support")

            # Ensure both 'Test' and 'Train' values exist
            if all(x in class_data.index for x in ["Train", "Test"]):
                df = pd.DataFrame(
                    {
                        "subset": ["Train", "Test"],
                        "value": class_data[["Train", "Test"]].values,
                    }
                )

                # Create a subplot for the class label
                subplot = px.bar(
                    df,
                    x="subset",
                    y="value",
                    labels={class_label: label},
                    title=f"{class_label} ({label})",
                )

                # Append the subplot to the list
                subplot_list.append(subplot)

        return subplot_list


    def plot_confusion_matrix(self, confusion_matrix, class_labels, title="Confusion Matrix"):
        """
        Create a confusion matrix plot using Plotly.

        Parameters:
            confusion_matrix (list of lists): The confusion matrix.
            class_labels (list of str): Labels for classes.
            title (str): Title for the plot.

        Returns:
            None
        """
        z_text = [[str(y) for y in row] for row in confusion_matrix]

        fig = ff.create_annotated_heatmap(
            z=confusion_matrix,
            x=class_labels,
            y=class_labels,
            annotation_text=z_text,
            colorscale="Viridis",
        )

        fig.update_layout(
            title_text=f"<i><b>{title}</b></i>",
            xaxis=dict(title="Predicted value"),
            yaxis=dict(title="Real value", autorange="reversed"),
            width=800,
            height=300
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=0.5,
                y=-0.15,
                showarrow=False,
                text="Predicted value",
                xref="paper",
                yref="paper",
            )
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=-0.35,
                y=0.5,
                showarrow=False,
                text="Real value",
                textangle=-90,
                xref="paper",
                yref="paper",
            )
        )

        fig.update_layout(margin=dict(t=50, l=200))
        fig["data"][0]["showscale"] = True
        fig.show()


    def plot_multiclass_roc_auc_curve(self, y_probs, y_test):
        """
        Plots the ROC-AUC curve for multiclass classification using Plotly.

        Args:
            y_probs (array): Predicted class probabilities for each class.
            y_test (array): True labels for each sample.

        Returns:
            None
        """
        y_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs)
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig = go.Figure()

        colors = ['blue', 'green', 'red']

        for i, color in zip(range(n_classes), colors):
            fig.add_trace(go.Scatter(
                x=fpr[i],
                y=tpr[i],
                mode='lines',
                name=f'Class {i} (AUC = {roc_auc[i]:.2f})',
                line=dict(color=color),
                fill='tonexty'
            ))

        fig.add_shape(
            type='line',
            x0=0, x1=1,
            y0=0, y1=1,
            line=dict(dash='dash')
        )

        fig.update_xaxes(title_text='False Positive Rate')
        fig.update_yaxes(title_text='True Positive Rate')

        fig.update_layout(
            title='ROC-AUC Curve for Multiclass Classification',
            width=800,
            height=600,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )

        fig.show()


    def plot_regression_table(self, df):
        """
        Generates a regression table plot using the provided dataframe.

        Parameters:
        - df: The dataframe containing the regression metrics.

        Returns:
        None
        """
        fig = make_subplots(rows=1, cols=1)

        trace = go.Table(
            header=dict(values=['Type'] + df.columns, fill_color='paleturquoise', align='left'),
            cells=dict(values=[df.index, df['Metric'], df['Value']],
                        fill_color='lavender',
                        align='left',
                        height=35)
        )

        fig.add_trace(trace)

        fig.update_layout(
            title='Regression Model Evaluation Metrics',
            margin=dict(l=50, r=50, t=30, b=0),
            width=800,
            height=300
        )
        fig.show()

    def show_regression_predictions(self, train_predictions, test_predictions):
        """
        Generates a plot to display the regression predictions for both the training and testing datasets.

        Parameters:
            train_predictions (list): A list of predicted values for the training dataset.
            test_predictions (list): A list of predicted values for the testing dataset.

        Returns:
            None
        """
        fig = go.Figure()

        train_trace = go.Scatter(x=np.arange(len(train_predictions)), y=train_predictions, mode='lines', name='Train', line=dict(color='blue', width=2))
        test_trace = go.Scatter(x=np.arange(len(test_predictions)), y=test_predictions, mode='lines', name='Test', line=dict(color='red', width=2))

        fig.add_trace(train_trace)
        fig.add_trace(test_trace)

        train_mean = sum(train_predictions) / len(train_predictions)
        test_mean = sum(test_predictions) / len(test_predictions)

        fig.add_shape(go.layout.Shape(type="line", x0=0, x1=len(train_predictions), y0=train_mean, y1=train_mean, line=dict(color="blue", width=2)))
        fig.add_shape(go.layout.Shape(type="line", x0=0, x1=len(test_predictions), y0=test_mean, y1=test_mean, line=dict(color="red", width=2)))

        fig.update_layout(
            title='Training and Testing Predictions',
            xaxis=dict(title='Sample Index'),
            yaxis=dict(title='Predicted Values'),
            width=800,
            height=600,
            showlegend=True
        )

        fig.show()

        return test_mean, train_mean

    def plot_coefficients(self, coefs, feature_names):
        """
        Generate a bar plot of the coefficients for each feature in a linear regression model.

        Parameters:
            coefs (array-like): The coefficients of the linear regression model.
            feature_names (array-like): The names of the features.

        Returns:
            fig (plotly.graph_objs._figure.Figure): The bar plot of the coefficients.
        """
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
        coef_df = coef_df.round({'Coefficient': 2}) 
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False) 

        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', text='Coefficient')
        fig.update_layout(
            title='Linear Regression Coefficients',
            xaxis_title='Coefficient Value',
            yaxis_title='Feature',
            showlegend=False,
            width=800,
            height=600,
            xaxis_range=[-5.5,5.5]
        )
        
        return fig
    
    def plot_features_importance(self, model,X,y):
        
        importance = permutation_importance(model, X, y, scoring='neg_mean_squared_error')

        sorted_idx = importance.importances_mean.argsort()


        fig = go.Figure(data=go.Bar(
            x=importance.importances_mean[sorted_idx],
            y=[X.columns[i] for i in sorted_idx],
            orientation='h'
        ))

        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            width=800,
            height=600,
            yaxis=dict(autorange="reversed")
        )

        fig.show()

    def plot_regression_correlation(self, df, x: str, y: str, title: str, poly:bool = True):
        """
        Generate a scatter plot with a regression line to visualize the correlation between two variables.

        Parameters:
            df (DataFrame): The pandas DataFrame containing the data.
            X (str): The column name for the x-axis variable.
            y (str): The column name for the y-axis variable.
            title (str): The title of the plot.

        Returns:
            None
        """
        fig = px.scatter(df, x=x, y=y, title=title, trendline='ols')
        fig.update_traces(textposition='top center')

        if poly:
            df['smooth'] = np.poly1d(np.polyfit(df[x], df[y], 3))(df[x])
            fig.add_trace(go.Scatter(x=df[x], y=df['smooth'], name='Polynomial Fit', mode='lines', line=dict(color='red', width=2)))

        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            width=800,
            height=600
        )

        fig.show()

    def plot_explained_variance(self, explained_variance):
        """
        Generates a plot to visualize the cumulative explained variance ratio for principal components analysis (PCA).

        Parameters:
            explained_variance (list): A list of explained variances for each principal component.

        Returns:
            None
        """

        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio for PCA')
        plt.grid(True)
        plt.show()

    
    def plot_histograms_and_poisson(self, data_group_a, data_group_b, mean_home_goals, mean_away_goals):
        """
        Plot histograms and Poisson distributions.
        
        Args:
            data_group_a (array-like): The data for group A.
            data_group_b (array-like): The data for group B.
            mean_home_goals (float): The mean of home goals.
            mean_away_goals (float): The mean of away goals.
        
        Returns:
            None
        """
        # Plot histograms
        hist_group_a = go.Histogram(x=data_group_a, histnorm='probability', name='Home Goals', opacity=0.5)
        hist_group_b = go.Histogram(x=data_group_b, histnorm='probability', name='Away Goals', opacity=0.5)

        # Plot Poisson distributions
        x = list(range(max(max(data_group_a), max(data_group_b)) + 1))
        poisson_home_goals = go.Scatter(x=x, y=poisson.pmf(x, mean_home_goals), mode='lines+markers', name='Poisson (Home Goals)', marker=dict(color='blue'))
        poisson_away_goals = go.Scatter(x=x, y=poisson.pmf(x, mean_away_goals), mode='lines+markers', name='Poisson (Away Goals)', marker=dict(color='red'))

        # Layout settings
        layout = go.Layout(
            title='Poisson Distribution of Home and Away Goals',
            xaxis=dict(title='Number of Goals'),
            yaxis=dict(title='Probability'),
            legend=dict(x=0.7, y=0.9),
            barmode='overlay'
        )

        fig = go.Figure(data=[hist_group_a, hist_group_b, poisson_home_goals, poisson_away_goals], layout=layout)
        fig.show()