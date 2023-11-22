import numpy as np
from scipy import stats
import pandas as pd
import xml.etree.ElementTree as ET


def parse_xml(xml_document):
    """
    Parse the XML document and return the root element.
    """
    return ET.fromstring(xml_document)

def calculate_card_stats(tree, home_team, away_team, card_type='y'):
    """
    Calculate card statistics for both the home team and the away team.
    """
    assert card_type == 'y' or card_type == 'r', "Please enter either y or r"
    
    stat_home_team = 0
    stat_away_team = 0
    
    for child in tree.iter('value'):
        try:
            if child.find('comment').text == card_type:
                if int(child.find('team').text) == home_team:
                    stat_home_team += 1
                else:
                    stat_away_team += 1
        except AttributeError:
            pass
    
    return stat_home_team, stat_away_team

def calculate_possession_stats(tree):
    """
    Calculate possession statistics for both the home team and the away team.
    """
    try:
        last_value = [child for child in tree.iter('value')][-1]
        return int(last_value.find('homepos').text), int(last_value.find('awaypos').text)
    except:
        return None, None

def calculate_other_stats(tree, home_team, away_team):
    """
    Calculate other statistics for both the home team and the away team.
    """
    stat_home_team = 0
    stat_away_team = 0
    
    for team in [int(stat.text) for stat in tree.findall('value/team')]:
        if team == home_team: 
            stat_home_team += 1
        else:
            stat_away_team += 1
    
    return stat_home_team, stat_away_team

def calculate_stats_both_teams(xml_document, home_team, away_team, card_type='y'):
    """
    Calculates the statistics for both the home team and the away team based on the provided XML document.
    
    Args:
        xml_document (str): The XML document containing the statistics.
        home_team (int): The ID of the home team.
        away_team (int): The ID of the away team.
        card_type (str, optional): The type of card to consider. Defaults to 'y'.
        
    Returns:
        tuple: A tuple containing the statistics for the home team and the away team.
            The statistics are represented as integers.
    """
    tree = parse_xml(xml_document)
    
    if tree.tag == 'card':
        return calculate_card_stats(tree, home_team, away_team, card_type)
    elif tree.tag == 'possession':
        return calculate_possession_stats(tree)
    else:
        return calculate_other_stats(tree, home_team, away_team)

class DataCleaner:
    def __init__(self, threshold: float = 0.5):
        """Initialize the DataCleaner class with a threshold value."""
        self.threshold = threshold

    def get_outliers_scores(self, df, quantile: float = 0.999, method: str = 'z_score'):
        """
        Calculate the count of outliers in each column of a given DataFrame df, based on a specified threshold.
        The function uses either the z-score method or the IQR method to determine the outliers.

        Parameters:
            df (pandas DataFrame): The input DataFrame containing numerical values.
            quantile (float): The threshold value to determine outliers. It should be between 0 and 1, representing the quantile value.
            method (str): The method to use for determining outliers. Can be either 'z_score' or 'IQR'.

        Returns:
            outliers (pandas Series): A Series object containing the count of outliers in each column of the DataFrame.
                                    The index of the Series corresponds to the column names in the DataFrame.
        """
        df = df.copy()

        if method == 'z_score':
            quantile_z_score = stats.norm.ppf(quantile)
            z_score_df = pd.DataFrame(np.abs(stats.zscore(df)), columns=df.columns)
            outliers = (z_score_df > quantile_z_score).sum(axis=0)

        elif method == 'IQR':
            q1 = df.quantile(1 - quantile)
            q3 = df.quantile(quantile)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = ((df < lower_bound) | (df > upper_bound)).sum(axis=0)

        return outliers

    def calculate_skewness(self, df):
        """
        Calculates the skewness of each column in a given DataFrame.

        Parameters:
            df (pandas DataFrame): The input DataFrame containing numerical values.

        Returns:
            pandas.Series: A Series containing the skewness values for each column in the DataFrame.
        """
        return df.apply(lambda x: stats.skew(x))

    def calculate_kurtosis(self, df):
        """
        Calculates the kurtosis of each column in a given DataFrame.

        Parameters:
            df (pandas DataFrame): The input DataFrame containing numerical values.

        Returns:
            pandas.Series: A Series containing the kurtosis values for each column in the DataFrame.
        """
        return df.apply(lambda x: stats.kurtosis(x))

    def apply_log_transformation(self, df):
        """
        Apply log transformation to the given dataframe columns that have a skewness greater
        than the threshold.

        Parameters:
            df (pandas DataFrame): The input DataFrame containing numerical values.

        Returns:
            The transformed dataframe after applying log transformation.
        """
        skewness = self.calculate_skewness(df)
        skewed_columns = skewness[skewness > self.threshold].index
        print("Applying log transformation to the following columns:")
        for n, col in enumerate(skewed_columns):
            print(n + 1, " - ", col)
        df[skewed_columns] = df[skewed_columns].apply(lambda x: np.log1p(x))
        return df

    def apply_box_cox_transformation(self, df):
        """
        Apply box cox transformation to the given dataframe columns that have a skewness greater
        than the threshold.

        Parameters:
            df (pandas DataFrame): The input DataFrame containing numerical values.

        Returns:
            The transformed dataframe after applying log transformation.
        """
        skewness = self.calculate_skewness(df)
        skewed_columns = skewness[skewness > self.threshold].index
        print("Applying Box-Cox transformation to the following columns:")
        for n, col in enumerate(skewed_columns):
            print(n + 1, " - ", col)
        for col in skewed_columns:
            df[col], _ = stats.boxcox(df[col])
        return df

    def remove_outliers(self, data, method: str = 'z_score', threshold: float = 3, quantile: float = 0.75):
        """
        Remove outliers from the given data using the specified method.

        Parameters:
            data (np.ndarray): The input data array.
            method (str, optional): The method to use for outlier detection. Defaults to 'z_score'.
            threshold (float, optional): The threshold value for outlier detection. Defaults to 1.5.
            quantile (float, optional): The quantile value for outlier detection when method is 'IQR'. Defaults to 0.75.

        Returns:
            np.ndarray: The data array with outliers removed.
        """
        if method == 'z_score':
            z_scores = np.abs(stats.zscore(data))
            outliers = (z_scores > threshold).any(axis=1)
            data_no_outliers = data[~outliers]
        elif method == 'IQR':
            q1 = data.quantile(1 - quantile)
            q3 = data.quantile(quantile)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (data < lower_bound) | (data > upper_bound)
            
            outlier_rows = outliers.sum(axis=1) > 0 
            data_no_outliers = data[~outlier_rows]
            
        else:
            raise ValueError("Method must be 'z_score' or 'IQR'.")

        if method == 'IQR':
            return data_no_outliers
        else:
            return data_no_outliers

    def missing_values(self, df,null_percent=0):
        """
        Calculate the count and proportion of missing values in a DataFrame.

        Parameters:
            df (DataFrame): The input DataFrame.
            null_percent (float): The threshold percentage for missing values. 
                Default is 0, which means all missing values will be included.

        Returns:
            DataFrame: A DataFrame containing the count and proportion of missing values for each column.
                The columns are 'missing_count' and 'missing_prop'.
                The index of the DataFrame is the column names.

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [3, None, 5]})
            >>> missing_values(df)
                   missing_count  missing_prop
            A                1.0          33.3
            B                1.0          33.3
        """

        output_df = pd.DataFrame({'missing_count':[],'missing_prop':[]})
        nullcount_df = df.isna().sum()
        output_df['missing_count'] = nullcount_df.iloc[0:]
        output_df['missing_prop'] = output_df['missing_count']/len(df.index)*100
        output_df.index=nullcount_df.index

        if null_percent>0:
            return output_df[output_df['missing_prop']>=null_percent]
        else:
            return output_df
        
    