import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GeochemistryDataProcessor:
    """Class for processing geochemistry data"""
    
    @staticmethod
    def load_data(file_path, sheet_name=0):
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    @staticmethod
    def clean_data(df):
        """Clean and preprocess geochemistry data"""
        # Remove rows with all NaN values
        df_clean = df.dropna(how='all')
        
        # Fill missing values with column median for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        return df_clean
    
    @staticmethod
    def detect_outliers(df, column, threshold=3):
        """Detect outliers using Z-score method"""
        if column not in df.columns:
            return pd.Series([False] * len(df))
        
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outlier_mask = z_scores > threshold
        
        # Create full series with False for NaN values
        result = pd.Series([False] * len(df))
        valid_indices = df[column].dropna().index
        result.loc[valid_indices] = outlier_mask
        
        return result
    
    @staticmethod
    def calculate_statistics(df):
        """Calculate descriptive statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats_dict = {
            'mean': df[numeric_cols].mean(),
            'median': df[numeric_cols].median(),
            'std': df[numeric_cols].std(),
            'min': df[numeric_cols].min(),
            'max': df[numeric_cols].max(),
            'skewness': df[numeric_cols].skew(),
            'kurtosis': df[numeric_cols].kurtosis()
        }
        
        return pd.DataFrame(stats_dict)
    
    @staticmethod
    def normalize_data(df, method='standard'):
        """Normalize data using different methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
            df_normalized = df.copy()
            df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif method == 'minmax':
            df_normalized = df.copy()
            for col in numeric_cols:
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif method == 'log':
            df_normalized = df.copy()
            for col in numeric_cols:
                # Add small constant to avoid log(0)
                df_normalized[col] = np.log1p(df[col])
        else:
            df_normalized = df.copy()
        
        return df_normalized
    
    @staticmethod
    def calculate_correlation(df, method='pearson'):
        """Calculate correlation matrix"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'pearson':
            corr_matrix = df[numeric_cols].corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = df[numeric_cols].corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = df[numeric_cols].corr(method='kendall')
        else:
            corr_matrix = df[numeric_cols].corr()
        
        return corr_matrix
    
    @staticmethod
    def filter_by_quantile(df, column, lower_q=0.25, upper_q=0.75):
        """Filter data based on quantile ranges"""
        lower_bound = df[column].quantile(lower_q)
        upper_bound = df[column].quantile(upper_q)
        
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        return filtered_df
