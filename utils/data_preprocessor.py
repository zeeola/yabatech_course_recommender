import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for YABATECH Course Recommendation System
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = [
            'gender', 'state_of_origin', 'lga', 'previous_education',
            'preferred_study_mode', 'employment_status', 'course_preference_1',
            'course_preference_2', 'course_preference_3'
        ]
        self.numerical_columns = [
            'age', 'jamb_score', 'o_level_credits', 'years_of_experience'
        ]
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load synthetic dataset from CSV file
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def load_course_requirements(self, json_path: str) -> Dict[str, Any]:
        """
        Load course requirements from JSON file
        
        Args:
            json_path (str): Path to the JSON file
            
        Returns:
            Dict: Course requirements data
        """
        try:
            with open(json_path, 'r') as f:
                course_requirements = json.load(f)
            logger.info("Course requirements loaded successfully")
            return course_requirements
        except Exception as e:
            logger.error(f"Error loading course requirements: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_clean = df.copy()
        
        # Fill missing numerical values with median
        for col in self.numerical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill missing categorical values with mode
        for col in self.categorical_columns:
            if col in df_clean.columns:
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
        
        logger.info("Missing values handled successfully")
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the encoders or use existing ones
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    # Fit and transform during training
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Transform using existing encoder during prediction
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        unique_values = set(df_encoded[col].astype(str))
                        known_values = set(le.classes_)
                        unseen_values = unique_values - known_values
                        
                        if unseen_values:
                            # Add unseen values to encoder classes
                            le.classes_ = np.append(le.classes_, list(unseen_values))
                        
                        df_encoded[col] = le.transform(df_encoded[col].astype(str))
        
        logger.info("Categorical features encoded successfully")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Dataframe with scaled numerical features
        """
        df_scaled = df.copy()
        
        numerical_cols_present = [col for col in self.numerical_columns if col in df_scaled.columns]
        
        if numerical_cols_present:
            if fit:
                df_scaled[numerical_cols_present] = self.scaler.fit_transform(df_scaled[numerical_cols_present])
            else:
                df_scaled[numerical_cols_present] = self.scaler.transform(df_scaled[numerical_cols_present])
        
        logger.info("Numerical features scaled successfully")
        return df_scaled
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for better recommendations
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        df_features = df.copy()
        
        # Create age groups
        if 'age' in df_features.columns:
            df_features['age_group'] = pd.cut(df_features['age'], 
                                            bins=[0, 20, 25, 30, 35, 100], 
                                            labels=['<20', '20-24', '25-29', '30-34', '35+'])
        
        # Create JAMB score categories
        if 'jamb_score' in df_features.columns:
            df_features['jamb_category'] = pd.cut(df_features['jamb_score'],
                                                bins=[0, 180, 220, 260, 400],
                                                labels=['Below Average', 'Average', 'Good', 'Excellent'])
        
        # Create experience level
        if 'years_of_experience' in df_features.columns:
            df_features['experience_level'] = pd.cut(df_features['years_of_experience'],
                                                   bins=[-1, 0, 2, 5, 100],
                                                   labels=['No Experience', 'Entry Level', 'Mid Level', 'Senior Level'])
        
        # Create academic strength indicator
        if 'o_level_credits' in df_features.columns and 'jamb_score' in df_features.columns:
            df_features['academic_strength'] = (df_features['o_level_credits'] * 0.4 + 
                                              df_features['jamb_score'] * 0.6 / 40)  # Normalize JAMB score
        
        logger.info("Additional features created successfully")
        return df_features
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'recommended_course') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training the recommendation model
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Create additional features
        df_features = self.create_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features, fit=True)
        
        # Scale numerical features
        df_scaled = self.scale_numerical_features(df_encoded, fit=True)
        
        # Separate features and target
        if target_column in df_scaled.columns:
            X = df_scaled.drop(columns=[target_column])
            y = df_scaled[target_column]
        else:
            X = df_scaled
            y = None
            logger.warning(f"Target column '{target_column}' not found")
        
        logger.info("Training data prepared successfully")
        return X, y
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for making predictions
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Preprocessed features for prediction
        """
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Create additional features
        df_features = self.create_features(df_clean)
        
        # Encode categorical features (without fitting)
        df_encoded = self.encode_categorical_features(df_features, fit=False)
        
        # Scale numerical features (without fitting)
        df_scaled = self.scale_numerical_features(df_encoded, fit=False)
        
        logger.info("Prediction data prepared successfully")
        return df_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def get_feature_names(self) -> list:
        """
        Get list of feature names after preprocessing
        
        Returns:
            list: Feature names
        """
        feature_names = []
        
        # Add categorical columns
        feature_names.extend(self.categorical_columns)
        
        # Add numerical columns
        feature_names.extend(self.numerical_columns)
        
        # Add created features
        created_features = ['age_group', 'jamb_category', 'experience_level', 'academic_strength']
        feature_names.extend(created_features)
        
        return feature_names
    
    def save_preprocessors(self, encoder_path: str, scaler_path: str):
        """
        Save fitted preprocessors
        
        Args:
            encoder_path (str): Path to save label encoders
            scaler_path (str): Path to save scaler
        """
        import pickle
        
        # Save label encoders
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info("Preprocessors saved successfully")
    
    def load_preprocessors(self, encoder_path: str, scaler_path: str):
        """
        Load fitted preprocessors
        
        Args:
            encoder_path (str): Path to load label encoders
            scaler_path (str): Path to load scaler
        """
        import pickle
        
        # Load label encoders
        with open(encoder_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info("Preprocessors loaded successfully")

# Utility functions
def validate_input_data(data: Dict[str, Any]) -> bool:
    """
    Validate input data for course recommendation
    
    Args:
        data (Dict): Input data dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['age', 'jamb_score', 'o_level_credits', 'gender', 'state_of_origin']
    
    for field in required_fields:
        if field not in data or data[field] is None:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate numerical ranges
    if not (16 <= data['age'] <= 50):
        logger.error("Age must be between 16 and 50")
        return False
    
    if not (0 <= data['jamb_score'] <= 400):
        logger.error("JAMB score must be between 0 and 400")
        return False
    
    if not (0 <= data['o_level_credits'] <= 9):
        logger.error("O-Level credits must be between 0 and 9")
        return False
    
    return True

def format_user_input(user_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Format user input into DataFrame for preprocessing
    
    Args:
        user_data (Dict): User input data
        
    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    # Create a single-row DataFrame
    df = pd.DataFrame([user_data])
    
    # Fill default values for missing optional fields
    defaults = {
        'lga': 'Unknown',
        'previous_education': 'Secondary',
        'preferred_study_mode': 'Full-time',
        'employment_status': 'Unemployed',
        'years_of_experience': 0,
        'course_preference_1': 'None',
        'course_preference_2': 'None',
        'course_preference_3': 'None'
    }
    
    for key, value in defaults.items():
        if key not in df.columns:
            df[key] = value
    
    return df