"""
YABATECH Course Recommendation System - Model Training Script
============================================================

This script handles the complete machine learning pipeline:
1. Data loading and preprocessing
2. Feature engineering and encoding
3. Model training and evaluation
4. Model persistence

Author: YABATECH AI Team
Date: 2025
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import re
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class YabatechModelTrainer:
    """
    Comprehensive model training class for YABATECH course recommendation system
    """
    
    def __init__(self, data_path='data/synthetic_dataset.csv', 
                 requirements_path='data/course_requirements.json',
                 model_dir='models/'):
        self.data_path = data_path
        self.requirements_path = requirements_path
        self.model_dir = model_dir
        self.grade_mapping = {
            'A1': 1, 'B2': 2, 'B3': 3, 'C4': 4, 'C5': 5, 'C6': 6,
            'D7': 7, 'E8': 8, 'F9': 9
        }
        self.subject_encoder = LabelEncoder()
        self.course_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_data(self):
        """Load and validate dataset"""
        print("📂 Loading dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded: {len(self.df)} records")
            print(f"📊 Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def load_course_requirements(self):
        """Load course requirements for rule-based filtering"""
        print("📋 Loading course requirements...")
        try:
            with open(self.requirements_path, 'r') as f:
                self.course_requirements = json.load(f)
            print(f"✅ Course requirements loaded: {len(self.course_requirements)} courses")
            return True
        except Exception as e:
            print(f"❌ Error loading course requirements: {e}")
            self.course_requirements = {}
            return False
    
    def parse_waec_grades(self, waec_string):
        """Parse WAEC grades string and return numerical scores"""
        if pd.isna(waec_string) or waec_string == 'None':
            return []
        
        # Split grades and clean
        grades = [grade.strip() for grade in str(waec_string).split(',')]
        numerical_grades = []
        
        for grade in grades:
            if grade in self.grade_mapping:
                numerical_grades.append(self.grade_mapping[grade])
            else:
                # Handle edge cases
                numerical_grades.append(5)  # Default to C5
        
        return numerical_grades
    
    def parse_waec_subjects(self, subjects_string):
        """Parse WAEC subjects string"""
        if pd.isna(subjects_string) or subjects_string == 'None':
            return []
        
        subjects = [subj.strip() for subj in str(subjects_string).split(',')]
        return subjects
    
    def parse_jamb_subjects(self, jamb_string):
        """Parse JAMB subjects string"""
        if pd.isna(jamb_string) or jamb_string == 'None':
            return []
        
        subjects = [subj.strip() for subj in str(jamb_string).split(',')]
        return subjects
    
    def calculate_waec_features(self, row):
        """Calculate comprehensive WAEC features"""
        waec_grades = self.parse_waec_grades(row['WAEC_Grades'])
        waec_subjects = self.parse_waec_subjects(row['WAEC_Subjects'])
        
        features = {}
        
        if waec_grades:
            features['waec_avg_grade'] = np.mean(waec_grades)
            features['waec_best_grade'] = min(waec_grades)  # Lower is better
            features['waec_worst_grade'] = max(waec_grades)
            features['waec_num_subjects'] = len(waec_grades)
            features['waec_credits'] = sum(1 for grade in waec_grades if grade <= 6)  # C6 and above
            features['waec_distinctions'] = sum(1 for grade in waec_grades if grade <= 3)  # B3 and above
        else:
            features.update({
                'waec_avg_grade': 7, 'waec_best_grade': 9, 'waec_worst_grade': 9,
                'waec_num_subjects': 0, 'waec_credits': 0, 'waec_distinctions': 0
            })
        
        # Subject-specific features
        key_subjects = ['Mathematics', 'English Language', 'Physics', 'Chemistry', 'Biology']
        for subject in key_subjects:
            has_subject = any(subject.lower() in subj.lower() for subj in waec_subjects)
            features[f'has_{subject.lower().replace(" ", "_")}'] = int(has_subject)
        
        return features
    
    def calculate_jamb_features(self, row):
        """Calculate JAMB-related features"""
        jamb_subjects = self.parse_jamb_subjects(row['JAMB_Subjects'])
        
        features = {
            'jamb_score': row.get('JAMB_Score', 0),
            'jamb_cutoff': row.get('JAMB_Cutoff', 150),
            'jamb_above_cutoff': int(row.get('JAMB_Score', 0) >= row.get('JAMB_Cutoff', 150)),
            'jamb_num_subjects': len(jamb_subjects) if jamb_subjects else 0
        }
        
        # Calculate JAMB score categories
        jamb_score = row.get('JAMB_Score', 0)
        if jamb_score >= 250:
            features['jamb_category'] = 4  # Excellent
        elif jamb_score >= 200:
            features['jamb_category'] = 3  # Good
        elif jamb_score >= 180:
            features['jamb_category'] = 2  # Fair
        elif jamb_score >= 150:
            features['jamb_category'] = 1  # Pass
        else:
            features['jamb_category'] = 0  # Below cutoff
        
        return features
    
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("🔄 Preprocessing data...")
        
        # Remove rows without admitted courses
        self.df = self.df[self.df['Admitted_Course'] != 'None'].copy()
        self.df = self.df.dropna(subset=['Admitted_Course'])
        
        print(f"📊 Records after filtering: {len(self.df)}")
        
        # Calculate WAEC features
        waec_features_list = []
        for _, row in self.df.iterrows():
            waec_features_list.append(self.calculate_waec_features(row))
        
        waec_df = pd.DataFrame(waec_features_list)
        
        # Calculate JAMB features
        jamb_features_list = []
        for _, row in self.df.iterrows():
            jamb_features_list.append(self.calculate_jamb_features(row))
        
        jamb_df = pd.DataFrame(jamb_features_list)
        
        # Combine all features
        self.feature_df = pd.concat([
            self.df[['Student_ID', 'Post_UTME_Score', 'Aggregate_Score', 'Course_Cutoff', 'Admitted_Course']].reset_index(drop=True),
            waec_df.reset_index(drop=True),
            jamb_df.reset_index(drop=True)
        ], axis=1)
        
        # Fill missing values
        self.feature_df['Post_UTME_Score'] = self.feature_df['Post_UTME_Score'].fillna(0)
        self.feature_df['Aggregate_Score'] = self.feature_df['Aggregate_Score'].fillna(0)
        self.feature_df['Course_Cutoff'] = self.feature_df['Course_Cutoff'].fillna(50)
        
        # Add derived features
        self.feature_df['aggregate_above_cutoff'] = (
            self.feature_df['Aggregate_Score'] >= self.feature_df['Course_Cutoff']
        ).astype(int)
        
        self.feature_df['score_margin'] = (
            self.feature_df['Aggregate_Score'] - self.feature_df['Course_Cutoff']
        )
        
        print("✅ Data preprocessing completed")
        print(f"📈 Feature columns: {list(self.feature_df.columns)}")
        
    def encode_features(self):
        """Encode categorical features"""
        print("🏷️ Encoding features...")
        
        # Encode target variable (courses)
        self.feature_df['course_encoded'] = self.course_encoder.fit_transform(
            self.feature_df['Admitted_Course']
        )
        
        # Prepare feature matrix
        feature_columns = [
            'waec_avg_grade', 'waec_best_grade', 'waec_worst_grade', 'waec_num_subjects',
            'waec_credits', 'waec_distinctions', 'has_mathematics', 'has_english_language',
            'has_physics', 'has_chemistry', 'has_biology', 'jamb_score', 'jamb_cutoff',
            'jamb_above_cutoff', 'jamb_num_subjects', 'jamb_category', 'Post_UTME_Score',
            'Aggregate_Score', 'Course_Cutoff', 'aggregate_above_cutoff', 'score_margin'
        ]
        
        self.X = self.feature_df[feature_columns].copy()
        self.y = self.feature_df['course_encoded'].copy()
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print("✅ Feature encoding completed")
        print(f"📊 Feature matrix shape: {self.X_scaled.shape}")
        print(f"🎯 Target classes: {len(np.unique(self.y))}")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("✂️ Splitting data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"📊 Training set: {self.X_train.shape[0]} samples")
        print(f"📊 Testing set: {self.X_test.shape[0]} samples")
        
    def train_model(self):
        """Train Random Forest model with hyperparameter tuning"""
        print("🤖 Training Random Forest model...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initial model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        print("🔍 Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
        
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"🎯 Accuracy: {accuracy:.4f}")
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        print(f"📈 Precision: {precision:.4f}")
        print(f"📈 Recall: {recall:.4f}")
        print(f"📈 F1-Score: {f1:.4f}")
        
        # Feature importance
        feature_names = [
            'waec_avg_grade', 'waec_best_grade', 'waec_worst_grade', 'waec_num_subjects',
            'waec_credits', 'waec_distinctions', 'has_mathematics', 'has_english_language',
            'has_physics', 'has_chemistry', 'has_biology', 'jamb_score', 'jamb_cutoff',
            'jamb_above_cutoff', 'jamb_num_subjects', 'jamb_category', 'Post_UTME_Score',
            'Aggregate_Score', 'Course_Cutoff', 'aggregate_above_cutoff', 'score_margin'
        ]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_scaled, self.y, cv=5)
        print(f"\n🔄 Cross-validation scores: {cv_scores}")
        print(f"🔄 Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
    
    def save_models(self):
        """Save trained model and encoders"""
        print("💾 Saving models...")
        
        # Save main model
        with open(os.path.join(self.model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save course encoder
        with open(os.path.join(self.model_dir, 'course_encoder.pkl'), 'wb') as f:
            pickle.dump(self.course_encoder, f)
        
        # Save scaler
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        feature_names = [
            'waec_avg_grade', 'waec_best_grade', 'waec_worst_grade', 'waec_num_subjects',
            'waec_credits', 'waec_distinctions', 'has_mathematics', 'has_english_language',
            'has_physics', 'has_chemistry', 'has_biology', 'jamb_score', 'jamb_cutoff',
            'jamb_above_cutoff', 'jamb_num_subjects', 'jamb_category', 'Post_UTME_Score',
            'Aggregate_Score', 'Course_Cutoff', 'aggregate_above_cutoff', 'score_margin'
        ]
        
        with open(os.path.join(self.model_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(feature_names, f)
        
        # Save grade mapping
        with open(os.path.join(self.model_dir, 'grade_mapping.pkl'), 'wb') as f:
            pickle.dump(self.grade_mapping, f)
        
        print("✅ All models saved successfully!")
    
    def run_complete_pipeline(self):
        """Execute the complete training pipeline"""
        print("🚀 Starting YABATECH Course Recommendation Model Training")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Load data
        if not self.load_data():
            return False
        
        # Load course requirements
        self.load_course_requirements()
        
        # Preprocess data
        self.preprocess_data()
        
        # Encode features
        self.encode_features()
        
        # Split data
        self.split_data()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Save models
        self.save_models()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        print(f"⏱️ Total training time: {duration}")
        print(f"📁 Models saved in: {self.model_dir}")
        print("=" * 60)
        
        return True

def main():
    """Main function to run the training pipeline"""
    trainer = YabatechModelTrainer()
    success = trainer.run_complete_pipeline()
    
    if success:
        print("\n✅ Ready to run the Flask application!")
        print("Run: python app.py")
    else:
        print("\n❌ Training failed. Please check the data and try again.")

if __name__ == "__main__":
    main()


def launch_flask_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return "✅ YABATECH Course Recommender Flask App is running!"

    # 👇 Ensure Render sees this
    app.run(host="0.0.0.0", port=10000)

if __name__ == "__main__":
    success = main()
    if success:
        launch_flask_app()
