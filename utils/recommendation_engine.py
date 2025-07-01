"""
YABATECH Recommendation Engine
=============================

Hybrid recommendation system combining ML-based and rule-based approaches
for course recommendations.

Author: YABATECH AI Team
Date: 2025
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Hybrid recommendation engine combining ML predictions with rule-based filtering
    """
    
    def __init__(self, model, course_encoder, scaler, grade_mapping, course_requirements):
        self.model = model
        self.course_encoder = course_encoder
        self.scaler = scaler
        self.grade_mapping = grade_mapping
        self.course_requirements = course_requirements
        
        # Feature names (must match training order)
        self.feature_names = [
            'waec_avg_grade', 'waec_best_grade', 'waec_worst_grade', 'waec_num_subjects',
            'waec_credits', 'waec_distinctions', 'has_mathematics', 'has_english_language',
            'has_physics', 'has_chemistry', 'has_biology', 'jamb_score', 'jamb_cutoff',
            'jamb_above_cutoff', 'jamb_num_subjects', 'jamb_category', 'Post_UTME_Score',
            'Aggregate_Score', 'Course_Cutoff', 'aggregate_above_cutoff', 'score_margin'
        ]
    
    def parse_waec_data(self, waec_subjects: str, waec_grades: str) -> Dict[str, Any]:
        """Parse WAEC subjects and grades"""
        try:
            # Handle string inputs
            if isinstance(waec_subjects, str):
                subjects = [s.strip() for s in waec_subjects.split(',')]
            else:
                subjects = waec_subjects
            
            if isinstance(waec_grades, str):
                grades = [g.strip() for g in waec_grades.split(',')]
            else:
                grades = waec_grades
            
            # Convert grades to numerical values
            numerical_grades = []
            for grade in grades:
                if grade in self.grade_mapping:
                    numerical_grades.append(self.grade_mapping[grade])
                else:
                    # Try to extract grade from strings like "B2", "C4", etc.
                    grade_match = re.search(r'[A-F][1-9]', grade.upper())
                    if grade_match:
                        clean_grade = grade_match.group()
                        numerical_grades.append(self.grade_mapping.get(clean_grade, 5))
                    else:
                        numerical_grades.append(5)  # Default to C5
            
            # Calculate features
            features = {}
            if numerical_grades:
                features['waec_avg_grade'] = np.mean(numerical_grades)
                features['waec_best_grade'] = min(numerical_grades)
                features['waec_worst_grade'] = max(numerical_grades)
                features['waec_num_subjects'] = len(numerical_grades)
                features['waec_credits'] = sum(1 for g in numerical_grades if g <= 6)
                features['waec_distinctions'] = sum(1 for g in numerical_grades if g <= 3)
            else:
                features.update({
                    'waec_avg_grade': 7, 'waec_best_grade': 9, 'waec_worst_grade': 9,
                    'waec_num_subjects': 0, 'waec_credits': 0, 'waec_distinctions': 0
                })
            
            # Subject-specific features
            subjects_lower = [s.lower() for s in subjects]
            features['has_mathematics'] = int(any('math' in s for s in subjects_lower))
            features['has_english_language'] = int(any('english' in s for s in subjects_lower))
            features['has_physics'] = int(any('physics' in s for s in subjects_lower))
            features['has_chemistry'] = int(any('chemistry' in s for s in subjects_lower))
            features['has_biology'] = int(any('biology' in s for s in subjects_lower))
            
            return features, subjects, grades
            
        except Exception as e:
            logger.error(f"Error parsing WAEC data: {e}")
            return {}, [], []
    
    def parse_jamb_data(self, jamb_subjects: str, jamb_score: int, jamb_cutoff: int = 150) -> Dict[str, Any]:
        """Parse JAMB subjects and score"""
        try:
            # Handle subjects
            if isinstance(jamb_subjects, str):
                subjects = [s.strip() for s in jamb_subjects.split(',')]
            else:
                subjects = jamb_subjects
            
            # Calculate features
            features = {
                'jamb_score': int(jamb_score) if jamb_score else 0,
                'jamb_cutoff': int(jamb_cutoff),
                'jamb_above_cutoff': int(jamb_score >= jamb_cutoff) if jamb_score else 0,
                'jamb_num_subjects': len(subjects) if subjects else 0
            }
            
            # JAMB score categories
            score = int(jamb_score) if jamb_score else 0
            if score >= 250:
                features['jamb_category'] = 4
            elif score >= 200:
                features['jamb_category'] = 3
            elif score >= 180:
                features['jamb_category'] = 2
            elif score >= 150:
                features['jamb_category'] = 1
            else:
                features['jamb_category'] = 0
            
            return features, subjects
            
        except Exception as e:
            logger.error(f"Error parsing JAMB data: {e}")
            return {}, []
    
    def calculate_aggregate_score(self, jamb_score: int, post_utme_score: int = 0) -> float:
        """Calculate aggregate score (simplified YABATECH formula)"""
        try:
            jamb_score = int(jamb_score) if jamb_score else 0
            post_utme_score = int(post_utme_score) if post_utme_score else 0
            
            # Simplified aggregate calculation
            # Real YABATECH formula may be different
            jamb_contrib = (jamb_score / 400) * 50  # 50% weight
            post_utme_contrib = (post_utme_score / 100) * 50  # 50% weight
            
            aggregate = jamb_contrib + post_utme_contrib
            return round(aggregate, 2)
            
        except Exception as e:
            logger.error(f"Error calculating aggregate: {e}")
            return 0.0
    
    def create_feature_vector(self, student_data: Dict[str, Any]) -> np.ndarray:
        """Create feature vector for ML prediction"""
        try:
            # Parse WAEC data
            waec_features, waec_subjects, waec_grades = self.parse_waec_data(
                student_data.get('waec_subjects', ''),
                student_data.get('waec_grades', '')
            )
            
            # Parse JAMB data
            jamb_features, jamb_subjects = self.parse_jamb_data(
                student_data.get('jamb_subjects', ''),
                student_data.get('jamb_score', 0),
                student_data.get('jamb_cutoff', 150)
            )
            
            # Calculate aggregate score
            post_utme_score = student_data.get('post_utme_score', 0)
            aggregate_score = self.calculate_aggregate_score(
                student_data.get('jamb_score', 0),
                post_utme_score
            )
            
            # Default course cutoff
            course_cutoff = student_data.get('course_cutoff', 50)
            
            # Combine all features
            features = {}
            features.update(waec_features)
            features.update(jamb_features)
            features.update({
                'Post_UTME_Score': float(post_utme_score),
                'Aggregate_Score': aggregate_score,
                'Course_Cutoff': float(course_cutoff),
                'aggregate_above_cutoff': int(aggregate_score >= course_cutoff),
                'score_margin': aggregate_score - course_cutoff
            })
            
            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def apply_rule_based_filtering(self, student_data: Dict[str, Any], ml_predictions: List[str]) -> List[Dict[str, Any]]:
        """Apply rule-based filtering to ML predictions"""
        try:
            filtered_recommendations = []
            
            # Parse student data
            waec_features, waec_subjects, waec_grades = self.parse_waec_data(
                student_data.get('waec_subjects', ''),
                student_data.get('waec_grades', '')
            )
            jamb_features, jamb_subjects = self.parse_jamb_data(
                student_data.get('jamb_subjects', ''),
                student_data.get('jamb_score', 0)
            )
            
            for course in ml_predictions:
                # Get course requirements
                course_req = self.course_requirements.get(course, {})
                
                # Initialize eligibility
                eligible = True
                eligibility_reasons = []
                
                # Check JAMB score requirement
                min_jamb = course_req.get('min_jamb_score', 150)
                student_jamb = student_data.get('jamb_score', 0)
                if student_jamb < min_jamb:
                    eligible = False
                    eligibility_reasons.append(f"JAMB score {student_jamb} below minimum {min_jamb}")
                
                # Check required subjects
                required_subjects = course_req.get('required_subjects', [])
                student_subjects_lower = [s.lower() for s in waec_subjects + jamb_subjects]
                
                missing_subjects = []
                for req_subject in required_subjects:
                    if not any(req_subject.lower() in s for s in student_subjects_lower):
                        missing_subjects.append(req_subject)
                
                if missing_subjects:
                    eligible = False
                    eligibility_reasons.append(f"Missing subjects: {', '.join(missing_subjects)}")
                
                # Check minimum credits
                min_credits = course_req.get('min_credits', 5)
                student_credits = waec_features.get('waec_credits', 0)
                if student_credits < min_credits:
                    eligible = False
                    eligibility_reasons.append(f"Only {student_credits} credits, need {min_credits}")
                
                # Calculate recommendation score
                score = self.calculate_recommendation_score(student_data, course, course_req)
                
                filtered_recommendations.append({
                    'course': course,
                    'eligible': eligible,
                    'score': score,
                    'eligibility_reasons': eligibility_reasons,
                    'requirements': course_req
                })
            
            # Sort by score (descending)
            filtered_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error in rule-based filtering: {e}")
            return []
    
    def calculate_recommendation_score(self, student_data: Dict[str, Any], course: str, course_req: Dict[str, Any]) -> float:
        """Calculate recommendation score for a course"""
        try:
            score = 0.0
            
            # JAMB score component (0-40 points)
            jamb_score = student_data.get('jamb_score', 0)
            min_jamb = course_req.get('min_jamb_score', 150)
            if jamb_score >= min_jamb:
                jamb_component = min(40, (jamb_score - min_jamb) / 10)
                score += jamb_component
            
            # WAEC grades component (0-30 points)
            waec_features, _, _ = self.parse_waec_data(
                student_data.get('waec_subjects', ''),
                student_data.get('waec_grades', '')
            )
            avg_grade = waec_features.get('waec_avg_grade', 7)
            waec_component = max(0, 30 - (avg_grade - 1) * 5)
            score += waec_component
            
            # Subject match component (0-20 points)
            required_subjects = course_req.get('required_subjects', [])
            waec_subjects = student_data.get('waec_subjects', '').split(',')
            jamb_subjects = student_data.get('jamb_subjects', '').split(',')
            all_subjects = waec_subjects + jamb_subjects
            
            matched_subjects = 0
            for req_subject in required_subjects:
                if any(req_subject.lower() in s.lower() for s in all_subjects):
                    matched_subjects += 1
            
            if required_subjects:
                subject_component = (matched_subjects / len(required_subjects)) * 20
                score += subject_component
            
            # Post-UTME component (0-10 points)
            post_utme = student_data.get('post_utme_score', 0)
            if post_utme > 0:
                post_utme_component = min(10, post_utme / 10)
                score += post_utme_component
            
            return round(score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating recommendation score: {e}")
            return 0.0
    
    def get_recommendations(self, student_data: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
        """Get hybrid recommendations for a student"""
        try:
            logger.info(f"Getting recommendations for student data: {student_data}")
            
            # Create feature vector
            feature_vector = self.create_feature_vector(student_data)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get ML predictions
            predictions_proba = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Get top courses from ML model
            top_indices = np.argsort(predictions_proba)[-top_n:][::-1]
            ml_courses = [self.course_encoder.inverse_transform([idx])[0] for idx in top_indices]
            
            logger.info(f"ML predictions: {ml_courses}")
            
            # Apply rule-based filtering
            recommendations = self.apply_rule_based_filtering(student_data, ml_courses)
            
            # Add ML probability scores
            for i, rec in enumerate(recommendations):
                course_idx = np.where(self.course_encoder.classes_ == rec['course'])[0]
                if len(course_idx) > 0:
                    rec['ml_probability'] = float(predictions_proba[course_idx[0]])
                else:
                    rec['ml_probability'] = 0.0
            
            # Generate explanations
            for rec in recommendations:
                rec['explanation'] = self.generate_explanation(student_data, rec)
            
            # Limit to top_n
            final_recommendations = recommendations[:top_n]
            
            logger.info(f"Final recommendations: {len(final_recommendations)}")
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def generate_explanation(self, student_data: Dict[str, Any], recommendation: Dict[str, Any]) -> str:
        """Generate explanation for recommendation"""
        try:
            course = recommendation['course']
            score = recommendation['score']
            eligible = recommendation['eligible']
            
            explanation = f"**{course}**\n\n"
            
            if eligible:
                explanation += f"✅ **Eligible** (Score: {score}/100)\n\n"
                
                # Positive factors
                jamb_score = student_data.get('jamb_score', 0)
                if jamb_score >= 200:
                    explanation += f"🎯 Strong JAMB score: {jamb_score}\n"
                elif jamb_score >= 150:
                    explanation += f"✓ Good JAMB score: {jamb_score}\n"
                
                waec_features, _, _ = self.parse_waec_data(
                    student_data.get('waec_subjects', ''),
                    student_data.get('waec_grades', '')
                )
                
                credits = waec_features.get('waec_credits', 0)
                if credits >= 7:
                    explanation += f"📚 Excellent WAEC credits: {credits}\n"
                elif credits >= 5:
                    explanation += f"✓ Good WAEC credits: {credits}\n"
                
            else:
                explanation += f"❌ **Not Eligible** (Score: {score}/100)\n\n"
                explanation += "**Issues:**\n"
                for reason in recommendation['eligibility_reasons']:
                    explanation += f"• {reason}\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Recommendation for {recommendation.get('course', 'Unknown Course')}"