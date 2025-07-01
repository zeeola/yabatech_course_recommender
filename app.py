"""
YABATECH Course Recommendation System - Flask Web Application
============================================================

This Flask application provides a web interface for the course recommendation system
with hybrid ML-based and rule-based filtering.

Author: YABATECH AI Team
Date: 2025
"""

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import json
import pickle
import os
import re
from datetime import datetime
import logging
from utils.recommendation_engine import RecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'yabatech_course_recommender_2025'

class YabatechRecommenderApp:
    """
    Main application class for YABATECH Course Recommendation System
    """
    
    def __init__(self):
        self.model_dir = 'models/'
        self.data_dir = 'data/'
        self.model = None
        self.course_encoder = None
        self.scaler = None
        self.feature_names = None
        self.grade_mapping = None
        self.course_requirements = {}
        self.recommendation_engine = None
        
        # Load models and data
        self.load_models()
        self.load_course_requirements()
        
        # Initialize recommendation engine
        self.recommendation_engine = RecommendationEngine(
            model=self.model,
            course_encoder=self.course_encoder,
            scaler=self.scaler,
            grade_mapping=self.grade_mapping,
            course_requirements=self.course_requirements
        )
    
    def load_models(self):
        """Load trained models and encoders"""
        try:
            logger.info("Loading trained models...")
            
            # Load main model
            with open(os.path.join(self.model_dir, 'model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            
            # Load course encoder
            with open(os.path.join(self.model_dir, 'course_encoder.pkl'), 'rb') as f:
                self.course_encoder = pickle.load(f)
            
            # Load scaler
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            with open(os.path.join(self.model_dir, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load grade mapping
            with open(os.path.join(self.model_dir, 'grade_mapping.pkl'), 'rb') as f:
                self.grade_mapping = pickle.load(f)
            
            logger.info("✅ All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def load_course_requirements(self):
        """Load course requirements for rule-based filtering"""
        try:
            requirements_path = os.path.join(self.data_dir, 'course_requirements.json')
            with open(requirements_path, 'r') as f:
                self.course_requirements = json.load(f)
            logger.info(f"✅ Course requirements loaded: {len(self.course_requirements)} courses")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading course requirements: {e}")
            self.course_requirements = {}
            return False

# Initialize the application
yabatech_app = YabatechRecommenderApp()

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': yabatech_app.model is not None,
        'course_requirements_loaded': len(yabatech_app.course_requirements) > 0
    }
    return jsonify(status)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Main recommendation endpoint
    Accepts student data and returns course recommendations
    """
    try:
        data = request.get_json()
        logger.info(f"Received recommendation request: {data}")
        
        # Validate required fields
        required_fields = ['waec_subjects', 'waec_grades', 'jamb_subjects', 'jamb_score']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        # Get recommendations using the recommendation engine
        recommendations = yabatech_app.recommendation_engine.get_recommendations(data)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Chatbot endpoint for natural language interaction
    """
    try:
        data = request.get_json()
        message = data.get('message', '').lower().strip()
        
        logger.info(f"Chat message received: {message}")
        
        # Simple chatbot responses
        response = process_chat_message(message)
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}")
        return jsonify({
            'error': f'Chat error: {str(e)}',
            'success': False
        }), 500

def process_chat_message(message):
    """
    Process natural language chat messages
    """
    # Greeting patterns
    if any(word in message for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return {
            'text': "Hello! Welcome to YABATECH Course Recommendation System. I'm here to help you find the perfect course based on your WAEC and JAMB results. How can I assist you today?",
            'type': 'greeting'
        }
    
    # Help patterns
    elif any(word in message for word in ['help', 'how', 'what can you do', 'assist']):
        return {
            'text': """I can help you with:
            
🎓 **Course Recommendations**: Provide your WAEC subjects/grades and JAMB scores
📋 **Course Requirements**: Information about specific course prerequisites  
📊 **Eligibility Check**: Verify if you meet course requirements
💡 **Academic Guidance**: General advice about course selection

To get started, you can:
1. Use the recommendation form on the right
2. Ask me about specific courses
3. Inquire about admission requirements

What would you like to know?""",
            'type': 'help'
        }
    
    # Course information patterns
    elif any(word in message for word in ['course', 'program', 'study', 'diploma']):
        courses_list = list(yabatech_app.course_requirements.keys()) if yabatech_app.course_requirements else [
            "ND Computer Science", "ND Electrical Engineering", "ND Mechanical Engineering",
            "ND Civil Engineering", "ND Business Administration", "ND Accounting",
            "ND Mass Communication", "ND Architecture"
        ]
        
        return {
            'text': f"""YABATECH offers various National Diploma (ND) programs including:

🔧 **Engineering**: Civil, Electrical, Mechanical, Chemical
💻 **Technology**: Computer Science, Information Technology
💼 **Business**: Business Administration, Accounting, Banking & Finance
🎨 **Arts & Design**: Mass Communication, Fine Arts, Architecture
🏥 **Sciences**: Science Laboratory Technology, Statistics

Total available programs: {len(courses_list)}

Would you like information about a specific course or get personalized recommendations?""",
            'type': 'course_info'
        }
    
    # Requirements patterns
    elif any(word in message for word in ['requirement', 'prerequisite', 'need', 'qualify']):
        return {
            'text': """📋 **General YABATECH Requirements**:

**WAEC/NECO/GCE Requirements**:
- Minimum of 5 credits including English Language and Mathematics
- Credits must be obtained at not more than 2 sittings
- Relevant subjects for your chosen course

**JAMB Requirements**:
- Minimum JAMB score varies by course (usually 150-180)
- Must choose relevant subjects for your course
- JAMB score contributes to aggregate calculation

**Post-UTME**:
- Participate in YABATECH Post-UTME screening
- Score contributes to final aggregate

Use the recommendation form to check specific requirements for your preferred courses!""",
            'type': 'requirements'
        }
    
    # Score/grade patterns
    elif any(word in message for word in ['score', 'grade', 'result', 'jamb', 'waec']):
        return {
            'text': """📊 **Understanding Your Scores**:

**WAEC Grades** (Best to Worst):
- A1, B2, B3 = Excellent (Distinction)
- C4, C5, C6 = Credit (Good)
- D7, E8 = Pass (Fair)
- F9 = Fail

**JAMB Scores**:
- 250+ = Excellent chances
- 200-249 = Very good chances  
- 180-199 = Good chances
- 150-179 = Fair chances
- Below 150 = May need to retake

**Tips**: Focus on getting credits (C6 and above) in relevant subjects for your chosen course!""",
            'type': 'score_info'
        }
    
    # Default response
    else:
        return {
            'text': """I understand you're looking for guidance! Here's how I can help:

🎯 **Get Course Recommendations**: Fill out the form with your WAEC and JAMB details
❓ **Ask Questions**: About courses, requirements, or admission process
📋 **Check Eligibility**: For specific programs you're interested in

Try asking:
- "What courses can I study?"
- "What are the requirements?"
- "Tell me about engineering courses"
- Or use the recommendation form →

What specific information do you need?""",
            'type': 'default'
        }

@app.route('/courses')
def get_courses():
    """Get list of available courses"""
    try:
        courses = list(yabatech_app.course_requirements.keys()) if yabatech_app.course_requirements else []
        return jsonify({
            'success': True,
            'courses': courses,
            'total': len(courses)
        })
    except Exception as e:
        logger.error(f"Error getting courses: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/course/<course_name>')
def get_course_info(course_name):
    """Get detailed information about a specific course"""
    try:
        course_info = yabatech_app.course_requirements.get(course_name)
        if not course_info:
            return jsonify({
                'error': 'Course not found',
                'success': False
            }), 404
        
        return jsonify({
            'success': True,
            'course': course_name,
            'requirements': course_info
        })
    except Exception as e:
        logger.error(f"Error getting course info: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

    
if __name__ == "__main__":
    # Train models on first run if they don't exist
    if yabatech_app.model is None:
        logger.warning("Models not found – running training script.")
        os.system("python train_model.py")
        yabatech_app.load_models()

    port = int(os.environ.get("PORT", 10000))   # Render injects PORT
    app.run(host="0.0.0.0", port=port, debug=False)
