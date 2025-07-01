# YABATECH Course Recommender System

A machine learning-powered web application that provides personalized course recommendations for Yaba College of Technology (YABATECH) students based on their academic profile, interests, and career goals.

## 🎯 Features

- **Personalized Recommendations**: Get course suggestions tailored to your academic background and career aspirations
- **Machine Learning Powered**: Uses trained ML models to provide accurate and relevant recommendations
- **User-Friendly Interface**: Clean, responsive web interface with real-time validation
- **Comprehensive Filtering**: Consider multiple factors including CGPA, interests, career goals, and financial capacity
- **Export Options**: Print or share your recommendations
- **Real-time Validation**: Instant feedback on form inputs with helpful error messages

## 🏗️ Project Structure

```
yabatech_course_recommender/
│
├── data/                          # Data files
│   ├── synthetic_dataset.csv      # Training dataset
│   └── course_requirements.json   # Course information and requirements
│
├── models/                        # Trained ML models
│   ├── model.pkl                  # Main recommendation model
│   ├── encoder.pkl                # Label encoder for categorical variables
│   └── scaler.pkl                 # Feature scaler for numerical variables
│
├── templates/                     # HTML templates
│   └── index.html                 # Main web interface
│
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css             # Stylesheet
│   └── js/
│       └── script.js             # Frontend JavaScript
│
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_preprocessor.py      # Data preprocessing utilities
│   └── recommendation_engine.py   # Recommendation logic
│
├── train_model.py                 # Model training script
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yabatech_course_recommender
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if models don't exist)
   ```bash
   python train_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser** and navigate to `http://localhost:5000`

## 📊 Data Requirements

### Input Features

The system uses the following student profile features for recommendations:

- **Age**: Student's age (16-50 years)
- **Gender**: Male/Female/Other
- **CGPA**: Cumulative Grade Point Average (0.0-4.0 scale)
- **Interest**: Primary area of interest
- **Career Goal**: Desired career path
- **Income**: Family income level (optional)
- **Location**: Preferred study location
- **Study Mode**: Full-time/Part-time preference

### Course Data

Course information includes:
- Course name and description
- Entry requirements
- Duration and mode of study
- Career prospects
- School/department information

## 🤖 Machine Learning Model

### Algorithm
The system uses a machine learning classifier (likely Random Forest or similar ensemble method) trained on historical student data and course outcomes.

### Features Engineering
- Categorical encoding for non-numerical features
- Feature scaling for numerical variables
- Feature selection based on importance

### Model Performance
The model is evaluated using standard classification metrics:
- Accuracy
- Precision and Recall
- F1-Score
- Cross-validation scores

## 🌐 Web Application

### Backend (Flask)
- **Framework**: Flask
- **API Endpoints**:
  - `GET /`: Main application page
  - `POST /recommend`: Get course recommendations
- **Data Processing**: Real-time feature preprocessing
- **Model Inference**: Load and use trained models

### Frontend
- **HTML5**: Semantic markup with accessibility features
- **CSS3**: Responsive design with modern styling
- **JavaScript**: Interactive form validation and AJAX requests
- **Progressive Enhancement**: Works without JavaScript (basic functionality)

## 📋 API Usage

### Recommendation Endpoint

**URL**: `POST /recommend`

**Request Body**:
```json
{
  "age": 20,
  "gender": "Male",
  "cgpa": 3.5,
  "interest": "Technology",
  "career_goal": "Software Developer",
  "income": 150000,
  "location": "Lagos",
  "study_mode": "Full-time"
}
```

**Response**:
```json
{
  "success": true,
  "recommendations": [
    {
      "course_name": "Computer Science",
      "school": "School of Technology",
      "confidence": 0.92,
      "duration": "2 years",
      "requirements": ["O'Level Mathematics", "O'Level English"],
      "description": "Comprehensive computer science program...",
      "career_prospects": "Software development, IT consulting..."
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:
```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
MODEL_PATH=models/
DATA_PATH=data/
```

### Model Parameters
Adjust model parameters in `train_model.py`:
- Training data split ratio
- Model hyperparameters
- Feature selection criteria
- Cross-validation folds

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Manual Testing
1. Test with various student profiles
2. Verify edge cases (extreme values)
3. Check form validation
4. Test error handling

## 📈 Performance Optimization

### Model Optimization
- Feature selection to reduce dimensionality
- Model quantization for faster inference
- Caching frequently requested recommendations

### Web Performance
- Minified CSS and JavaScript
- Compressed static assets
- Efficient database queries
- CDN for static files (if deployed)

## 🚀 Deployment

### Local Development
Already covered in Quick Start section.

### Production Deployment

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

#### Cloud Platforms
- **Heroku**: `git push heroku main`
- **AWS**: Use Elastic Beanstalk or EC2
- **Google Cloud**: Use App Engine or Compute Engine
- **Azure**: Use App Service

## 🔍 Troubleshooting

### Common Issues

1. **Model files not found**
   - Solution: Run `python train_model.py` to generate models

2. **Import errors**
   - Solution: Check virtual environment activation and dependencies

3. **Port already in use**
   - Solution: Change port in `app.py` or kill existing process

4. **Low recommendation accuracy**
   - Solution: Retrain model with more diverse data

### Debug Mode
Enable debug mode for development:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Write unit tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Developer**: [Your Name]
- **Institution**: Yaba College of Technology (YABATECH)
- **Contact**: [your.email@example.com]

## 🙏 Acknowledgments

- YABATECH for providing institutional support
- Open source community for libraries and tools
- Students who provided feedback and testing
- Machine learning community for algorithms and techniques

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/)

## 🔮 Future Enhancements

- [ ] Mobile app development
- [ ] Integration with YABATECH student portal
- [ ] Advanced recommendation algorithms (deep learning)
- [ ] Multi-language support
- [ ] Student feedback and rating system
- [ ] Course popularity trends
- [ ] Alumni success tracking
- [ ] Real-time course availability
- [ ] Scholarship recommendations
- [ ] Study group suggestions

## 📊 Version History

- **v1.0.0**: Initial release with basic recommendation functionality
- **v1.1.0**: Added web interface and improved model accuracy
- **v1.2.0**: Enhanced UI/UX and added export features

---

**Note**: This system is designed specifically for YABATECH courses and requirements. For other institutions, the course data and requirements would need to be updated accordingly.