// YABATECH Course Recommendation System - JavaScript

// Global variables
let chatbotOpen = false;
let currentStep = 0;
let userResponses = {};

// DOM elements
const chatbot = document.getElementById('chatbot');
const fab = document.querySelector('.fab');
const chatMessages = document.getElementById('chatMessages');
const recommendationForm = document.getElementById('recommendationForm');
const loadingSpinner = document.getElementById('loadingSpinner');
const recommendationsResult = document.getElementById('recommendationsResult');
const courseForm = document.getElementById('courseForm');

// Course database with detailed information
const courseDatabase = {
    'Computer Science': {
        minJambScore: 180,
        minCredits: 5,
        description: 'Learn programming, software development, and IT solutions. Perfect for tech enthusiasts.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Software Developer, System Analyst, IT Consultant',
        requirements: ['Mathematics', 'English Language', 'Physics'],
        cutoffMark: 180
    },
    'Mechanical Engineering': {
        minJambScore: 200,
        minCredits: 5,
        description: 'Design, manufacturing, and mechanical systems. Great for hands-on problem solvers.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Mechanical Engineer, Design Engineer, Production Manager',
        requirements: ['Mathematics', 'English Language', 'Physics', 'Chemistry'],
        cutoffMark: 200
    },
    'Electrical Engineering': {
        minJambScore: 190,
        minCredits: 5,
        description: 'Power systems, electronics, and telecommunications. Perfect for electrical enthusiasts.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Electrical Engineer, Power Systems Engineer, Electronics Specialist',
        requirements: ['Mathematics', 'English Language', 'Physics', 'Chemistry'],
        cutoffMark: 190
    },
    'Civil Engineering': {
        minJambScore: 185,
        minCredits: 5,
        description: 'Construction, infrastructure, and structural design. Build the future infrastructure.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Civil Engineer, Structural Engineer, Project Manager',
        requirements: ['Mathematics', 'English Language', 'Physics', 'Chemistry'],
        cutoffMark: 185
    },
    'Business Administration': {
        minJambScore: 160,
        minCredits: 5,
        description: 'Management, entrepreneurship, and business operations. Lead organizations to success.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Business Manager, Entrepreneur, Operations Manager',
        requirements: ['Mathematics', 'English Language', 'Economics'],
        cutoffMark: 160
    },
    'Accountancy': {
        minJambScore: 170,
        minCredits: 5,
        description: 'Financial accounting, auditing, and taxation. Master the language of business.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Accountant, Auditor, Financial Analyst',
        requirements: ['Mathematics', 'English Language', 'Economics'],
        cutoffMark: 170
    },
    'Mass Communication': {
        minJambScore: 150,
        minCredits: 5,
        description: 'Media, journalism, and communication. Shape public opinion and information.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Journalist, Media Producer, Public Relations Officer',
        requirements: ['English Language', 'Literature', 'Government'],
        cutoffMark: 150
    },
    'Architecture': {
        minJambScore: 210,
        minCredits: 6,
        description: 'Building design and construction planning. Create beautiful and functional spaces.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Architect, Urban Planner, Design Consultant',
        requirements: ['Mathematics', 'English Language', 'Physics', 'Fine Arts'],
        cutoffMark: 210
    },
    'Hospitality Management': {
        minJambScore: 140,
        minCredits: 5,
        description: 'Hotel, tourism, and event management. Excel in the service industry.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Hotel Manager, Event Planner, Tourism Officer',
        requirements: ['English Language', 'Mathematics', 'Economics'],
        cutoffMark: 140
    },
    'Marketing': {
        minJambScore: 155,
        minCredits: 5,
        description: 'Product promotion, sales, and customer relations. Drive business growth.',
        duration: '2 years (ND), 2 years (HND)',
        careerPaths: 'Marketing Manager, Sales Executive, Brand Manager',
        requirements: ['English Language', 'Mathematics', 'Economics'],
        cutoffMark: 155
    }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Add smooth scrolling to navigation links
    addSmoothScrolling();
    
    // Add form validation
    addFormValidation();
    
    // Add floating header on scroll
    addScrollEffects();
    
    // Initialize chatbot event listeners
    initializeChatbot();
}

// Smooth scrolling for navigation
function addSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerHeight = document.querySelector('.header').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Add form validation
function addFormValidation() {
    const inputs = document.querySelectorAll('input[required], select[required]');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearFieldError);
    });
}

function validateField(e) {
    const field = e.target;
    const value = field.value.trim();
    
    // Remove existing error styling
    field.classList.remove('error');
    
    // Validate based on field type
    if (!value && field.hasAttribute('required')) {
        showFieldError(field, 'This field is required');
        return false;
    }
    
    // Specific validations
    switch(field.id) {
        case 'age':
            if (value < 16 || value > 50) {
                showFieldError(field, 'Age must be between 16 and 50');
                return false;
            }
            break;
        case 'jamb_score':
            if (value < 0 || value > 400) {
                showFieldError(field, 'JAMB score must be between 0 and 400');
                return false;
            }
            break;
        case 'o_level_credits':
            if (value < 0 || value > 9) {
                showFieldError(field, 'O\'Level credits must be between 0 and 9');
                return false;
            }
            break;
    }
    
    return true;
}

function showFieldError(field, message) {
    field.classList.add('error');
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    errorDiv.style.color = '#ef4444';
    errorDiv.style.fontSize = '12px';
    errorDiv.style.marginTop = '5px';
    
    field.parentNode.appendChild(errorDiv);
}

function clearFieldError(e) {
    const field = e.target;
    field.classList.remove('error');
    const errorMessage = field.parentNode.querySelector('.error-message');
    if (errorMessage) {
        errorMessage.remove();
    }
}

// Scroll effects
function addScrollEffects() {
    let lastScrollTop = 0;
    const header = document.querySelector('.header');
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        // Add/remove scrolled class for styling
        if (scrollTop > 100) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
        
        lastScrollTop = scrollTop;
    });
}

// Initialize chatbot functionality
function initializeChatbot() {
    // FAB click event
    if (fab) {
        fab.addEventListener('click', toggleChatbot);
    }
    
    // Minimize button click event
    const minimizeBtn = document.querySelector('.minimize-btn');
    if (minimizeBtn) {
        minimizeBtn.addEventListener('click', toggleChatbot);
    }
    
    // Form submission
    if (courseForm) {
        courseForm.addEventListener('submit', handleFormSubmission);
    }
    
    // Form reset
    const resetBtn = document.querySelector('.reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetForm);
    }
}

// Toggle chatbot visibility
function toggleChatbot() {
    chatbotOpen = !chatbotOpen;
    
    if (chatbotOpen) {
        chatbot.style.display = 'flex';
        fab.style.display = 'none';
        
        // Add entrance animation
        chatbot.style.transform = 'translateY(20px)';
        chatbot.style.opacity = '0';
        
        setTimeout(() => {
            chatbot.style.transform = 'translateY(0)';
            chatbot.style.opacity = '1';
            chatbot.style.transition = 'all 0.3s ease';
        }, 10);
        
    } else {
        chatbot.style.transform = 'translateY(20px)';
        chatbot.style.opacity = '0';
        
        setTimeout(() => {
            chatbot.style.display = 'none';
            fab.style.display = 'flex';
            chatbot.style.transform = 'translateY(0)';
            chatbot.style.opacity = '1';
        }, 300);
    }
}

// Handle form submission
function handleFormSubmission(e) {
    e.preventDefault();
    
    // Validate all required fields
    const requiredFields = courseForm.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!validateField({ target: field })) {
            isValid = false;
        }
    });
    
    if (!isValid) {
        addBotMessage('Please fill in all required fields correctly before proceeding.');
        return;
    }
    
    // Collect form data
    const formData = new FormData(courseForm);
    const userData = {};
    
    for (let [key, value] of formData.entries()) {
        if (value !== '' && value !== 'None') {
            userData[key] = value;
        }
    }
    
    // Show loading state
    showLoadingState();
    
    // Add bot message about processing
    addBotMessage('Thank you for providing your information! Let me analyze your profile and find the best course recommendations for you...');
    
    // Process recommendations after a delay (simulate AI processing)
    setTimeout(() => {
        const recommendations = generateRecommendations(userData);
        displayRecommendations(recommendations, userData);
    }, 3000);
}

// Show loading state
function showLoadingState() {
    recommendationForm.style.display = 'none';
    loadingSpinner.style.display = 'block';
}

// Generate course recommendations based on user data
function generateRecommendations(userData) {
    const recommendations = [];
    const jambScore = parseInt(userData.jamb_score) || 0;
    const credits = parseInt(userData.o_level_credits) || 0;
    const preferences = [
        userData.course_preference_1,
        userData.course_preference_2,
        userData.course_preference_3
    ].filter(pref => pref && pref !== 'None');
    
    // Score each course based on eligibility and preferences
    Object.entries(courseDatabase).forEach(([courseName, courseInfo]) => {
        let score = 0;
        let eligible = true;
        let reasons = [];
        
        // Check basic eligibility
        if (jambScore >= courseInfo.minJambScore && credits >= courseInfo.minCredits) {
            score += 40; // Base eligibility score
            reasons.push('Meets admission requirements');
        } else {
            eligible = false;
            if (jambScore < courseInfo.minJambScore) {
                reasons.push(`JAMB score below requirement (${courseInfo.minJambScore})`);
            }
            if (credits < courseInfo.minCredits) {
                reasons.push(`Insufficient O'Level credits (need ${courseInfo.minCredits})`);
            }
        }
        
        // Preference bonus
        const prefIndex = preferences.indexOf(courseName);
        if (prefIndex !== -1) {
            score += (30 - prefIndex * 5); // First choice gets 30, second gets 25, third gets 20
            reasons.push(`Selected as preference #${prefIndex + 1}`);
        }
        
        // JAMB score bonus (higher score = better match)
        const scoreBonus = Math.min(20, (jambScore - courseInfo.minJambScore) / 5);
        score += scoreBonus;
        
        // Experience relevance (simplified)
        const experience = parseInt(userData.years_of_experience) || 0;
        if (experience > 0) {
            score += Math.min(10, experience * 2);
            reasons.push('Has relevant work experience');
        }
        
        // Add to recommendations if eligible or if specifically requested
        if (eligible || preferences.includes(courseName)) {
            recommendations.push({
                course: courseName,
                score: Math.round(score),
                eligible: eligible,
                info: courseInfo,
                reasons: reasons,
                matchPercentage: Math.min(100, Math.round(score))
            });
        }
    });
    
    // Sort by score (highest first) and return top 5
    return recommendations
        .sort((a, b) => b.score - a.score)
        .slice(0, 5);
}

// Display recommendations
function displayRecommendations(recommendations, userData) {
    loadingSpinner.style.display = 'none';
    recommendationsResult.style.display = 'block';
    
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';
    
    if (recommendations.length === 0) {
        recommendationsList.innerHTML = `
            <div class="no-recommendations">
                <p>I couldn't find any suitable course recommendations based on your current profile. Please consider:</p>
                <ul>
                    <li>Improving your JAMB score through retaking the exam</li>
                    <li>Obtaining additional O'Level credits</li>
                    <li>Exploring foundation programs</li>
                    <li>Contacting the admissions office for guidance</li>
                </ul>
            </div>
        `;
        addBotMessage('Based on your profile, I have some suggestions for improving your eligibility. Please check the recommendations above.');
        return;
    }
    
    recommendations.forEach((rec, index) => {
        const recElement = createRecommendationElement(rec, index + 1);
        recommendationsList.appendChild(recElement);
    });
    
    // Add bot message
    const topCourse = recommendations[0];
    addBotMessage(`Great news! I found ${recommendations.length} suitable course${recommendations.length > 1 ? 's' : ''} for you. Your top recommendation is ${topCourse.course} with a ${topCourse.matchPercentage}% match. Check out all the recommendations above!`);
    
    // Scroll to recommendations
    recommendationsResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create recommendation element
function createRecommendationElement(recommendation, rank) {
    const div = document.createElement('div');
    div.className = 'recommendation-item';
    
    const statusClass = recommendation.eligible ? 'eligible' : 'not-eligible';
    const statusText = recommendation.eligible ? 'Eligible' : 'Not Eligible';
    const statusColor = recommendation.eligible ? '#10b981' : '#ef4444';
    
    div.innerHTML = `
        <h4>
            #${rank} ${recommendation.course}
            <span class="match-score" style="background: ${statusColor}">${recommendation.matchPercentage}% Match</span>
        </h4>
        <p><strong>Status:</strong> <span style="color: ${statusColor}; font-weight: 600;">${statusText}</span></p>
        <p>${recommendation.info.description}</p>
        
        <div class="course-details">
            <div><strong>Duration:</strong> ${recommendation.info.duration}</div>
            <div><strong>Cut-off Mark:</strong> ${recommendation.info.cutoffMark}</div>
        </div>
        
        <p><strong>Career Paths:</strong> ${recommendation.info.careerPaths}</p>
        
        <div class="recommendation-reasons">
            <strong>Why this matches you:</strong>
            <ul>
                ${recommendation.reasons.map(reason => `<li>${reason}</li>`).join('')}
            </ul>
        </div>
    `;
    
    return div;
}

// Add bot message to chat
function addBotMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'bot-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <p>${message}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Reset form and recommendations
function resetForm() {
    courseForm.reset();
    resetRecommendations();
    addBotMessage('Form has been reset. You can now enter new information for course recommendations.');
}

function resetRecommendations() {
    recommendationForm.style.display = 'block';
    loadingSpinner.style.display = 'none';
    recommendationsResult.style.display = 'none';
    
    // Clear any field errors
    const errorFields = courseForm.querySelectorAll('.error');
    const errorMessages = courseForm.querySelectorAll('.error-message');
    
    errorFields.forEach(field => field.classList.remove('error'));
    errorMessages.forEach(msg => msg.remove());
}

// Scroll to recommender function (called from CTA button)
function scrollToRecommender() {
    if (!chatbotOpen) {
        toggleChatbot();
    }
    
    setTimeout(() => {
        recommendationForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

// Utility function to format numbers
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Add entrance animations for elements when they come into view
function addScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements that should animate
    const animateElements = document.querySelectorAll(
        '.course-card, .feature-item, .admission-card, .stat-item'
    );
    
    animateElements.forEach(el => {
        el.classList.add('animate-on-scroll');
        observer.observe(el);
    });
}

// Initialize animations after DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(addScrollAnimations, 500);
});

// Add CSS for animations
const animationStyles = `
<style>
.animate-on-scroll {
    opacity: 0;
    transform: translateY(30px);
    transition: all 0.6s ease;
}

.animate-in {
    opacity: 1;
    transform: translateY(0);
}

.error {
    border-color: #ef4444 !important;
    box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1) !important;
}

.header.scrolled {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
}

.no-recommendations {
    text-align: center;
    padding: 30px 20px;
    background: #fef3c7;
    border-radius: 10px;
    border: 1px solid #f59e0b;
}

.no-recommendations ul {
    text-align: left;
    margin-top: 15px;
    padding-left: 20px;
}

.no-recommendations li {
    margin-bottom: 8px;
    color: #92400e;
}

.recommendation-reasons {
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid #e2e8f0;
}

.recommendation-reasons ul {
    margin-top: 8px;
    padding-left: 20px;
}

.recommendation-reasons li {
    margin-bottom: 5px;
    color: #64748b;
    font-size: 13px;
}

@media (max-width: 768px) {
    .chatbot-container {
        animation: slideUp 0.3s ease;
    }
    
    @keyframes slideUp {
        from {
            transform: translateY(100%);
        }
        to {
            transform: translateY(0);
        }
    }
}
</style>
`;

// Add the animation styles to the document head
document.head.insertAdjacentHTML('beforeend', animationStyles);

// Export functions for global access
window.toggleChatbot = toggleChatbot;
window.scrollToRecommender = scrollToRecommender;
window.resetRecommendations = resetRecommendations;