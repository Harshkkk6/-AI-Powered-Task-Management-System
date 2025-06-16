# AI-Powered Task Management System

This project implements an AI-powered task management system that analyzes Jira task data using natural language processing and machine learning techniques.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── task_management_analysis.py
├── task_managment_project.ipynb
└── jira_dataset.csv
```

## Features

- Data loading and exploration
- Exploratory Data Analysis (EDA)
- Text preprocessing using NLTK
- Feature engineering with TF-IDF and Word2Vec
- Machine learning models (Naive Bayes and SVM)
- Model evaluation and visualization

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Powered-Task-Management-System
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook task_managment_project.ipynb
```

2. Run the cells sequentially to:
   - Load and explore the data
   - Perform EDA
   - Clean and preprocess the text
   - Train and evaluate machine learning models

## Project Components

### Week 1: Data Analysis and Preprocessing
- Data loading and initial exploration
- EDA with visualizations
- Data cleaning and standardization
- Text preprocessing with NLTK

### Week 2: Machine Learning
- Feature engineering with TF-IDF and Word2Vec
- Model training (Naive Bayes and SVM)
- Model evaluation and visualization
- Performance metrics analysis

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib & seaborn: Data visualization
- nltk: Natural language processing
- scikit-learn: Machine learning
- gensim: Word embeddings
- jupyter: Interactive development

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
