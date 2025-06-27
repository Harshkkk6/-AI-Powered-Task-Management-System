import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

def analyze_dataset():
    # Load the data
    df = pd.read_csv('cleaned_jira_dataset.csv')
    
    print("\n=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print("\n=== Class Distribution ===")
    print(df['issue_type'].value_counts())
    print("\n=== Priority Distribution ===")
    print(df['priority'].value_counts())
    
    # Analyze text lengths
    df['text_length'] = df['clean_summary'].str.len()
    print("\n=== Text Length Statistics ===")
    print(df['text_length'].describe())
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Fill NaN values in clean_summary
    df['clean_summary'] = df['clean_summary'].fillna('')
    
    # Analyze text content
    print("\n=== Most Common Words ===")
    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['clean_summary'])
    feature_names = vectorizer.get_feature_names_out()
    print("Top 20 most common words:", feature_names.tolist())
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Class distribution plot
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='issue_type')
    plt.title('Issue Type Distribution')
    plt.xticks(rotation=45)
    
    # Priority distribution plot
    plt.subplot(2, 2, 2)
    sns.countplot(data=df, x='priority')
    plt.title('Priority Distribution')
    plt.xticks(rotation=45)
    
    # Text length distribution
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('Text Length Distribution')
    
    # Word cloud
    plt.subplot(2, 2, 4)
    text = ' '.join(df['clean_summary'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Task Descriptions')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png')
    print("\nAnalysis plots saved as 'data_analysis.png'")
    
    # Analyze class overlap
    print("\n=== Class Overlap Analysis ===")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean_summary'])
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate average TF-IDF scores per class
    class_means = {}
    for issue_type in df['issue_type'].unique():
        mask = df['issue_type'] == issue_type
        class_means[issue_type] = X[mask].mean(axis=0).A1
    
    # Find most distinctive words per class
    print("\nMost distinctive words per class:")
    for issue_type, means in class_means.items():
        top_indices = means.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"\n{issue_type}: {', '.join(top_words)}")

if __name__ == "__main__":
    analyze_dataset() 