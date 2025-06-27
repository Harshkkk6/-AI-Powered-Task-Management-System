import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="AI Task Management System",
    page_icon="📋",
    layout="wide"
)

# Load dataset for workload analysis
df = pd.read_csv('cleaned_jira_dataset.csv')

# Convert 'task_deadline' to pandas datetime objects, handling errors
df['task_deadline'] = pd.to_datetime(df['task_deadline'], errors='coerce')

# Load models and encoders
print('Loading models and encoders...')
task_bundle = joblib.load('task_classifier.pkl')
priority_bundle = joblib.load('priority_predictor.pkl')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load TF-IDF vectorizers
task_tfidf = joblib.load('tfidf_vectorizer.pkl')
priority_tfidf = joblib.load('priority_tfidf_vectorizer.pkl')

# Main app
def main():
    # Header
    st.title("🤖 AI-Powered Task Management System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['New Task', 'Live Dashboard', 'Smart Predictor', 'Alerts', 'Data Export'])

    if page == 'New Task':
        st.header('🆕 Submit New Task')
        with st.form(key='task_form'):
            task_summary = st.text_area('Task Summary', height=150)
            deadline = st.date_input('Deadline (Optional)', value=None)
            assignee = st.selectbox('Assignee (Optional)', options=[''] + list(df['task_assignee'].unique()))
            submit_button = st.form_submit_button(label='Submit Task')

        if submit_button:
            if task_summary:
                # Predict issue type
                issue_type, issue_confidence = predict_issue_type(task_summary)
                st.write(f'Predicted Issue Type: {issue_type} (Confidence: {issue_confidence.max():.2f})')

                # Predict priority
                priority, priority_confidence = predict_priority(task_summary)
                st.write(f'Predicted Priority: {priority} (Confidence: {priority_confidence.max():.2f})')

                # Recommend assignee if not chosen
                if not assignee:
                    recommended_assignee = recommend_least_loaded_user()
                    st.write(f'Recommended Assignee: {recommended_assignee} (Least Loaded)')
                else:
                    st.write(f'Assignee: {assignee}')

                # Display deadline if provided
                if deadline:
                    st.write(f'Deadline: {deadline}')

                # Find and display similar tasks
                similar_tasks, similarities = find_similar_tasks(task_summary)
                st.header('Similar Tasks')
                for i, (_, task) in enumerate(similar_tasks.iterrows()):
                    st.write(f'Task {i+1}: {task["clean_summary"]} (Similarity: {similarities[i]:.2f})')
                    st.write(f'Priority: {task["priority"]}, Assignee: {task["task_assignee"]}')

                # Append new task to new_tasks.csv
                new_task = pd.DataFrame({
                    'clean_summary': [task_summary],
                    'issue_type': [issue_type],
                    'priority': [priority],
                    'task_assignee': [assignee if assignee else recommended_assignee],
                    'task_deadline': [deadline]
                })
                new_task.to_csv('new_tasks.csv', mode='a', header=False, index=False)
                st.success('Task saved successfully!')
            else:
                st.error('Please enter a task summary.')

    elif page == 'Live Dashboard':
        st.header('📊 Live Workload Visualizer')
        filter_priority = st.multiselect('Filter by Priority', options=df['priority'].unique())
        filter_issue_type = st.multiselect('Filter by Issue Type', options=df['issue_type'].unique())

        filtered_df = df.copy()
        if filter_priority:
            filtered_df = filtered_df[filtered_df['priority'].isin(filter_priority)]
        if filter_issue_type:
            filtered_df = filtered_df[filtered_df['issue_type'].isin(filter_issue_type)]

        workload = filtered_df['task_assignee'].value_counts()
        st.bar_chart(workload)

    elif page == 'Smart Predictor':
        st.header('🧠 Smart NLP Prediction Tool')
        task_summary_input = st.text_area('Paste Task Summary Here', height=150)
        if task_summary_input:
            # Predict issue type
            issue_type, issue_confidence = predict_issue_type(task_summary_input)
            st.write(f'Predicted Issue Type: {issue_type} (Confidence: {issue_confidence.max():.2f})')

            # Predict priority
            priority, priority_confidence = predict_priority(task_summary_input)
            st.write(f'Predicted Priority: {priority} (Confidence: {priority_confidence.max():.2f})')

    elif page == 'Alerts':
        st.header('📎 Deadline Alerts Panel')
        
        # Get current date and date 3 days from now
        current_date = pd.Timestamp.now().normalize()
        future_date = current_date + pd.Timedelta(days=3)
        
        # Filter tasks
        mask_due_soon = (df['task_deadline'] >= current_date) & (df['task_deadline'] <= future_date)
        mask_overdue = df['task_deadline'] < current_date
        
        due_soon_tasks = df[mask_due_soon].sort_values('task_deadline')
        overdue_tasks = df[mask_overdue].sort_values('task_deadline')
        
        # Display due soon tasks
        st.subheader('Tasks Due in Next 3 Days')
        if len(due_soon_tasks) > 0:
            for _, task in due_soon_tasks.iterrows():
                deadline_str = task['task_deadline'].strftime('%Y-%m-%d')
                st.markdown(f"""
                    <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        <strong>Task:</strong> {task['clean_summary']}<br>
                        <strong>Due:</strong> {deadline_str}<br>
                        <strong>Priority:</strong> {task['priority']}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info('No tasks due in the next 3 days.')
        
        # Display overdue tasks
        st.subheader('Overdue Tasks')
        if len(overdue_tasks) > 0:
            for _, task in overdue_tasks.iterrows():
                deadline_str = task['task_deadline'].strftime('%Y-%m-%d')
                st.markdown(f"""
                    <div style='background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        <strong>Task:</strong> {task['clean_summary']}<br>
                        <strong>Due:</strong> {deadline_str}<br>
                        <strong>Priority:</strong> {task['priority']}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info('No overdue tasks.')

    elif page == 'Data Export':
        st.header('📤 Export & Save')
        if st.button('Download Task List as CSV'):
            df.to_csv('task_list.csv', index=False)
            st.success('Task list downloaded successfully!')

    # Footer
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses machine learning models to predict task types and priorities based on task descriptions.
    The models were trained on historical Jira data and can help in:
    - Automatically categorizing new tasks
    - Determining task priorities
    - Monitoring team workload
    """)

# Function to preprocess text
def preprocess_text(text):
    return text.lower().strip()

# Function to get BERT embeddings
def get_bert_embeddings(text):
    return bert_model.encode([text], show_progress_bar=False)[0]

# Function to predict issue type
def predict_issue_type(text):
    processed_text = preprocess_text(text)
    bert_features = get_bert_embeddings(processed_text)
    tfidf_features = task_tfidf.transform([processed_text]).toarray()
    features = np.hstack([bert_features, tfidf_features])
    prediction = task_bundle['model'].predict([features])[0]
    confidence = task_bundle['model'].predict_proba([features])[0]
    return task_bundle['label_encoder'].inverse_transform([prediction])[0], confidence

# Function to predict priority
def predict_priority(text):
    processed_text = preprocess_text(text)
    bert_features = get_bert_embeddings(processed_text)
    tfidf_features = priority_tfidf.transform([processed_text]).toarray()
    features = np.hstack([bert_features, tfidf_features])
    prediction = priority_bundle['model'].predict([features])[0]
    confidence = priority_bundle['model'].predict_proba([features])[0]
    return priority_bundle['label_encoder'].inverse_transform([prediction])[0], confidence

# Function to recommend least loaded user
def recommend_least_loaded_user():
    workload = df['task_assignee'].value_counts()
    return workload.index[0]

# Function to find similar tasks
def find_similar_tasks(text, top_n=3):
    processed_text = preprocess_text(text)
    bert_features = get_bert_embeddings(processed_text)
    tfidf_features = task_tfidf.transform([processed_text]).toarray()
    features = np.hstack([bert_features, tfidf_features])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(features, task_tfidf.transform(df['clean_summary']).toarray())
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    
    similar_tasks = df.iloc[top_indices]
    return similar_tasks, similarities[0][top_indices]

if __name__ == "__main__":
    main() 