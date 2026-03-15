%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

# --- STYLES ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADING & PROCESSING ---
@st.cache_data
def load_and_preprocess():
    books = pd.read_csv('books.csv', on_bad_lines='warn')
    articles = pd.read_csv('medium_data.csv', on_bad_lines='warn')
    
    def clean(text):
        return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

    # Feature Engineering: Combine multiple fields for deeper context
    books['metadata'] = (books['title'].fillna('') + " " + 
                         books['authors'].fillna('') + " " + 
                         books.get('categories', pd.Series(['']*len(books))).fillna('')).apply(clean)
    
    articles['metadata'] = (articles['title'].fillna('') + " " + 
                            articles.get('subtitle', pd.Series(['']*len(articles))).fillna('')).apply(clean)
    return books, articles

@st.cache_resource
def compute_tfidf(metadata_series):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(metadata_series)
    return vectorizer, matrix

# --- LOGIC ---
def get_recommendations(query, df, vectorizer, matrix, top_n=5):
    query_vec = vectorizer.transform([query.lower()])
    similarity = cosine_similarity(query_vec, matrix).flatten()
    indices = np.argsort(similarity)[::-1][:top_n]
    
    results = df.iloc[indices].copy()
    results['score'] = similarity[indices]
    return results

def get_learning_path(topic):
    paths = {
        "ai": ["Python for DS", "Linear Algebra & Calculus", "Supervised Learning", "Neural Networks", "LLM Fine-tuning"],
        "machine learning": ["Statistics & Probability", "Data Cleaning", "Regression & Classification", "Ensemble Methods", "Model Deployment"],
        "data science": ["SQL & Databases", "Exploratory Data Analysis", "Statistical Modeling", "Data Visualization", "Business Analytics"],
        "physics": ["Classical Mechanics", "Electromagnetism", "Thermodynamics", "Quantum Mechanics", "Special Relativity"],
        "robotics": ["C++ Programming", "Kinematics", "Sensor Fusion", "Path Planning", "Embedded Systems"],
        "programming": ["Logic", "Memory Management", "Design Patterns", "System Architecture", "Cloud Infrastructure"]
    }
    return paths.get(topic.lower(), ["Fundamentals", "Foundational Tools", "Advanced Concepts", "Practical Application", "Specialized Project"])

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠 How it Works")
    st.info("""
    **1. NLP Preprocessing**  
    Cleaning and tokenizing user input.
    
    **2. TF-IDF Vectorization**  
    Converting text into mathematical vectors based on term importance.
    
    **3. Cosine Similarity**  
    Measuring the angle between user query and resource metadata.
    """)
    st.markdown("--- ")
    st.write("**Try Example Topics:**")
    if st.button("🤖 AI"): st.session_state.topic = "AI"
    if st.button("🧪 Physics"): st.session_state.topic = "Physics"
    if st.button("📊 Data Science"): st.session_state.topic = "Data Science"

# --- MAIN UI ---
st.title("🤖 AI Learning Assistant")
st.write("Enter your interest to generate a professional learning roadmap.")

if 'topic' not in st.session_state: st.session_state.topic = ""

user_input = st.text_input("What would you like to master?", value=st.session_state.topic, placeholder="e.g. Computer Vision")

if st.button("Generate Learning Plan"):
    if not user_input:
        st.warning("⚠️ Please enter a topic to continue.")
    else:
        with st.spinner(f"Analyzing resources for '{user_input}'..."):
            books_df, articles_df = load_and_preprocess()
            
            # Load/Compute Vectors
            b_vec, b_mat = compute_tfidf(books_df['metadata'])
            a_vec, a_mat = compute_tfidf(articles_df['metadata'])
            
            rec_books = get_recommendations(user_input, books_df, b_vec, b_mat)
            rec_articles = get_recommendations(user_input, articles_df, a_vec, a_mat)
            
            st.success("Roadmap Generated!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📚 Top Recommended Books")
                for _, row in rec_books.iterrows():
                    with st.expander(f"{row['title']}"):
                        st.write(f"**Author:** {row.get('authors', 'Unknown')}")
                        st.metric("Match Score", f"{row['score']*100:.1f}%")
                
                st.subheader("📰 Insightful Articles")
                for _, row in rec_articles.iterrows():
                    st.markdown(f"- {row['title']} `({row['score']*100:.1f}% Match)`")

            with col2:
                st.subheader("🚀 Learning Path")
                path = get_learning_path(user_input)
                for i, step in enumerate(path, 1):
                    st.info(f"**Step {i}**: {step}")
                
                st.subheader("🎥 Video Content")
                yt_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(user_input + ' tutorial')}"
                st.link_button("Search YouTube Tutorials", yt_url, use_container_width=True)
"
