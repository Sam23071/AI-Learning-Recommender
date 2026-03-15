import streamlit as st
import pandas as pd
import numpy as np
import re
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM STYLES ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_and_preprocess():

    books = pd.read_csv("books.csv", on_bad_lines="skip")
    articles = pd.read_csv("medium_data.csv", on_bad_lines="skip")

    def clean(text):
        return re.sub(r"[^\w\s]", "", str(text).lower()).strip()

    # Combine metadata fields
    books["metadata"] = (
        books["title"].fillna("") + " " +
        books.get("authors", "").fillna("") + " " +
        books.get("categories", pd.Series([""]*len(books))).fillna("")
    ).apply(clean)

    articles["metadata"] = (
        articles["title"].fillna("") + " " +
        articles.get("subtitle", pd.Series([""]*len(articles))).fillna("")
    ).apply(clean)

    return books, articles


# ---------------- TFIDF ----------------
@st.cache_resource
def compute_tfidf(metadata):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2)
    )

    matrix = vectorizer.fit_transform(metadata)

    return vectorizer, matrix


# ---------------- RECOMMENDER ----------------
def get_recommendations(query, df, vectorizer, matrix, top_n=5):

    query_vec = vectorizer.transform([query.lower()])

    similarity = cosine_similarity(query_vec, matrix).flatten()

    indices = np.argsort(similarity)[::-1][:top_n]

    results = df.iloc[indices].copy()

    results["score"] = similarity[indices]

    return results


# ---------------- LEARNING PATH ----------------
def get_learning_path(topic):

    paths = {

        "ai": [
            "Python for AI",
            "Linear Algebra & Calculus",
            "Machine Learning Algorithms",
            "Deep Learning & Neural Networks",
            "Build AI Projects"
        ],

        "machine learning": [
            "Statistics & Probability",
            "Data Preprocessing",
            "Regression & Classification",
            "Model Evaluation",
            "Deploy ML Models"
        ],

        "data science": [
            "Python & Pandas",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Visualization",
            "Business Analytics Project"
        ],

        "physics": [
            "Classical Mechanics",
            "Electromagnetism",
            "Thermodynamics",
            "Quantum Mechanics",
            "Advanced Physics Problems"
        ],

        "programming": [
            "Logic Building",
            "Data Structures",
            "Algorithms",
            "System Design",
            "Build Real Projects"
        ]
    }

    return paths.get(
        topic.lower(),
        [
            "Fundamentals",
            "Intermediate Concepts",
            "Advanced Topics",
            "Practical Application",
            "Capstone Project"
        ]
    )


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("⚙️ How It Works")

    st.info("""
    **1️⃣ NLP Preprocessing**  
    Cleaning and preparing text data.

    **2️⃣ TF-IDF Vectorization**  
    Converts text into numerical vectors.

    **3️⃣ Cosine Similarity**  
    Finds the most relevant learning resources.
    """)

    st.markdown("---")

    st.write("### Try Example Topics")

    if st.button("🤖 AI"):
        st.session_state.topic = "AI"

    if st.button("🧪 Physics"):
        st.session_state.topic = "Physics"

    if st.button("📊 Data Science"):
        st.session_state.topic = "Data Science"


# ---------------- MAIN UI ----------------
st.title("🤖 AI Learning Assistant")

st.caption(
    "NLP-powered system that recommends books, articles and tutorials "
    "based on your learning interest."
)

if "topic" not in st.session_state:
    st.session_state.topic = ""

user_input = st.text_input(
    "What would you like to learn?",
    value=st.session_state.topic,
    placeholder="Example: Machine Learning"
)


# ---------------- RUN ENGINE ----------------
if st.button("Generate Learning Plan"):

    if not user_input:

        st.warning("⚠️ Please enter a topic first.")

    else:

        with st.spinner("Generating recommendations..."):

            books_df, articles_df = load_and_preprocess()

            book_vec, book_mat = compute_tfidf(books_df["metadata"])
            art_vec, art_mat = compute_tfidf(articles_df["metadata"])

            rec_books = get_recommendations(
                user_input,
                books_df,
                book_vec,
                book_mat
            )

            rec_articles = get_recommendations(
                user_input,
                articles_df,
                art_vec,
                art_mat
            )

        st.success("Learning roadmap generated!")

        col1, col2 = st.columns([2,1])

        # -------- BOOKS --------
        with col1:

            st.subheader("📚 Recommended Books")

            for _, row in rec_books.iterrows():

                with st.expander(row["title"]):

                    st.write(
                        f"**Author:** {row.get('authors','Unknown')}"
                    )

                    st.metric(
                        "Relevance Score",
                        f"{row['score']*100:.1f}%"
                    )

            st.subheader("📰 Recommended Articles")

            for _, row in rec_articles.iterrows():

                st.markdown(
                    f"- {row['title']} "
                    f"`({row['score']*100:.1f}% match)`"
                )

        # -------- RIGHT PANEL --------
        with col2:

            st.subheader("🚀 Learning Path")

            path = get_learning_path(user_input)

            for i, step in enumerate(path,1):

                st.info(f"Step {i}: {step}")

            st.subheader("🎥 Video Tutorials")

            yt_url = (
                "https://www.youtube.com/results?search_query=" +
                urllib.parse.quote(user_input + " tutorial")
            )

            st.link_button(
                "Search YouTube Tutorials",
                yt_url,
                use_container_width=True
            )
