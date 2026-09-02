
import streamlit as st
import base64
import io
import wave
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import numpy as np
import pickle
import faiss
import joblib

from sentence_transformers import SentenceTransformer
from google import genai
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

def get_movie_poster(title):

    import requests
    import os

    api_key = (
        st.secrets.get("TMDB_API_KEY", None)
        if hasattr(st, "secrets")
        else None
    ) or os.getenv("TMDB_API_KEY")

    if not api_key:
        return None

    try:

        clean_title = title.split("(")[0].strip()

        url = "https://api.themoviedb.org/3/search/movie"

        params = {
            "api_key": api_key,
            "query": clean_title
        }

        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        if response.status_code != 200:
            return None

        data = response.json()

        results = data.get("results", [])

        if not results:
            return None

        poster_path = results[0].get(
            "poster_path"
        )

        if not poster_path:
            return None

        return (
            "https://image.tmdb.org/t/p/w500"
            + poster_path
        )

    except Exception:
        return None



# ============================================================
# PAGE CONFIGURATION
# ============================================================

# Selected movie from recommendation cards
if "selected_recommendation_movie" not in st.session_state:
    st.session_state.selected_recommendation_movie = None

st.set_page_config(
    page_title="Movie AI Intelligence",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    opacity: 0.75;
    margin-bottom: 30px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 16px;
    padding: 18px;
    background: rgba(128,128,128,0.06);
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

div[data-testid="stMetric"] label {
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    font-weight: 800;
}

.movie-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 14px;
    margin: 8px 0 18px 0;
    background: rgba(128,128,128,0.05);
    box-shadow: 0 5px 18px rgba(0,0,0,0.07);
    transition: all 0.2s ease;
}

.movie-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 9px 24px rgba(0,0,0,0.12);
}

.movie-title {
    font-size: 17px;
    font-weight: 750;
    line-height: 1.35;
    margin-top: 8px;
}

.movie-genres {
    font-size: 13px;
    opacity: 0.72;
    margin-top: 6px;
    margin-bottom: 12px;
}

.movie-stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}

.movie-stat {
    padding: 10px;
    border-radius: 10px;
    background: rgba(128,128,128,0.08);
    text-align: center;
}

.stat-label {
    font-size: 11px;
    opacity: 0.7;
}

.stat-value {
    font-size: 17px;
    font-weight: 750;
}

.chat-movie-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 18px;
    padding: 18px;
    margin: 10px 0 20px 0;
    background: rgba(128,128,128,0.06);
    box-shadow: 0 5px 18px rgba(0,0,0,0.08);
}

.chat-memory {
    border-radius: 12px;
    padding: 10px 14px;
    margin: 10px 0;
    background: rgba(128,128,128,0.08);
    font-size: 14px;
}

.section-header {
    font-size: 27px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 15px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    width: 100%;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)





# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource(show_spinner="Loading Movie AI models...")
def load_project():
    required_files = [
        "movie_data.pkl",
        "movie_rating_model.pkl",
        "feature_columns.pkl",
        "movie_faiss.index",
        "movie_documents.pkl",
    ]

    missing_files = [
        file_name
        for file_name in required_files
        if not (BASE_DIR / file_name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing project files: " + ", ".join(missing_files)
        )

    movie_data = pd.read_pickle(
        BASE_DIR / "movie_data.pkl"
    )

    model = joblib.load(
        BASE_DIR / "movie_rating_model.pkl"
    )

    feature_columns = joblib.load(
        BASE_DIR / "feature_columns.pkl"
    )

    index = faiss.read_index(
        str(BASE_DIR / "movie_faiss.index")
    )

    with open(
        BASE_DIR / "movie_documents.pkl",
        "rb"
    ) as f:
        documents = pickle.load(f)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return (
        movie_data,
        model,
        feature_columns,
        index,
        documents,
        embedding_model
    )


(
    movie_data,
    rf_model,
    feature_columns,
    index,
    documents,
    embedding_model
) = load_project()


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def load_gemini():

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY", "")
    )

    if not api_key:

        return None

    return genai.Client(
        api_key=api_key
    )


client = load_gemini()


# ============================================================
# SESSION MEMORY
# ============================================================

if "previous_interaction_id" not in st.session_state:

    st.session_state.previous_interaction_id = None


if "last_movie_title" not in st.session_state:

    st.session_state.last_movie_title = None


# ============================================================
# RAG FUNCTION
# ============================================================

def retrieve_movies(
    query,
    k=5
):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        results.append({
            "score": float(score),
            "document": documents[idx]
        })

    return results


# ============================================================
# GEMINI CHATBOT
# ============================================================

def movie_chatbot(user_question):

    # ========================================================
    # INITIALIZE MEMORY
    # ========================================================

    if "last_movie_title" not in st.session_state:
        st.session_state.last_movie_title = ""

    if "previous_interaction_id" not in st.session_state:
        st.session_state.previous_interaction_id = None

    # ========================================================
    # CLEAN QUESTION
    # ========================================================

    user_question = str(
        user_question
    ).strip()

    if not user_question:
        return "Please ask me something about a movie."

    # ========================================================
    # MOVIE MEMORY
    # ========================================================

    last_movie = (
        st.session_state.last_movie_title
    )

    # ========================================================
    # FOLLOW-UP DETECTION
    # ========================================================

    follow_up_words = [
        "it",
        "this",
        "this movie",
        "that",
        "that movie",
        "its",
        "it's",
        "the movie",

        # Rating questions
        "its rating",
        "what is its rating",
        "how good is it",
        "how is its rating",

        # Genre / story questions
        "its genre",
        "its genres",
        "what are its genres",
        "its story",
        "its plot",
        "tell me more",

        # Recommendation questions
        "would you recommend",
        "would you recommend it",
        "recommend it",
        "should i watch it",
        "is it worth watching",
        "movies like it",
        "similar to it",

        # Information questions
        "who directed it",
        "who acted in it",
        "how many ratings",
        "how many people rated it",
        "more about it",
        "what about it",
        "why",
        "why is it",
        "why should i watch it"
    ]

    question_lower = user_question.lower()

    is_follow_up = any(
        word in question_lower
        for word in follow_up_words
    )

    # ========================================================
    # SEARCH QUERY
    # ========================================================

    if is_follow_up and last_movie:

        search_query = (
            f"{user_question} "
            f"about {last_movie}"
        )

    else:

        search_query = user_question

    # ========================================================
    # RAG
    # ========================================================

    retrieved = retrieve_movies(
        search_query,
        k=5
    )

    context = "\n\n".join(
        item["document"]
        for item in retrieved
    )

    # ========================================================
    # UPDATE MOVIE MEMORY FROM RAG
    # ========================================================

    if retrieved:

        first_doc = retrieved[0]["document"]

        if first_doc:

            lines = first_doc.split("\n")

            for line in lines:

                if "Title:" in line:

                    detected_movie = (
                        line.split(
                            "Title:",
                            1
                        )[1]
                        .strip()
                    )

                    if detected_movie:

                        st.session_state.last_movie_title = (
                            detected_movie
                        )

                        last_movie = detected_movie

                        break

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are a friendly Movie AI Assistant.

You help users understand:
- movies
- genres
- actual ratings
- ML predicted ratings
- rating differences
- number of ratings
- movie recommendations

CURRENT MOVIE:
{last_movie}

RETRIEVED MOVIE INFORMATION:
{context}

USER QUESTION:
{user_question}

IMPORTANT RULES:

1. Use the retrieved movie information for
   movie-specific facts.

2. Never invent ratings or statistics.

3. Actual Rating comes from the dataset.

4. Predicted Rating comes from our ML model.

5. Number of Ratings comes from the dataset.

6. If the user says:
   "it", "this movie", "that movie",
   "its", "the movie", "why", or "would
   you recommend it", refer to CURRENT MOVIE.

7. If the user asks a follow-up question,
   do NOT ask them for the movie name again.

8. If the user asks whether they should watch
   the movie, give a simple recommendation
   based on the available rating information.

9. Explain answers in a friendly and simple way.

10. Do not mention RAG, embeddings, FAISS,
    prompts, context or internal system details
    unless the user specifically asks.

11. If information is unavailable in the dataset,
    clearly say that it is unavailable.

12. Keep normal answers concise but useful.
"""

    # ========================================================
    # GEMINI
    # ========================================================

    try:

        if client is None:

            raise Exception(
                "Gemini client is unavailable"
            )

        if (
            st.session_state.previous_interaction_id
            is None
        ):

            interaction = client.interactions.create(
                model="gemini-3.6-flash",
                input=prompt
            )

        else:

            interaction = client.interactions.create(
                model="gemini-3.6-flash",
                previous_interaction_id=(
                    st.session_state
                    .previous_interaction_id
                ),
                input=prompt
            )

        st.session_state.previous_interaction_id = (
            interaction.id
        )

        return interaction.output_text

    # ========================================================
    # LOCAL RAG FALLBACK
    # ========================================================

    except Exception:

        fallback = []

        fallback.append(
            "🎬 **Movie AI Assistant**"
        )

        if last_movie:

            fallback.append(
                f"### 🎥 {last_movie}"
            )

        if retrieved:

            best_doc = retrieved[0]["document"]

            fallback.append(
                "### 🔎 Movie Information"
            )

            fallback.append(
                best_doc
            )

            # ------------------------------------------------
            # FOLLOW-UP FRIENDLY RESPONSE
            # ------------------------------------------------

            if any(
                word in question_lower
                for word in [
                    "rating",
                    "rate",
                    "score"
                ]
            ):

                fallback.append(
                    "⭐ The rating information "
                    "is shown above from the movie dataset."
                )

            elif "genre" in question_lower:

                fallback.append(
                    "🎭 The movie genres are shown "
                    "above from the dataset."
                )

            elif (
                "recommend" in question_lower
                or "watch" in question_lower
            ):

                fallback.append(
                    "🍿 Based on the available rating "
                    "information, this movie may be "
                    "worth considering."
                )

            else:

                fallback.append(
                    "💡 Ask me about its rating, "
                    "genres, recommendations, or "
                    "other available movie information."
                )

        else:

            fallback.append(
                "❌ I couldn't find matching movie "
                "information in the database."
            )

        fallback.append(
            "ℹ️ Gemini is currently unavailable, "
            "so I'm using the local movie database."
        )

        return "\n\n".join(
            fallback
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Movie AI Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Movie Recommendation • Rating Prediction • RAG • Gemini AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬 Navigation")

pages = [
    "📊 Dashboard",
    "🔎 Movie Intelligence",
    "⭐ Rating Predictions",
    "🎯 Recommendations",
    "🤖 AI Chatbot"
]

if "page_navigation" not in st.session_state:
    st.session_state.page_navigation = "📊 Dashboard"

page = st.sidebar.radio(
    "Select Page",
    pages,
    index=pages.index(
        st.session_state.page_navigation
    )
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.header(
        "📊 Movie Dataset Overview"
    )


    # ========================================================
    # PROFESSIONAL KPI CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:30px;">🎬</div>
                <div style="font-size:14px; opacity:0.7;">
                    Total Movies
                </div>
                <div style="font-size:28px; font-weight:800;">
                    {len(movie_data):,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:30px;">⭐</div>
                <div style="font-size:14px; opacity:0.7;">
                    Average Rating
                </div>
                <div style="font-size:28px; font-weight:800;">
                    {movie_data['average_rating'].mean():.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:30px;">👥</div>
                <div style="font-size:14px; opacity:0.7;">
                    Total Ratings
                </div>
                <div style="font-size:28px; font-weight:800;">
                    {int(movie_data['rating_count'].sum()):,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:30px;">🤖</div>
                <div style="font-size:14px; opacity:0.7;">
                    Avg Predicted
                </div>
                <div style="font-size:28px; font-weight:800;">
                    {movie_data['predicted_rating'].mean():.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)


    st.subheader(
        "⭐ Rating Distribution"
    )

    rating_distribution = (
        movie_data["average_rating"]
        .round(1)
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        rating_distribution
    )


    st.subheader(
        "🎭 Genre Distribution"
    )

    genre_counts = {}

    for genres in movie_data["genres"].fillna(""):

        for genre in genres.split("|"):

            if genre:

                genre_counts[genre] = (
                    genre_counts.get(
                        genre,
                        0
                    ) + 1
                )


    genre_df = (
        pd.DataFrame(
            list(
                genre_counts.items()
            ),
            columns=[
                "Genre",
                "Movies"
            ]
        )
        .sort_values(
            "Movies",
            ascending=False
        )
        .head(15)
    )


    st.bar_chart(
        genre_df.set_index("Genre")
    )


    # ========================================================
    # TOP RATED MOVIES
    # ========================================================

    st.subheader("🏆 Top Rated Movies")

    top_movies = (
        movie_data[
            movie_data["rating_count"] >= 100
        ]
        .sort_values(
            "average_rating",
            ascending=False
        )
        [
            [
                "title",
                "genres",
                "average_rating",
                "rating_count",
                "predicted_rating"
            ]
        ]
        .head(10)
        .copy()
    )

    top_movies = top_movies.rename(
        columns={
            "title": "🎬 Movie",
            "genres": "🎭 Genres",
            "average_rating": "⭐ Actual Rating",
            "rating_count": "👥 Ratings",
            "predicted_rating": "🤖 Predicted"
        }
    )

    # ========================================================
    # TOP RATED MOVIE CARDS
    # ========================================================

    st.markdown(
        '<div class="section-header">🍿 Top Rated Movies</div>',
        unsafe_allow_html=True
    )

    # Create Top Rated Movies directly from movie_data
    top_movies = (
        movie_data[
            movie_data["rating_count"] >= 100
        ]
        .sort_values(
            "average_rating",
            ascending=False
        )
        .head(10)
        .reset_index(drop=True)
    )

    top_card_cols = st.columns(5)

    for i in range(len(top_movies)):

        movie = top_movies.iloc[i]

        with top_card_cols[i % 5]:

            movie_title = str(
                movie["title"]
            )

            movie_genres = str(
                movie["genres"]
            )

            actual_rating = float(
                movie["average_rating"]
            )

            rating_count = int(
                movie["rating_count"]
            )

            poster_url = get_movie_poster(
                movie_title
            )

            if poster_url:

                st.image(
                    poster_url,
                    use_container_width=True
                )

            st.markdown(
                f"**#{i + 1} 🎬 {movie_title}**"
            )

            st.caption(
                f"🎭 {movie_genres}"
            )

            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    "⭐ Rating",
                    f"{actual_rating:.2f}"
                )

            with col_b:

                st.metric(
                    "👥 Ratings",
                    f"{rating_count:,}"
                )

            st.divider()


    # ========================================================
    # MOST POPULAR MOVIES
    # ========================================================

    st.subheader("🔥 Most Popular Movies")

    popular_movies = (
        movie_data
        .sort_values(
            "rating_count",
            ascending=False
        )
        .head(10)
        .reset_index(drop=True)
    )

    popular_card_cols = st.columns(5)

    for i in range(len(popular_movies)):

        movie = popular_movies.iloc[i]

        with popular_card_cols[i % 5]:

            movie_title = str(
                movie["title"]
            )

            movie_genres = str(
                movie["genres"]
            )

            actual_rating = float(
                movie["average_rating"]
            )

            predicted_rating = float(
                movie["predicted_rating"]
            )

            rating_count = int(
                movie["rating_count"]
            )

            poster_url = get_movie_poster(
                movie_title
            )

            # Poster
            if poster_url:

                st.image(
                    poster_url,
                    use_container_width=True
                )

            # Title
            st.markdown(
                f"**#{i + 1} 🎬 {movie_title}**"
            )

            # Genres
            st.caption(
                f"🎭 {movie_genres}"
            )

            # Rating
            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    "⭐ Rating",
                    f"{actual_rating:.2f}"
                )

            with col_b:

                st.metric(
                    "👥 Ratings",
                    f"{rating_count:,}"
                )

            st.metric(
                "🤖 Predicted",
                f"{predicted_rating:.2f}"
            )

            st.divider()


    # ========================================================
    # RATING VS POPULARITY
    # ========================================================

    st.subheader("📈 Rating vs Popularity")

    chart_data = (
        movie_data[
            movie_data["rating_count"] > 50
        ][
            [
                "average_rating",
                "rating_count"
            ]
        ]
        .copy()
    )

    chart_data["rating_count"] = np.log1p(
        chart_data["rating_count"]
    )

    chart_data = chart_data.rename(
        columns={
            "average_rating": "⭐ Average Rating",
            "rating_count": "👥 Rating Count (Log)"
        }
    )

    st.scatter_chart(
        chart_data,
        x="⭐ Average Rating",
        y="👥 Rating Count (Log)",
        use_container_width=True
    )


# ============================================================
# MOVIE INTELLIGENCE
# ============================================================

elif page == "🔎 Movie Intelligence":

    st.header("🔎 Specific Movie Intelligence")

    selected_from_card = st.session_state.get(
        "selected_recommendation_movie",
        ""
    )

    if selected_from_card:
        movie_name = selected_from_card

        st.info(
            f"🎬 Selected from Recommendations: **{movie_name}**"
        )

        # Clear the selection so normal search works
        st.session_state.selected_recommendation_movie = ""

    else:
        movie_name = st.text_input(
            "🔎 Enter movie name",
            placeholder="Example: Toy Story"
        )

    if movie_name:

        # ----------------------------------------------------
        # FIND MOVIE
        # ----------------------------------------------------

        result = movie_data[
            movie_data["title"]
            .astype(str)
            .str.strip()
            == str(movie_name).strip()
        ]

        if result.empty:
            result = movie_data[
                movie_data["title"]
                .astype(str)
                .str.contains(
                    str(movie_name),
                    case=False,
                    na=False,
                    regex=False
                )
            ]

        if result.empty:

            st.error(
                f"❌ Movie not found: {movie_name}"
            )

        else:

            movie = result.iloc[0]

            title = str(movie["title"])
            genres = str(movie["genres"])

            actual_rating = float(
                movie["average_rating"]
            )

            predicted_rating = float(
                movie["predicted_rating"]
            )

            rating_count = int(
                movie["rating_count"]
            )

            rating_difference = float(
                movie["rating_difference"]
            )

            # ------------------------------------------------
            # POSTER
            # ------------------------------------------------

            poster_url = get_movie_poster(title)

            # ------------------------------------------------
            # PROFESSIONAL CARD
            # ------------------------------------------------

            st.markdown(
                '<div class="movie-intel-card">',
                unsafe_allow_html=True
            )

            poster_col, info_col = st.columns(
                [1, 2]
            )

            with poster_col:

                if poster_url:

                    st.image(
                        poster_url,
                        use_container_width=True
                    )

                else:

                    st.markdown(
                        "<h1 style='text-align:center;'>🎬</h1>",
                        unsafe_allow_html=True
                    )

            with info_col:

                st.markdown(
                    f"""
                    <div class="movie-intel-title">
                        🎬 {title}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="movie-intel-genres">
                        🎭 {genres.replace("|", " • ")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "⭐ Actual Rating",
                        f"{actual_rating:.2f}/5"
                    )

                with col2:
                    st.metric(
                        "🤖 Predicted Rating",
                        f"{predicted_rating:.2f}/5"
                    )

                col3, col4 = st.columns(2)

                with col3:
                    st.metric(
                        "👥 Rating Count",
                        f"{rating_count:,}"
                    )

                with col4:
                    st.metric(
                        "📊 Difference",
                        f"{rating_difference:+.2f}"
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # GENRES
            # ------------------------------------------------

            st.subheader("🎭 Movie Genres")

            genre_list = [
                g.strip()
                for g in genres.split("|")
                if g.strip()
            ]

            genre_cols = st.columns(
                min(len(genre_list), 4)
                if genre_list
                else 1
            )

            for i, genre in enumerate(genre_list):

                with genre_cols[
                    i % len(genre_cols)
                ]:

                    st.info(
                        f"🎬 {genre}"
                    )

            # ------------------------------------------------
            # RATING ANALYSIS
            # ------------------------------------------------

            st.subheader("📊 Rating Analysis")

            analysis_col1, analysis_col2 = st.columns(2)

            with analysis_col1:

                st.metric(
                    "⭐ Dataset Rating",
                    f"{actual_rating:.2f}"
                )

            with analysis_col2:

                st.metric(
                    "🤖 ML Prediction",
                    f"{predicted_rating:.2f}",
                    delta=f"{rating_difference:+.2f}"
                )

            # ------------------------------------------------
            # INSIGHT
            # ------------------------------------------------

            if predicted_rating > actual_rating:

                insight = (
                    "The ML model predicts a slightly higher "
                    "rating than the current dataset average."
                )

            elif predicted_rating < actual_rating:

                insight = (
                    "The ML model predicts a slightly lower "
                    "rating than the current dataset average."
                )

            else:

                insight = (
                    "The ML prediction is very close to the "
                    "current dataset average."
                )

            st.markdown(
                f"""
                <div class="movie-intel-description">
                    <b>💡 AI Rating Insight</b><br><br>
                    {insight}<br><br>
                    This movie has received
                    <b>{rating_count:,}</b> ratings.
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # BACK BUTTON
            # ------------------------------------------------

            if st.button(
                "🎯 Back to Recommendations",
                use_container_width=True
            ):

                st.session_state.page_navigation = (
                    "🎯 Recommendations"
                )

                st.session_state.selected_recommendation_movie = ""

                st.rerun()


# ============================================================
# RATING PREDICTIONS
# ============================================================

elif page == "⭐ Rating Predictions":

    st.header(
        "⭐ Rating Prediction for All Movies"
    )


    st.write(
        "The ML model has generated a predicted "
        "rating for every movie."
    )


    col1, col2 = st.columns(2)


    with col1:

        min_rating = st.slider(
            "Minimum predicted rating",
            0.5,
            5.0,
            3.5,
            0.1
        )


    with col2:

        min_count = st.number_input(
            "Minimum number of ratings",
            min_value=0,
            value=100
        )


    filtered = movie_data[
        (
            movie_data["predicted_rating"]
            >= min_rating
        )
        &
        (
            movie_data["rating_count"]
            >= min_count
        )
    ].sort_values(
        "predicted_rating",
        ascending=False
    )


    display_df = filtered[
        [
            "title",
            "genres",
            "average_rating",
            "rating_count",
            "predicted_rating",
            "rating_difference"
        ]
    ].head(100)


    st.dataframe(
        display_df,
        use_container_width=True
    )


    csv = filtered.to_csv(
        index=False
    )


    st.download_button(
        "⬇️ Download Predictions CSV",
        csv,
        "movie_predictions.csv",
        "text/csv"
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🎯 Recommendations":

    st.header("🎯 Movie Recommendations")

    st.write(
        "Find movies similar to a movie you already like "
        "using genre similarity, ratings and ML predictions."
    )

    movie_name = st.text_input(
        "🔎 Enter a movie you like",
        placeholder="Example: Forrest Gump"
    )

    number_of_recommendations = st.slider(
        "🍿 Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    if movie_name:

        result = movie_data[
            movie_data["title"].str.contains(
                movie_name,
                case=False,
                na=False,
                regex=False
            )
        ]

        if result.empty:

            st.error(
                "❌ Movie not found. Try another movie name."
            )

        else:

            selected = result.iloc[0]

            st.success(
                f"🎬 Selected Movie: {selected['title']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "⭐ Actual Rating",
                    f"{selected['average_rating']:.2f}"
                )

            with col2:
                st.metric(
                    "🤖 Predicted Rating",
                    f"{selected['predicted_rating']:.2f}"
                )

            with col3:
                st.metric(
                    "👥 Rating Count",
                    f"{int(selected['rating_count']):,}"
                )

            st.write(
                f"🎭 **Genres:** {selected['genres']}"
            )

            st.divider()

            selected_genres = set(
                str(selected["genres"]).split("|")
            )

            candidates = movie_data[
                movie_data["title"] != selected["title"]
            ].copy()

            def genre_similarity(genres):

                movie_genres = set(
                    str(genres).split("|")
                )

                return len(
                    selected_genres & movie_genres
                )

            candidates["genre_match"] = (
                candidates["genres"]
                .apply(genre_similarity)
            )

            candidates["recommendation_score"] = (
                candidates["genre_match"] * 3
                + candidates["average_rating"] * 2
                + candidates["predicted_rating"] * 3
                + np.log1p(
                    candidates["rating_count"]
                ) * 0.1
            )

            recommendations = (
                candidates[
                    candidates["genre_match"] > 0
                ]
                .sort_values(
                    "recommendation_score",
                    ascending=False
                )
                .head(number_of_recommendations)
            )

            st.subheader(
                "🍿 Recommended Movies"
            )

            if recommendations.empty:

                st.warning(
                    "No similar movies were found."
                )

            else:

                for rank, (_, movie) in enumerate(
                    recommendations.iterrows(),
                    start=1
                ):

                    poster_url = get_movie_poster(
                        movie["title"]
                    )

                    common_genres = (
                        selected_genres
                        &
                        set(
                            str(movie["genres"]).split("|")
                        )
                    )

                    genre_text = ", ".join(
                        sorted(common_genres)
                    )

                    # ----------------------------------------
                    # MOVIE CARD
                    # ----------------------------------------

                    with st.container(border=True):

                        poster_col, info_col = (
                            st.columns([1, 3])
                        )

                        with poster_col:

                            if poster_url:

                                st.image(
                                    poster_url,
                                    use_container_width=True
                                )

                            else:

                                st.markdown(
                                    """
                                    <div style="
                                        height:220px;
                                        display:flex;
                                        align-items:center;
                                        justify-content:center;
                                        border-radius:12px;
                                        background:#222;
                                        font-size:50px;
                                    ">
                                    🎬
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        with info_col:

                            st.markdown(
                                f"## #{rank} {movie['title']}"
                            )

                            st.caption(
                                f"🎭 {movie['genres']}"
                            )

                            col1, col2 = st.columns(2)

                            with col1:

                                st.metric(
                                    "⭐ Actual Rating",
                                    f"{movie['average_rating']:.2f}"
                                )

                            with col2:

                                st.metric(
                                    "🤖 Predicted Rating",
                                    f"{movie['predicted_rating']:.2f}"
                                )

                            col3, col4 = st.columns(2)

                            with col3:

                                st.metric(
                                    "👥 Ratings",
                                    f"{int(movie['rating_count']):,}"
                                )

                            with col4:

                                st.metric(
                                    "🎯 Recommendation Score",
                                    f"{movie['recommendation_score']:.2f}"
                                )

                            st.info(
                                f"💡 Shares genres with "
                                f"**{selected['title']}**: "
                                f"{genre_text}"
                            )

                            if st.button(
                                "🎬 View Movie",
                                key=f"view_movie_{rank}_{movie['movieId'] if 'movieId' in movie else rank}"
                            ):

                                st.session_state.selected_recommendation_movie = (
                                    movie["title"]
                                )

                                st.session_state.page_navigation = (
                                    "🔎 Movie Intelligence"
                                )

                                st.rerun()

# ============================================================

def get_movie_recommendations(movie_name, n=5):

    result = movie_data[
        movie_data["title"].str.contains(
            movie_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if result.empty:
        return None, []

    selected = result.iloc[0]

    selected_genres = set(
        str(selected["genres"]).split("|")
    )

    candidates = movie_data[
        movie_data["title"] != selected["title"]
    ].copy()

    def genre_similarity(genres):

        movie_genres = set(
            str(genres).split("|")
        )

        return len(
            selected_genres & movie_genres
        )

    candidates["genre_match"] = (
        candidates["genres"]
        .apply(genre_similarity)
    )

    max_rating = max(
        candidates["average_rating"].max(),
        1
    )

    max_predicted = max(
        candidates["predicted_rating"].max(),
        1
    )

    max_count = max(
        np.log1p(
            candidates["rating_count"]
        ).max(),
        1
    )

    candidates["recommendation_score"] = (

        candidates["genre_match"] * 3

        + (
            candidates["average_rating"]
            / max_rating
        ) * 2

        + (
            candidates["predicted_rating"]
            / max_predicted
        ) * 3

        + (
            np.log1p(
                candidates["rating_count"]
            )
            / max_count
        )
    )

    recommendations = (
        candidates[
            candidates["genre_match"] > 0
        ]
        .sort_values(
            "recommendation_score",
            ascending=False
        )
        .head(n)
    )

    return selected, recommendations



# ============================================================
# TMDB MOVIE POSTER
# ============================================================

# ============================================================
# GEMINI 3.1 VOICE OUTPUT
# ============================================================

def generate_voice_response(text):

    if client is None:

        return None

    try:

        interaction = client.interactions.create(

            model="gemini-3.1-flash-tts-preview",

            input=f"""
Speak naturally and clearly as a friendly
Movie AI Assistant.

Answer in a conversational way.

{text}
""",

            response_format={
                "type": "audio"
            },

            generation_config={
                "speech_config": [
                    {
                        "voice": "Kore"
                    }
                ]
            }
        )

        audio_data = interaction.output_audio.data

        # ----------------------------------------------------
        # Decode Gemini audio
        # ----------------------------------------------------

        if isinstance(audio_data, str):

            audio_data = base64.b64decode(
                audio_data
            )


        # ----------------------------------------------------
        # Create WAV file
        # Gemini TTS audio is PCM 24 kHz
        # ----------------------------------------------------

        wav_buffer = io.BytesIO()

        with wave.open(
            wav_buffer,
            "wb"
        ) as wav_file:

            wav_file.setnchannels(1)

            wav_file.setsampwidth(2)

            wav_file.setframerate(24000)

            wav_file.writeframes(
                audio_data
            )


        return wav_buffer.getvalue()


    except Exception as e:

        st.warning(
            f"🔊 Voice response unavailable: {e}"
        )

        return None


# ============================================================
# AI CHATBOT
# ============================================================

if page == "🤖 AI Chatbot":

    st.header(
        "🤖 Movie AI Assistant"
    )

    st.write(
        "Ask about movies, ratings, predictions "
        "and recommendations."
    )


    # --------------------------------------------------------
    # CHAT MEMORY
    # --------------------------------------------------------

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []


    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # --------------------------------------------------------
    # VOICE INPUT
    # --------------------------------------------------------

    st.markdown(
        "### 🎙️ Voice Assistant"
    )

    st.caption(
        "Click the microphone and speak your movie question."
    )


    voice_text = speech_to_text(

        language="en",

        start_prompt="🎙️ Start Speaking",

        stop_prompt="⏹️ Stop",

        just_once=True,

        use_container_width=True,

        key="movie_voice_input"
    )


    # --------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------

    if voice_text:

        question = voice_text

        st.info(
            f"🎤 You said: {voice_text}"
        )

    else:

        question = st.chat_input(
            "Ask something about movies..."
        )


    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        st.session_state.chat_history.append(

            {
                "role": "user",
                "content": question
            }

        )


        # ----------------------------------------------------
        # GEMINI + RAG
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🎬 Thinking..."
            ):

                answer = movie_chatbot(
                    question
                )


            # ------------------------------------------------
            # CURRENT MOVIE CARD
            # ------------------------------------------------

            current_movie = st.session_state.get(
                "last_movie_title",
                ""
            )

            if current_movie:

                movie_result = movie_data[
                    movie_data["title"]
                    .astype(str)
                    .str.strip()
                    == str(current_movie).strip()
                ]

                if movie_result.empty:

                    movie_result = movie_data[
                        movie_data["title"]
                        .astype(str)
                        .str.contains(
                            str(current_movie),
                            case=False,
                            na=False,
                            regex=False
                        )
                    ]

                if not movie_result.empty:

                    chat_movie = movie_result.iloc[0]

                    poster_url = get_movie_poster(
                        chat_movie["title"]
                    )

                    st.markdown(
                        f"""
<div class="chat-memory">
🧠 <b>Current Movie:</b> {chat_movie["title"]}
</div>
""",
                        unsafe_allow_html=True
                    )

                    card_col1, card_col2 = st.columns(
                        [1, 3]
                    )

                    with card_col1:

                        if poster_url:

                            st.image(
                                poster_url,
                                width=180
                            )

                    with card_col2:

                        st.markdown(
                            f"""
<div class="chat-movie-card">

<div class="chat-movie-title">
🎬 {chat_movie["title"]}
</div>

<div class="chat-movie-genres">
🎭 {chat_movie["genres"]}
</div>

<div class="chat-movie-stats">

<div class="chat-movie-stat">
<div class="chat-movie-label">
⭐ Actual Rating
</div>
<div class="chat-movie-value">
{chat_movie["average_rating"]:.2f}
</div>
</div>

<div class="chat-movie-stat">
<div class="chat-movie-label">
🤖 Predicted
</div>
<div class="chat-movie-value">
{chat_movie["predicted_rating"]:.2f}
</div>
</div>

<div class="chat-movie-stat">
<div class="chat-movie-label">
👥 Ratings
</div>
<div class="chat-movie-value">
{int(chat_movie["rating_count"]):,}
</div>
</div>

</div>

</div>
""",
                            unsafe_allow_html=True
                        )


            # ------------------------------------------------
            # SHOW TEXT ANSWER
            # ------------------------------------------------

            st.write(
                answer
            )


            # ------------------------------------------------
            # GENERATE VOICE
            # ------------------------------------------------

            with st.spinner(
                "🔊 Generating voice..."
            ):

                audio_bytes = (
                    generate_voice_response(
                        answer
                    )
                )


            # ------------------------------------------------
            # PLAY AUDIO
            # ------------------------------------------------

            if audio_bytes:

                st.audio(
                    audio_bytes,
                    format="audio/wav",
                    autoplay=False
                )

                st.caption(
                    "🔊 Gemini 3.1 voice response"
                )


        # ----------------------------------------------------
        # SAVE AI RESPONSE
        # ----------------------------------------------------

        st.session_state.chat_history.append(

            {
                "role": "assistant",
                "content": answer
            }

        )


    # --------------------------------------------------------
    # CLEAR CONVERSATION
    # --------------------------------------------------------

    if st.button(
        "🔄 Clear Conversation"
    ):

        st.session_state.chat_history = []

        st.session_state.previous_interaction_id = None

        st.session_state.last_movie_title = None

        st.rerun()

