# 🎬 Movie AI Intelligence System

### AI-Powered Movie Recommendation, Rating Intelligence & Conversational Assistant
---
## 📌 Overview

**Movie AI Intelligence System** is an interactive AI-powered movie analytics and recommendation application built with **Python and Streamlit**.

The system combines:

* 🤖 Machine Learning
* 🎯 Movie recommendations
* ⭐ Rating prediction
* 🔍 Semantic search
* 🧠 Retrieval-Augmented Generation (RAG)
* ⚡ FAISS vector similarity search
* 📝 Sentence Transformer embeddings
* 💬 Conversational AI with Gemini
* 🎙️ Voice-based movie queries
* 🔊 AI-generated voice responses
* 🎬 TMDB movie posters
* 📊 Movie analytics and visualization

The application provides a single platform where users can explore movie statistics, compare actual and predicted ratings, discover similar movies, and interact with an AI movie assistant.

---

# ✨ Features

## 📊 1. Movie Analytics Dashboard

The dashboard provides an overview of the movie dataset with interactive visualizations and KPIs.

### Key metrics

* 🎬 Total number of movies
* ⭐ Average movie rating
* 👥 Total number of ratings
* 🤖 Average predicted rating

### Visual analytics

* ⭐ Rating distribution
* 🎭 Genre distribution
* 🏆 Top-rated movies
* 🔥 Most popular movies
* 📈 Rating vs. popularity analysis

Movie cards can also display movie posters retrieved through the TMDB API.

---

## 🔎 2. Movie Intelligence

Search for a specific movie and view detailed movie intelligence.

For each movie, the system displays:

* 🎬 Movie title
* 🎭 Genres
* ⭐ Actual dataset rating
* 🤖 ML predicted rating
* 👥 Number of ratings
* 📊 Difference between actual and predicted ratings
* 💡 AI-generated rating insight
* 🎬 Movie poster

Example:

```text
Movie: Forrest Gump

Actual Rating:      4.16 / 5
Predicted Rating:   4.05 / 5
Rating Count:       10,000+
Difference:         -0.11
```

---

# ⭐ 3. Rating Prediction

The application uses a trained **Random Forest-based machine learning model** to generate predicted ratings.

Users can filter movies using:

* Minimum predicted rating
* Minimum number of ratings

The results include:

| Information       | Description          |
| ----------------- | -------------------- |
| Movie             | Movie title          |
| Genres            | Movie genres         |
| Actual Rating     | Dataset rating       |
| Rating Count      | Number of ratings    |
| Predicted Rating  | ML-generated rating  |
| Rating Difference | Prediction vs actual |

Users can also download the filtered prediction results as a CSV file.

---

# 🎯 4. Movie Recommendation System

The recommendation engine finds movies similar to a selected movie.

The current implementation considers:

* 🎭 Genre similarity
* ⭐ Actual movie rating
* 🤖 Predicted rating
* 👥 Rating popularity

A recommendation score is calculated using these factors.

Conceptually:

```text
Recommendation Score
        │
        ├── Genre Similarity
        │
        ├── Actual Rating
        │
        ├── Predicted Rating
        │
        └── Rating Popularity
```

Users can select between **5 and 20 recommendations**.

Each recommendation provides:

* Movie poster
* Movie title
* Genres
* Actual rating
* Predicted rating
* Rating count
* Recommendation score
* Shared genres

Users can then select **View Movie** to open the detailed Movie Intelligence page.

---

# 🔍 5. Semantic Movie Search

The application uses **Sentence Transformers** to convert movie queries into vector embeddings.

The embedding model used is:

```text
all-MiniLM-L6-v2
```

The generated query embedding is searched against the pre-built FAISS index.

### Semantic Search Pipeline

```text
User Query
     │
     ▼
Sentence Transformer
     │
     ▼
Query Embedding
     │
     ▼
FAISS Similarity Search
     │
     ▼
Relevant Movie Documents
     │
     ▼
Movie Context
```

This allows users to search using natural-language concepts such as:

```text
Movies about friendship

Emotional movies

Good action movies

Movies similar to Forrest Gump

Movies from the 2000s
```

---

# 🧠 6. RAG-Based AI Chatbot

The Movie AI Assistant uses a **Retrieval-Augmented Generation (RAG)** architecture.

Instead of relying only on the language model's general knowledge, the application first retrieves relevant movie information from the project's movie knowledge base.

### RAG Architecture

```text
              User Question
                    │
                    ▼
             Query Processing
                    │
                    ▼
        Sentence Transformer
              Embedding
                    │
                    ▼
             FAISS Search
                    │
                    ▼
        Relevant Movie Documents
                    │
                    ▼
                Context
                    │
                    ▼
             Google Gemini
                    │
                    ▼
             AI Response
```

The chatbot is instructed to use retrieved movie information for movie-specific facts such as:

* Ratings
* Predicted ratings
* Genres
* Number of ratings
* Recommendations
* Rating differences

The system also includes a **local RAG fallback** when Gemini is unavailable.

---

# 💬 7. Conversational Movie Memory

The chatbot maintains conversational context using Streamlit session state and Gemini interaction history.

For example:

```text
User:
Tell me about Forrest Gump.

Assistant:
[Forrest Gump information]

User:
What is its rating?

Assistant:
The rating is ...

User:
Would you recommend it?

Assistant:
Based on its available rating information...
```

The assistant can understand follow-up references such as:

```text
it
this movie
that movie
its rating
its genre
its story
would you recommend it?
is it worth watching?
```

This creates a more natural movie conversation.

---

# 🎙️ 8. Voice Movie Assistant

The chatbot supports voice input using:

```text
streamlit-mic-recorder
```

Users can click the microphone button and speak their movie question.

### Voice Pipeline

```text
User Speech
     │
     ▼
Speech-to-Text
     │
     ▼
Movie AI Chatbot
     │
     ▼
RAG + Gemini
     │
     ▼
Text Response
```

---

# 🔊 9. AI Voice Response

The application can also convert the chatbot's response into spoken audio using Gemini's text-to-speech capability.

The current implementation uses:

```text
gemini-3.1-flash-tts-preview
```

The generated audio is converted into WAV format and played directly inside Streamlit.

### Complete Voice Interaction

```text
🎙️ User speaks
       ↓
Speech-to-Text
       ↓
RAG Retrieval
       ↓
Gemini AI
       ↓
Text Answer
       ↓
Gemini TTS
       ↓
🔊 Voice Response
```

---

# 🎬 10. Movie Posters

Movie posters are dynamically retrieved through the **TMDB API**.

The application:

1. Receives a movie title
2. Searches TMDB
3. Retrieves the poster path
4. Displays the poster inside Streamlit

If the API is unavailable, the application continues without the poster.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │        USER          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   STREAMLIT APP      │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
     │   Dashboard    │    │ Recommendation │    │ Movie          │
     │   Analytics    │    │    Engine      │    │ Intelligence   │
     └────────────────┘    └────────────────┘    └────────────────┘
             │                      │                      │
             │                      ▼                      │
             │              ┌────────────────┐             │
             │              │ Genre + Rating │             │
             │              │ Similarity     │             │
             │              └────────────────┘             │
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Movie Data        │
                         │   movie_data.pkl    │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                     ▼              ▼              ▼
              ┌────────────┐ ┌────────────┐ ┌────────────┐
              │ Random     │ │ Sentence   │ │ Movie      │
              │ Forest     │ │ Transformer│ │ Documents  │
              │ Model      │ │ Embeddings │ │            │
              └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                    │              │              │
                    ▼              ▼              ▼
              Rating         Query Vector     FAISS
              Prediction          │            Index
                                  └──────┬───────┘
                                         │
                                         ▼
                                  Retrieved Context
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │ Google      │
                                  │ Gemini AI   │
                                  └──────┬───────┘
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                         Text Response        Gemini TTS
                                                    │
                                                    ▼
                                             🔊 Audio Response

                         TMDB API
                            │
                            ▼
                       🎬 Posters
```

---

# 🧩 Technology Stack

## Programming

* Python

## Application

* Streamlit

## Data Processing

* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* Random Forest

## Natural Language Processing

* Sentence Transformers
* `all-MiniLM-L6-v2`

## Vector Search

* FAISS

## Generative AI

* Google Gemini API

## Voice AI

* Speech-to-text using `streamlit-mic-recorder`
* Gemini text-to-speech

## External API

* TMDB API

## Model Serialization

* Pickle
* Joblib

---

# 📂 Project Structure

The current application loads the following project artifacts:

```text
movie-ai-intelligence/
│
├── app.py
├── requirements.txt
│
├── movie_data.pkl
├── movie_documents.pkl
├── movie_faiss.index
├── movie_rating_model.pkl
├── feature_columns.pkl
│
├── .gitignore
│
└── README.md
```

### File Description

| File                     | Purpose                                        |
| ------------------------ | ---------------------------------------------- |
| `app.py`                 | Main Streamlit application                     |
| `movie_data.pkl`         | Processed movie dataset and movie intelligence |
| `movie_documents.pkl`    | Movie documents used for retrieval             |
| `movie_faiss.index`      | FAISS vector index                             |
| `movie_rating_model.pkl` | Trained Random Forest rating model             |
| `feature_columns.pkl`    | ML feature column information                  |
| `requirements.txt`       | Python dependencies                            |
| `.gitignore`             | Files excluded from Git                        |

> The application currently requires all five model/data artifacts above to start successfully.

---

# 📊 Machine Learning Pipeline

The rating intelligence component uses movie-level and rating-related features.

Important features include:

```text
Average Movie Rating
Rating Count
User Rating Information
Movie Rating Statistics
Movie Genres
Engineered Features
```

The trained model is loaded from:

```text
movie_rating_model.pkl
```

The model generates:

```text
predicted_rating
```

The application then compares:

```text
Actual Rating
      vs
Predicted Rating
```

to calculate:

```text
rating_difference
```

---

# 🔎 Retrieval Pipeline

The retrieval system uses:

```text
Sentence Transformer
        ↓
384-dimensional embedding
        ↓
FAISS Index
        ↓
Top-K Similar Movies
        ↓
Movie Documents
        ↓
Gemini Context
```

The application retrieves the top **5 movie documents** for chatbot queries.

---

# 🔐 API Configuration

The application uses two optional external API keys.

### Gemini API

Used for:

* Conversational AI
* RAG response generation
* Voice generation

Configure:

```text
GEMINI_API_KEY
```

### TMDB API

Used for:

* Movie posters

Configure:

```text
TMDB_API_KEY
```

### Streamlit Secrets

For Streamlit Community Cloud, add the keys under:

```text
App → Settings / Manage App → Secrets
```

Example:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
TMDB_API_KEY = "your_tmdb_api_key"
```

**Never commit API keys to GitHub.**

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/priyaadharshinipe/movie-ai-intelligence.git
```

## 2. Open the project

```bash
cd movie-ai-intelligence
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Configure API keys

Set your Gemini and TMDB API keys using environment variables or Streamlit secrets.

## 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# ☁️ Streamlit Community Cloud Deployment

The application can be deployed using **Streamlit Community Cloud**.

Basic deployment process:

```text
GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ▼
Select app.py
       │
       ▼
Configure Secrets
       │
       ▼
Deploy
       │
       ▼
🌐 Live Movie AI Application
```

Before deployment, make sure the repository contains:

```text
app.py
requirements.txt
movie_data.pkl
movie_documents.pkl
movie_faiss.index
movie_rating_model.pkl
feature_columns.pkl
```

Large model/data files may require Git LFS or another external storage strategy if they exceed GitHub's normal file-size limits.

---

# 📈 Current Application Pages

The Streamlit application currently contains five main pages:

| Page                  | Purpose                               |
| --------------------- | ------------------------------------- |
| 📊 Dashboard          | Movie analytics and visualizations    |
| 🔎 Movie Intelligence | Detailed movie-level analysis         |
| ⭐ Rating Predictions  | ML rating predictions and filtering   |
| 🎯 Recommendations    | Similar movie recommendations         |
| 🤖 AI Chatbot         | RAG + Gemini conversational assistant |

---

# 💡 Example Queries

The AI assistant can handle questions such as:

```text
Tell me about Forrest Gump

What is its rating?

What are its genres?

How many people rated it?

Would you recommend it?

Movies similar to Forrest Gump

Tell me about emotional movies

Recommend good action movies
```

---

# 🛡️ Fallback Architecture

The chatbot has a fallback mechanism.

If Gemini is unavailable:

```text
User Question
      ↓
FAISS Retrieval
      ↓
Movie Documents
      ↓
Local RAG Response
```

This allows the application to still provide retrieved movie information without depending entirely on the external AI service.

---

# 🚀 Future Improvements

Potential improvements include:

* 🎯 Personalized recommendations based on individual user history
* 🔄 Hybrid recommendation system
* 👤 User profiles and authentication
* 🧠 Long-term conversational memory
* 🎬 Movie trailers
* 📅 Real-time movie information
* 🌐 More external movie APIs
* 📱 Mobile-friendly interface
* 📊 Advanced model evaluation
* 🔍 Improved semantic ranking
* 🧠 Personalized ranking models
* ⚡ Faster vector retrieval
* 📈 Recommendation performance evaluation
* ☁️ Scalable cloud deployment

---

# 🎯 Project Objectives

The project aims to demonstrate how multiple AI technologies can be integrated into a single practical application.

### Primary objectives

* Build an intelligent movie recommendation system
* Develop ML-based rating prediction
* Perform semantic movie retrieval
* Implement a RAG pipeline
* Integrate Generative AI
* Support conversational movie exploration
* Add voice interaction
* Provide movie analytics
* Build an interactive Streamlit application

---

# 🧪 Key Concepts Demonstrated

This project demonstrates practical implementation of:

```text
Data Processing
       ↓
Feature Engineering
       ↓
Machine Learning
       ↓
Vector Embeddings
       ↓
FAISS Retrieval
       ↓
RAG
       ↓
Generative AI
       ↓
Conversational Memory
       ↓
Voice AI
       ↓
Interactive Web Application
```

---

# 👩‍💻 Author

## Priyaa Dharshini P.E

**Artificial Intelligence & Data Science**

Interested in:

* Artificial Intelligence
* Machine Learning
* Data Science
* Generative AI
* Intelligent Applications

---

# ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

**Movie AI Intelligence System**
*Recommendation • Rating Intelligence • Semantic Search • RAG • Conversational AI*

---








