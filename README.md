\# 🎬 Movie AI Intelligence System



An AI-powered Movie Recommendation and Rating Intelligence System that combines Machine Learning, semantic search, FAISS, TF-IDF, and an intelligent chatbot to help users discover and explore movies.



\## 🚀 Features



\- 🎯 Movie recommendation system

\- ⭐ Movie rating prediction

\- 🔍 Semantic movie search

\- 🤖 AI-powered movie chatbot

\- 🧠 RAG (Retrieval-Augmented Generation) pipeline

\- 📊 Movie rating intelligence and analytics

\- 🎭 Genre-based movie exploration

\- 🔎 Search movies using natural-language queries

\- ⚡ FAISS-based similarity search

\- 📚 TF-IDF-based text processing

\- 📈 Machine Learning-based rating prediction



\## 🏗️ System Architecture



```text

&#x20;                   ┌──────────────────────┐

&#x20;                   │       User           │

&#x20;                   └──────────┬───────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌──────────────────────┐

&#x20;                   │   Movie AI App        │

&#x20;                   │       app.py          │

&#x20;                   └──────────┬───────────┘

&#x20;                              │

&#x20;             ┌────────────────┼────────────────┐

&#x20;             │                │                │

&#x20;             ▼                ▼                ▼

&#x20;      ┌────────────┐   ┌─────────────┐   ┌──────────────┐

&#x20;      │ Movie      │   │ Semantic    │   │ Rating       │

&#x20;      │ Recommend. │   │ Search/RAG  │   │ Prediction   │

&#x20;      └─────┬──────┘   └──────┬──────┘   └──────┬───────┘

&#x20;            │                 │                 │

&#x20;            ▼                 ▼                 ▼

&#x20;      Movie Data         FAISS / TF-IDF     ML Model

\## 🧠 Technologies Used



\### Programming

\- Python



\### Machine Learning

\- Scikit-learn

\- Random Forest

\- TF-IDF



\### AI / RAG

\- Retrieval-Augmented Generation (RAG)

\- Semantic Search

\- Vector Embeddings

\- FAISS



\### Application

\- Streamlit



\### Data Processing

\- Pandas

\- NumPy



\## 📂 Project Structure



```text

movie-ai-intelligence/

│

├── app.py

├── requirements.txt

├── feature\_columns.pkl

├── tfidf\_model.pkl

├── .gitignore

└── README.md

📊 Machine Learning



The system uses movie and rating information to generate movie intelligence and predictions.



Important features include:



Average movie rating

Rating count

User rating information

Movie rating statistics

Movie genres

Other engineered movie features



A Random Forest-based approach was used for rating prediction.



🔍 Semantic Search



The application converts movie information into vector representations and uses similarity search to retrieve movies relevant to a user's query.



Example queries:



Movies similar to Forrest Gump

Best emotional movies

Movies about friendship

Good action movies from the 2000s

🤖 RAG Chatbot



The chatbot uses a Retrieval-Augmented Generation approach:



User Question

&#x20;     ↓

Query Processing

&#x20;     ↓

Semantic Retrieval

&#x20;     ↓

Relevant Movie Documents

&#x20;     ↓

Context

&#x20;     ↓

AI Response



This allows the chatbot to answer movie-related questions using the project's movie knowledge base.



📦 Installation



Clone the repository:



git clone https://github.com/priyaadharshinipe/movie-ai-intelligence.git



Move into the project directory:



cd movie-ai-intelligence



Install the required Python packages:



pip install -r requirements.txt

▶️ Run the Application



Start the Streamlit application:



streamlit run app.py



The application will open in your browser.



🔐 API Keys



If an external AI API is used, keep the API key private.



Do not hard-code API keys directly into app.py.



Use environment variables or Streamlit secrets instead.



Never upload API keys to GitHub.



📈 Project Goals



The main goals of this project are:



Build an intelligent movie recommendation system

Predict and analyze movie ratings

Provide semantic movie discovery

Implement a RAG-based movie chatbot

Combine Machine Learning and Generative AI

Create an interactive user-friendly application

🔮 Future Improvements

Personalized recommendations based on user history

Real-time movie information

Improved recommendation algorithms

Hybrid recommendation system

Advanced conversational memory

Movie posters and trailers

Cloud deployment

User authentication

Continuous model improvement

👩‍💻 Author



Priyaa Dharshini P.E



Artificial Intelligence \& Data Science



⭐ Project



Movie AI Intelligence System





