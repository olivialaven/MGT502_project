# Building a Book Recommendation Systems for Library Users
![Library Overview](images/images:library_overview.png)


## 🧠 Overview

Ever wandered through a library, overwhelmed by the sheer number of books, wondering what to read next? Now imagine the library knows you so well, it can whisper, *“You might also like…”* just like your favorite streaming platform. That's exactly what this project aims to do.

We tried to build a smart recommendation system for the university’s library platform to help students and staff discover books they’ll actually want to read—based on what books they’ve interacted with before, what others like them are reading, and even the hidden magic tucked inside book metadata.

Our system predicts which books each user is most likely to enjoy and presents the **Top 10 personalized recommendations**. We used a mix of techniques like collaborative filtering and text embeddings to make these suggestions as accurate and interesting as possible.

💡 **Goal**: We evaluated our models using **Precision@10**, with the goal of achieving a leaderboard score higher than 0.1452.

---
### 🔎 Navigate through the process with us:

- [📊 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda) 
- [🛠️ Models and Methods](#️-models-and-methods)
- [🧪 Evaluation Results](#-evaluation-results)
- [🌟 Best Model & Interpretation](#-best-model--interpretation)
- [🧪 Data Augmentation](#-data-augmentation)
- [🎯 Final Leaderboard Score](#-final-leaderboard-score)
- [💡 Interface (Streamlit App)](#-interface-streamlit-app)
- [📹 Video Presentation](#-video-presentation)
- [📌 Key Takeaways](#-key-takeaways)

---
## 📊 Exploratory Data Analysis (EDA)
### 🗃️ Aquiring our data
For this assignment, two datasets were provided: one dataset containing basic information about the items (the books) and one dataset containing the interactions between the users and the items. First, we need to download



📚

# 📚 University Library Recommender System

## 🧠 Overview
This project builds a recommendation system to enhance the university library platform by helping students and members discover books aligned with their interests. The system predicts books users might like, based on historical book rental data, and displays the top-10 recommendations for each user.

We evaluated our models using **Precision@10**, with the goal of achieving a leaderboard score higher than 0.1452.

---

## 🔍 Exploratory Data Analysis (EDA)

### Interaction Data
- Total users: ...
- Total books: ...
- Sparsity of interaction matrix: ...
- Average number of books per user: ...
- Long tail distribution of books: ...

### Book Metadata
- Metadata fields: Title, Author, Subjects, Language, etc.
- Most common subjects: ...
- Distribution of languages: ...
- Missing data: ...

(→ Insert some plots or upload them in `/results` and embed them here)

---

## 🛠️ Models and Methods

### 1. User-User Collaborative Filtering
- Similarity metric: Cosine
- Top-k neighbors: ...
- Tuned hyperparameters: ...
- Pros/Cons: ...

### 2. Item-Item Collaborative Filtering
- Similarity metric: ...
- Tuned hyperparameters: ...

### 3. Other Techniques
- **TF-IDF + Cosine Similarity**
- **BERT-based embeddings (e.g., `all-MiniLM`)**
- **Hybrid model**: Weighted ensemble of user-user, item-item, and content-based
- **Neural recommender (optional)**: Briefly describe if used

---

## 🧪 Evaluation Results

| Model               | Precision@10 | Recall@10 |
|--------------------|--------------|-----------|
| User-User CF        |              |           |
| Item-Item CF        |              |           |
| TF-IDF Similarity   |              |           |
| BERT Embeddings     |              |           |
| Hybrid Model        |              |           |

---

## 🌟 Best Model & Interpretation

- The hybrid model outperformed individual models, especially on cold-start users.
- Examples of “good” recommendations:  
  - User X → [Book A, Book B, Book C]  
  - Why it’s good: matches history in genre/subject.

- Examples of “bad” recommendations:  
  - User Y → [Book D, Book E, Book F]  
  - Why it’s bad: user prefers fiction, got academic books instead.

---

## 🧪 Data Augmentation

We used the **Google Books API** to enrich metadata with:
- Book cover images
- Description/summaries
- Genre/tags
- Published date

This additional data improved our embedding quality and recommendation relevance.

---

## 🎯 Final Leaderboard Score

- Public leaderboard Precision@10: **...**
- Rank: **...**

---

## 💡 Interface (Streamlit App)

You can try our recommendation system with a simple UI here:  
👉 [Try the App](LINK_TO_YOUR_STREAMLIT_APP)

- Input a user ID
- See top-10 book recommendations (with covers!)
- Includes a fallback recommendation list if the user is new

![Screenshot](results/ui_screenshot.png)

---

## 📹 Video Presentation

🎥 [Watch our 3-minute demo](https://www.youtube.com/...)  
In the video, we explain:
- The problem and why it matters
- Our methodology and models
- Evaluation results
- A live UI demo

---

## 📁 Project Structure

```plaintext
├── README.md
├── hybrid_recommender.ipynb      # Full notebook (linked below)
├── data/
│   ├── interactions.csv
│   ├── items_metadata.csv
│   └── sample_submission.csv
├── results/
│   └── precision_plot.png
├── app/
│   └── streamlit_app.py

