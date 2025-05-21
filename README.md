# Building a Book Recommendation Systems for Library Users
![Library Overview](images/images:library_overview.png)


## 🧠 Overview

Ever wandered through a library, overwhelmed by the sheer number of books, wondering what to read next? Now imagine the library knows you so well, it can whisper, *“You might also like…”* just like your favorite streaming platform. That's exactly what this project aims to do.

We tried to build a smart recommendation system for the university’s library platform to help students and staff discover books they’ll actually want to read-based on what books they’ve interacted with before, what others like them are reading, and even the hidden magic tucked inside book metadata.

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
### Aquiring our data
For this assignment, two datasets were provided. The first one, _items_, contains information about the books, such as the book id (helps us link the correct item to the one a user interacted with), title, author, ISBN, publisher, and some subjects. The books are in French. The second dataset, _interactions_, contains the known interactions between the users and the items, as well as a timestamp (when the interaction occurred). This _interactions_ dataset is the training part of a larger dataset; the test part is what our recommendations are compared to on the leaderboard.

Based on a first look of our datasets, we have:
- Total users = 7,838
- Total books (including books no one has interacted with) = 15,291
- Number of interactions = 87,047
- Items no user has interacted with = 182
  
### Checking for duplicates
It is possible that a user has interacted with the same item multiple times. However, given that there is a timestamp for each user's interaction with an item, we can see if there are any duplicates since only one interaction should be registered for one instance of an interaction. There should also not be any duplicates in the _items_ dataset.

To test this, we run the following:
```python
# Checking duplicates in the interactions and items datasets
interaction_duplicates = interactions.duplicated().sum()
items_duplicates = items.duplicated().sum()
print(f'Interaction data duplicates = {interaction_duplicates}, Item data duplicates = {items_duplicates}')
```
The results show that we have two (2) duplicates in _interactions_; no duplicates in _items_. We handled these duplicates by dropping the second appearance of the interaction (same user, item, and timestamp).

### Checking missing values
To understand the coverage of our data, we decided to check how many missing values we have in each column of the two datasets. The results showed that the _interactions_ dataframe is complete and has no missing values. The _items_ dataframe, however, has quite a few missing values:

| Feature       | Missing Values |
|---------------|----------------|
| Title         | 0              |
| Author        | 2,653          |
| ISBN Valid    | 723            |
| Publisher     | 25             |
| Subjects      | 2,223          |
| i             | 0              |

This limited amount of data could limit our possibilities to create a more precise recommender system. One way to address this issue is to use API calls to retrieve more metadata. Therefore, we used Google Books and ISBNdb API calls to try to fill some missing values, as well as adding more information about the items. These API calls added valuable information such as:

  - filling in missing Authors
  - adding book cover images
  - adding description/summaries
  - adding book dimensions
  - adding publication date

...and so on. This additional data was used to later improve our embedding quality and, as a result, recommendation relevance. Although we are getting ahead of ourselves with this, in order to make it easier to follow along in the process, we decided to put the code for this metadata aquisition and cleaning in a seperate file. The code for this can be found ***here***. Addimg the information from the API calls reduced, for example, the number of missing authors from 2,653 to only 789.

### Visualizing the data

#### 


















Below is a breakdown of the dataframe containing the interactions:
- Total users: ...
- Total books: ...
- Sparsity of interaction matrix: ...
- Average number of books per user: ...
- Long tail distribution of books: ...

#### Book Metadata
- Metadata fields: Title, Author, Subjects, Language, etc.
- Most common subjects: ...
- Distribution of languages: ...
- Missing data: ...

Instead, we will now start from importing the complete items dataframe (including all metadata from the API calls), and the interactions dataframe:

```python
# Load the datasets
interactions = pd.read_csv('https://raw.githubusercontent.com/olivialaven/MGT502_project/refs/heads/main/interactions_train.csv')
display(interactions.head())
```
```text
index    u	i	t
0	4456	8581	1.687541e+09
1	142	1964	1.679585e+09
2	362	3705	1.706872e+09
3	1809	11317	1.673533e+09
4	4384	1323	1.681402e+09
```
```python
items = pd.read_csv("https://raw.githubusercontent.com/olivialaven/MGT502_project/refs/heads/main/merged_items.csv")
# Display only columns due to that there is a lot of data
display(items.head(0))
```
```text
Title	Author	ISBN Valid	Publisher	Subjects	i	description	mainCategory	publisher	synopsis	...	date_published	subjects	isbn13	msrp	binding	isbn	isbn10	edition	related	dewey_decimal
```
The resulting  






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
👉 [Try the App](https://booksmt.streamlit.app)

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

