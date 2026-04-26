Task 3: Comparative Corpus Analysis — AI/ML/NLP Research Evolution
EMATM0067 Introduction to AI and Text Analytics — Spring 2026
Analysis of temporal drift in AI, machine learning, and NLP research abstracts using multiple text analytics methods.

Project Overview
This project analyzes how research focus, framing, and communication have changed in AI/ML/NLP over three decades (1990–2021) using the arXiv abstracts corpus. We compare multiple text representation and analysis methods to understand what insights each approach reveals about the evolution of the field.
Research Question
How has the focus, framing, and communication of AI/ML/NLP research changed over time, and how do different text analytics methods affect the insights we can draw?


Repository Structure

├── README.md                          # This file
├── Text_AI_assignment.ipynb          # Main analysis notebook (all cells)
├── requirements.txt                   # Python dependencies


Pipeline

1. Load & Filter Data
   ├─ Download 2M arXiv abstracts from HuggingFace
   ├─ Filter to AI/ML/NLP categories (cs.AI, cs.LG, cs.CL, cs.NE, stat.ML)
   └─ Parse years and split into 5 periods (1990–99, 2000–06, 2007–12, 2013–17, 2018–21)

2. Exploratory Data Analysis
   ├─ Paper count per year
   ├─ Abstract length distribution
   ├─ Vocabulary size per period
   └─ Category breakdown

3. Text Preprocessing
   ├─ Lowercase, remove punctuation/numbers
   ├─ Remove stopwords (NLTK + domain-specific noise)
   └─ Lemmatisation

4. TF-IDF Baseline
   ├─ Build period-level TF-IDF matrix (unigrams + bigrams)
   ├─ Extract top terms per period
   └─ Compute cosine similarity between periods

5. SBERT Embeddings
   ├─ Encode abstracts with all-MiniLM-L6-v2 (384-dim)
   ├─ PCA to 2D for visualization
   ├─ Year-on-year drift magnitude
   └─ Period similarity heatmaps

6. LDA Topic Modelling
   ├─ Build bigram/trigram vocabulary
   ├─ Train models at k=10, 20, 30
   ├─ Score coherence (c_v metric)
   └─ Track topic prevalence over time
