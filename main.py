import os
import re
import string
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import seaborn as sns
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from wordcloud import WordCloud

import gensim
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

warnings.filterwarnings("ignore")

# Download NLTK data
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ── Global style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       11,
})

PERIOD_BINS   = [1990, 2000, 2007, 2013, 2018, 2022]
PERIOD_LABELS = ["1990–99", "2000–06", "2007–12", "2013–17", "2018–21"]
PERIOD_COLORS = {
    "1990–99": "#4878CF",
    "2000–06": "#6ACC65",
    "2007–12": "#D65F5F",
    "2013–17": "#B47CC7",
    "2018–21": "#C4AD66",
}
TARGET_CATS = {"cs.AI", "cs.LG", "cs.CL", "cs.NE", "stat.ML"}

print("All imports successful.")


# ============================================================
# STEP 1 — LOAD & FILTER DATA
# ============================================================

print("\n" + "="*60)
print("STEP 1 — Loading Dataset")
print("="*60)

from datasets import load_dataset

print("Downloading dataset from HuggingFace...")
raw = load_dataset(
    "gfissore/arxiv-abstracts-2021",
    split="train",
    trust_remote_code=True,
)
print(f"Total records: {len(raw):,}")

# ── Filter to AI/ML/NLP categories ──────────────────────────
def is_target(categories):
    if not categories:
        return False
    for cat in categories:
        for piece in cat.split():
            if piece in TARGET_CATS:
                return True
    return False

print("Filtering to AI/ML/NLP papers...")
filtered = raw.filter(lambda x: is_target(x["categories"]), num_proc=4)
print(f"After filter: {len(filtered):,} papers")

df = filtered.to_pandas()

# ── Parse year from arXiv ID ─────────────────────────────────
def parse_year(arxiv_id):
    arxiv_id = str(arxiv_id).strip()
    m = re.match(r'^(\d{2})(\d{2})\.\d+', arxiv_id)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    m = re.match(r'.*?/(\d{2})\d{5}', arxiv_id)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    return None

df["year"] = df["id"].apply(parse_year)
df = df.dropna(subset=["year", "abstract"])
df["year"] = df["year"].astype(int)
df = df[(df["year"] >= 1990) & (df["year"] <= 2021)]
df = df[df["abstract"].str.strip().str.len() > 50]
df = df.reset_index(drop=True)

# ── Assign time periods ──────────────────────────────────────
df["period"] = pd.cut(
    df["year"],
    bins=PERIOD_BINS,
    labels=PERIOD_LABELS,
    right=False
)

print(f"\nFinal corpus: {len(df):,} papers")
print(f"Year range  : {df['year'].min()} – {df['year'].max()}")
print("\nPapers per period:")
print(df["period"].value_counts().sort_index().to_string())


# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "="*60)
print("STEP 2 — Exploratory Data Analysis")
print("="*60)

df["word_count"] = df["abstract"].apply(lambda t: len(str(t).split()))

year_counts = df["year"].value_counts().sort_index()
avg_len     = df.groupby("year")["word_count"].mean()

# Vocabulary per period
vocab_size = {}
for period, group in df.groupby("period", observed=True):
    tokens = " ".join(group["abstract"].values).lower().split()
    vocab_size[str(period)] = len(set(tokens))

# Category counts
cat_counter = Counter()
for cats in df["categories"]:
    for cat in cats:
        for piece in cat.split():
            if piece in TARGET_CATS:
                cat_counter[piece] += 1

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle(
    "Exploratory Data Analysis — AI/ML/NLP Corpus",
    fontsize=15, fontweight="bold"
)

# 2a — Paper count per year
ax = axes[0, 0]
ax.bar(year_counts.index, year_counts.values,
       color="#4878CF", edgecolor="white", linewidth=0.5)
ax.set_title("Paper count per year", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Papers")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)
ax.tick_params(axis="x", rotation=45)

# 2b — Average abstract length
ax = axes[0, 1]
ax.plot(avg_len.index, avg_len.values,
        color="#DD8452", linewidth=2, marker="o", markersize=4)
ax.set_title("Avg abstract length (words) per year", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Mean word count")
ax.tick_params(axis="x", rotation=45)

# 2c — Word count distribution
ax = axes[1, 0]
ax.hist(df["word_count"], bins=60,
        color="#55A868", edgecolor="white", linewidth=0.5)
med = df["word_count"].median()
ax.axvline(med, color="red", linestyle="--", label=f"Median={med:.0f}")
ax.set_title("Distribution of abstract word counts", fontweight="bold")
ax.set_xlabel("Word count"); ax.set_ylabel("Frequency")
ax.legend()

# 2d — Vocabulary size per period
ax = axes[1, 1]
ax.bar(vocab_size.keys(), vocab_size.values(),
       color="#C44E52", edgecolor="white")
ax.set_title("Vocabulary size per period", fontweight="bold")
ax.set_xlabel("Period"); ax.set_ylabel("Unique tokens")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)

# 2e — Category breakdown
ax = axes[2, 0]
cat_df = pd.DataFrame(cat_counter.most_common(), columns=["cat", "count"])
ax.barh(cat_df["cat"], cat_df["count"], color="#8172B2")
ax.set_title("Papers by arXiv category", fontweight="bold")
ax.set_xlabel("Count")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)

# 2f — Period share pie
ax = axes[2, 1]
period_counts = df["period"].value_counts().sort_index()
ax.pie(
    period_counts.values,
    labels=period_counts.index,
    autopct="%1.1f%%",
    colors=[PERIOD_COLORS[p] for p in period_counts.index],
    startangle=140
)
ax.set_title("Share of papers per period", fontweight="bold")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → eda_plots.png")

print(f"\nMean word count  : {df['word_count'].mean():.1f}")
print(f"Median word count: {df['word_count'].median():.1f}")


# ============================================================
# STEP 3 — TEXT PREPROCESSING
# ============================================================

print("\n" + "="*60)
print("STEP 3 — Preprocessing")
print("="*60)

NLTK_STOPS  = set(stopwords.words("english"))
EXTRA_STOPS = {
    "propose", "proposed", "paper", "present", "show", "shown",
    "method", "approach", "result", "results", "using", "based",
    "also", "two", "one", "new", "use", "used", "however",
    "data", "model", "models", "task", "tasks", "set", "thus",
    "first", "second", "well", "large", "different", "may",
    "problem", "problems", "performance", "work", "works",
    "existing", "achieve", "achieved", "state", "art",
}
STOP_WORDS  = NLTK_STOPS | EXTRA_STOPS
lemmatizer  = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

print("Cleaning abstracts...")
tqdm.pandas(desc="Preprocessing")
df["clean_abstract"] = df["abstract"].progress_apply(preprocess)

print("\nSample — original:")
print(df["abstract"].iloc[0][:250])
print("\nSample — cleaned:")
print(df["clean_abstract"].iloc[0][:250])
