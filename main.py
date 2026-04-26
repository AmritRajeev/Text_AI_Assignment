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

# ============================================================
# STEP 4 — TF-IDF BASELINE
# ============================================================

print("\n" + "="*60)
print("STEP 4 — TF-IDF Baseline")
print("="*60)

# ── 4a Period-level TF-IDF ───────────────────────────────────
period_docs   = {
    str(p): " ".join(g["clean_abstract"].values)
    for p, g in df.groupby("period", observed=True)
}
period_names  = PERIOD_LABELS
period_texts  = [period_docs[p] for p in period_names]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20_000,
    min_df=2,
    sublinear_tf=True,
)
tfidf_matrix  = vectorizer.fit_transform(period_texts)
feature_names = vectorizer.get_feature_names_out()

print(f"TF-IDF matrix: {tfidf_matrix.shape[0]} periods "
      f"× {tfidf_matrix.shape[1]:,} features")

# ── 4b Top terms per period ──────────────────────────────────
TOP_N = 20

def top_terms(period_idx, n=TOP_N):
    row     = tfidf_matrix[period_idx].toarray().flatten()
    top_idx = row.argsort()[::-1][:n]
    return [(feature_names[i], round(row[i], 4)) for i in top_idx]

print("\nTop 15 terms per period:")
for i, period in enumerate(period_names):
    terms = top_terms(i, 15)
    print(f"\n  [{period}]")
    for term, score in terms:
        print(f"    {term:<35} {score:.4f}")

# ── 4c Bar charts ────────────────────────────────────────────
fig, axes = plt.subplots(1, len(period_names),
                          figsize=(5 * len(period_names), 7))
colors = sns.color_palette("husl", len(period_names))

for i, (period, ax) in enumerate(zip(period_names, axes)):
    terms  = top_terms(i, 15)
    words  = [t[0] for t in terms]
    scores = [t[1] for t in terms]
    ax.barh(words[::-1], scores[::-1], color=colors[i])
    ax.set_title(period, fontweight="bold", fontsize=11)
    ax.set_xlabel("TF-IDF score")
    ax.tick_params(axis="y", labelsize=8)

fig.suptitle(
    "Top 15 TF-IDF Terms per Period",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("tfidf_top_terms.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → tfidf_top_terms.png")

# ── 4d Word clouds ───────────────────────────────────────────
fig, axes = plt.subplots(1, len(period_names),
                          figsize=(5 * len(period_names), 4))
palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

for i, (period, ax) in enumerate(zip(period_names, axes)):
    freq_dict = {t[0]: t[1] for t in top_terms(i, 60)}
    wc = WordCloud(
        width=400, height=300,
        background_color="white",
        colormap=palettes[i % len(palettes)],
        max_words=60,
    ).generate_from_frequencies(freq_dict)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(period, fontweight="bold")
    ax.axis("off")

fig.suptitle(
    "Word Clouds per Period (TF-IDF weighted)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("wordclouds.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → wordclouds.png")

# ── 4e Cosine similarity heatmap ─────────────────────────────
sim_matrix = cosine_similarity(tfidf_matrix)
sim_df     = pd.DataFrame(sim_matrix,
                           index=period_names,
                           columns=period_names)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(sim_df, annot=True, fmt=".3f",
            cmap="YlOrRd", linewidths=0.5,
            ax=ax, vmin=0, vmax=1)
ax.set_title("Cosine Similarity Between Periods (TF-IDF)",
             fontweight="bold")
plt.tight_layout()
plt.savefig("tfidf_similarity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → tfidf_similarity_heatmap.png")

# ── 4f TF-IDF PCA (per-document, for comparison) ─────────────
print("\nFitting per-document TF-IDF for PCA...")
vec_pca  = TfidfVectorizer(max_features=10_000, min_df=3, sublinear_tf=True)
X_docs   = vec_pca.fit_transform(df["clean_abstract"])
svd      = TruncatedSVD(n_components=2, random_state=42)
X_2d     = svd.fit_transform(X_docs)
ev_tfidf = svd.explained_variance_ratio_

df["tfidf_pc1"] = X_2d[:, 0]
df["tfidf_pc2"] = X_2d[:, 1]

print(f"TF-IDF PCA explained variance: "
      f"PC1={ev_tfidf[0]*100:.2f}%  PC2={ev_tfidf[1]*100:.2f}%")


# ============================================================
# STEP 5 — SENTENCE EMBEDDINGS (SBERT)
# ============================================================

print("\n" + "="*60)
print("STEP 5 — SBERT Embeddings")
print("="*60)

CACHE_FILE  = "sbert_embeddings.npy"
MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 256
SAMPLE_SIZE = None   # set e.g. 50_000 for a fast first run

# ── 5a Stratified sample (optional) ─────────────────────────
if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    df_embed = (
        df.groupby("year", group_keys=False)
          .apply(lambda g: g.sample(
              min(len(g), max(1, int(SAMPLE_SIZE * len(g) / len(df)))),
              random_state=42))
    ).reset_index(drop=True)
    print(f"Stratified sample: {len(df_embed):,} abstracts")
else:
    df_embed = df.copy()
    print(f"Full corpus: {len(df_embed):,} abstracts")

# ── 5b Load or encode ────────────────────────────────────────
embeddings = None
if os.path.exists(CACHE_FILE):
    print(f"Loading cached embeddings from {CACHE_FILE}...")
    embeddings = np.load(CACHE_FILE)
    if embeddings.shape[0] != len(df_embed):
        print("Cache mismatch — re-encoding.")
        os.remove(CACHE_FILE)
        embeddings = None
    else:
        print(f"Loaded: {embeddings.shape}")

if embeddings is None:
    from sentence_transformers import SentenceTransformer
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Encoding abstracts (10-40 min on CPU, 2-5 min on GPU)...")
    embeddings = model.encode(
        df_embed["abstract"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(CACHE_FILE, embeddings)
    print(f"Saved to {CACHE_FILE}  shape={embeddings.shape}")

print(f"Embedding matrix: {embeddings.shape}")

# ── 5c PCA on embeddings ─────────────────────────────────────
print("Running PCA on embeddings...")
pca     = PCA(n_components=2, random_state=42)
emb_2d  = pca.fit_transform(embeddings)
ev_emb  = pca.explained_variance_ratio_

df_embed["emb_pc1"] = emb_2d[:, 0]
df_embed["emb_pc2"] = emb_2d[:, 1]

print(f"SBERT PCA explained variance: "
      f"PC1={ev_emb[0]*100:.2f}%  PC2={ev_emb[1]*100:.2f}%")

# Per-year centroids
centroids_emb = (
    df_embed.groupby("year")[["emb_pc1", "emb_pc2"]]
    .mean().reset_index().sort_values("year")
)

# ── Helper: confidence ellipse ───────────────────────────────
def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    if len(x) < 3:
        return
    cov     = np.cov(x, y)
    pearson = cov[0,1] / (np.sqrt(cov[0,0]) * np.sqrt(cov[1,1]) + 1e-9)
    ellipse = Ellipse(
        (0, 0),
        width=np.sqrt(1 + pearson) * 2,
        height=np.sqrt(1 - pearson) * 2,
        **kwargs
    )
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(np.sqrt(cov[0,0]) * n_std,
                     np.sqrt(cov[1,1]) * n_std)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# ── 5d Four-panel comparison figure ─────────────────────────
# Re-run TF-IDF PCA on df_embed subset for fair comparison
vec_cmp   = TfidfVectorizer(max_features=10_000, min_df=3, sublinear_tf=True)
X_tfidf_c = vec_cmp.fit_transform(df_embed["clean_abstract"])
svd_cmp   = TruncatedSVD(n_components=2, random_state=42)
X_t2d     = svd_cmp.fit_transform(X_tfidf_c)
ev_t      = svd_cmp.explained_variance_ratio_

df_embed["tfidf_pc1"] = X_t2d[:, 0]
df_embed["tfidf_pc2"] = X_t2d[:, 1]

centroids_tfidf = (
    df_embed.groupby("year")[["tfidf_pc1", "tfidf_pc2"]]
    .mean().reset_index().sort_values("year")
)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(
    "TF-IDF vs SBERT: Temporal Drift Comparison",
    fontsize=15, fontweight="bold"
)

def draw_trajectory(ax, centroids, xc, yc, label_col="year"):
    for i in range(len(centroids) - 1):
        ax.annotate(
            "",
            xy=(centroids[xc].iloc[i+1], centroids[yc].iloc[i+1]),
            xytext=(centroids[xc].iloc[i], centroids[yc].iloc[i]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0)
        )
    for _, row in centroids.iterrows():
        if int(row[label_col]) % 5 == 0:
            ax.annotate(
                str(int(row[label_col])),
                (row[xc], row[yc]),
                fontsize=8, fontweight="bold",
                xytext=(4, 4), textcoords="offset points"
            )

# Panel 1 — SBERT continuous colour
ax = axes[0, 0]
sc = ax.scatter(df_embed["emb_pc1"], df_embed["emb_pc2"],
                c=df_embed["year"], cmap="plasma",
                alpha=0.12, s=2, linewidths=0)
plt.colorbar(sc, ax=ax, label="Year")
draw_trajectory(ax, centroids_emb, "emb_pc1", "emb_pc2")
ax.set_title(f"SBERT PCA  "
             f"(PC1={ev_emb[0]*100:.1f}%  PC2={ev_emb[1]*100:.1f}%)",
             fontweight="bold")
ax.set_xlabel(f"PC1 ({ev_emb[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev_emb[1]*100:.1f}%)")

# Panel 2 — SBERT by period + ellipses
ax = axes[0, 1]
for period in PERIOD_LABELS:
    mask   = df_embed["period"].astype(str) == period
    subset = df_embed[mask]
    color  = PERIOD_COLORS[period]
    ax.scatter(subset["emb_pc1"], subset["emb_pc2"],
               c=color, alpha=0.12, s=2, linewidths=0)
    confidence_ellipse(
        subset["emb_pc1"].values, subset["emb_pc2"].values,
        ax, n_std=1.5, edgecolor=color, facecolor=color,
        alpha=0.12, linewidth=2, linestyle="--"
    )
    cx, cy = subset["emb_pc1"].mean(), subset["emb_pc2"].mean()
    ax.scatter(cx, cy, c=color, s=80, marker="*",
               zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(period, (cx, cy), fontsize=8, fontweight="bold",
                xytext=(5, 5), textcoords="offset points", color=color)

ax.set_title("SBERT — Periods + Confidence Ellipses", fontweight="bold")
ax.set_xlabel(f"PC1 ({ev_emb[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev_emb[1]*100:.1f}%)")
legend_patches = [
    mpatches.Patch(color=PERIOD_COLORS[p], label=p)
    for p in PERIOD_LABELS
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

# Panel 3 — TF-IDF PCA
ax = axes[1, 0]
sc2 = ax.scatter(df_embed["tfidf_pc1"], df_embed["tfidf_pc2"],
                 c=df_embed["year"], cmap="plasma",
                 alpha=0.12, s=2, linewidths=0)
plt.colorbar(sc2, ax=ax, label="Year")
draw_trajectory(ax, centroids_tfidf, "tfidf_pc1", "tfidf_pc2")
ax.set_title(f"TF-IDF PCA (baseline)  "
             f"(PC1={ev_t[0]*100:.1f}%  PC2={ev_t[1]*100:.1f}%)",
             fontweight="bold")
ax.set_xlabel(f"PC1 ({ev_t[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev_t[1]*100:.1f}%)")

# Panel 4 — Explained variance bar chart
ax = axes[1, 1]
methods   = ["TF-IDF\nPC1", "TF-IDF\nPC2", "SBERT\nPC1", "SBERT\nPC2"]
variances = [ev_t[0]*100, ev_t[1]*100, ev_emb[0]*100, ev_emb[1]*100]
bar_cols  = ["#4878CF", "#4878CF", "#D65F5F", "#D65F5F"]
bars = ax.bar(methods, variances, color=bar_cols, edgecolor="white")
for bar, val in zip(bars, variances):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_title("Explained Variance: TF-IDF vs SBERT", fontweight="bold")
ax.set_ylabel("Explained variance (%)")
ax.set_ylim(0, max(variances) * 1.4)
ax.legend(handles=[
    mpatches.Patch(color="#4878CF", label="TF-IDF"),
    mpatches.Patch(color="#D65F5F", label="SBERT"),
], fontsize=9)

plt.tight_layout()
plt.savefig("embeddings_vs_tfidf_pca.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → embeddings_vs_tfidf_pca.png")

# ── 5e Period similarity heatmaps ────────────────────────────
vec_period  = TfidfVectorizer(max_features=15_000, min_df=2, sublinear_tf=True)
period_texts_clean = [
    " ".join(df_embed[df_embed["period"].astype(str)==p]["clean_abstract"])
    for p in PERIOD_LABELS
]
tfidf_period_mat = vec_period.fit_transform(period_texts_clean)
tfidf_sim_df     = pd.DataFrame(
    cosine_similarity(tfidf_period_mat),
    index=PERIOD_LABELS, columns=PERIOD_LABELS
)

period_emb_centroids = np.array([
    embeddings[df_embed["period"].astype(str) == p].mean(axis=0)
    for p in PERIOD_LABELS
])
emb_sim_df = pd.DataFrame(
    cosine_similarity(period_emb_centroids),
    index=PERIOD_LABELS, columns=PERIOD_LABELS
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Inter-Period Cosine Similarity: TF-IDF vs SBERT",
    fontsize=13, fontweight="bold"
)
for ax, sim_df, title in zip(
    axes,
    [tfidf_sim_df, emb_sim_df],
    ["TF-IDF Cosine Similarity", "SBERT Cosine Similarity"]
):
    sns.heatmap(sim_df, annot=True, fmt=".3f",
                cmap="YlOrRd", linewidths=0.5,
                ax=ax, vmin=0, vmax=1,
                annot_kws={"size": 11})
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("period_similarity_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → period_similarity_comparison.png")

# ── 5f Year-on-year drift magnitude ──────────────────────────
years_sorted  = sorted(df_embed["year"].unique())
year_centroids = {
    yr: embeddings[df_embed["year"] == yr].mean(axis=0)
    for yr in years_sorted
    if (df_embed["year"] == yr).sum() > 5
}

drift_years, drift_scores = [], []
for i in range(1, len(years_sorted)):
    y0, y1 = years_sorted[i-1], years_sorted[i]
    if y0 in year_centroids and y1 in year_centroids:
        dist = 1 - cosine_similarity(
            year_centroids[y0].reshape(1,-1),
            year_centroids[y1].reshape(1,-1)
        )[0,0]
        drift_years.append(y1)
        drift_scores.append(dist)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(drift_years, drift_scores,
        color="#D65F5F", linewidth=2, marker="o", markersize=5)
ax.fill_between(drift_years, drift_scores, alpha=0.15, color="#D65F5F")

milestones = {
    2012: "AlexNet",
    2017: "Transformers",
    2020: "GPT-3",
}
for yr, label in milestones.items():
    if yr in drift_years:
        idx_m = drift_years.index(yr)
        ax.axvline(yr, color="gray", linestyle="--", alpha=0.6)
        ax.annotate(
            label,
            xy=(yr, drift_scores[idx_m]),
            xytext=(yr+0.2, max(drift_scores)*0.85),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
        )

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Cosine distance (year-on-year)", fontsize=11)
ax.set_title(
    "Year-on-Year Vocabulary Drift — SBERT Embeddings\n"
    "(higher = bigger shift in research language)",
    fontsize=13, fontweight="bold"
)
ax.set_xlim(min(drift_years), max(drift_years))
plt.tight_layout()
plt.savefig("drift_magnitude.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → drift_magnitude.png")

# ── 5g Most representative abstracts per period ──────────────
print("\nMost representative abstracts per period (SBERT):")
for period in PERIOD_LABELS:
    mask  = df_embed["period"].astype(str) == period
    idx   = np.where(mask)[0]
    if len(idx) == 0:
        continue
    vecs      = embeddings[idx]
    centroid  = vecs.mean(axis=0, keepdims=True)
    sims      = cosine_similarity(centroid, vecs)[0]
    top5      = idx[sims.argsort()[::-1][:5]]
    print(f"\n  [{period}]")
    for rank, gi in enumerate(top5, 1):
        print(f"  {rank}. [{df_embed['year'].iloc[gi]}] "
              f"{df_embed['title'].iloc[gi][:75]}")


# ============================================================
# STEP 6 — LDA TOPIC MODELLING
# ============================================================

print("\n" + "="*60)
print("STEP 6 — LDA Topic Modelling")
print("="*60)

# ── 6a Tokenise & build phrases ──────────────────────────────
print("Tokenising for LDA...")
df["tokens"] = df["clean_abstract"].apply(str.split)
df_lda = df[df["tokens"].apply(len) >= 5].copy().reset_index(drop=True)
print(f"Documents with ≥5 tokens: {len(df_lda):,}")

print("Building bigram/trigram phrases...")
bigram_model   = Phrases(df_lda["tokens"], min_count=20, threshold=10)
bigram_phraser = Phraser(bigram_model)
trigram_model  = Phrases(bigram_phraser[df_lda["tokens"]],
                          min_count=10, threshold=8)
trigram_phraser = Phraser(trigram_model)

df_lda["tokens_phrases"] = [
    trigram_phraser[bigram_phraser[t]]
    for t in tqdm(df_lda["tokens"], desc="Phrases")
]

# Show top phrases
all_phrases  = [t for toks in df_lda["tokens_phrases"]
                for t in toks if "_" in t]
top_phrases  = Counter(all_phrases).most_common(20)
print("\nTop 20 detected phrases:")
for phrase, count in top_phrases:
    print(f"  {phrase:<35} {count:,}")

# ── 6b Dictionary & BoW corpus ───────────────────────────────
print("\nBuilding dictionary...")
dictionary = corpora.Dictionary(df_lda["tokens_phrases"])
dictionary.filter_extremes(no_below=5, no_above=0.8)
print(f"Dictionary size: {len(dictionary):,} tokens")

print("Building bag-of-words corpus...")
bow_corpus = [
    dictionary.doc2bow(t)
    for t in tqdm(df_lda["tokens_phrases"], desc="BoW")
]

# ── 6c Train LDA models (k = 10, 20, 30) ────────────────────
LDA_CACHE = "lda_models"
os.makedirs(LDA_CACHE, exist_ok=True)
K_VALUES  = [10, 20, 30]
lda_models = {}

for k in K_VALUES:
    cache_path = os.path.join(LDA_CACHE, f"lda_k{k}")
    if os.path.exists(cache_path):
        print(f"Loading cached LDA k={k}...")
        lda_models[k] = LdaMulticore.load(cache_path)
    else:
        print(f"Training LDA k={k} ...")
        model = LdaMulticore(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=k,
            workers=3,
            passes=15,
            iterations=100,
            chunksize=2000,
            alpha="asymmetric",
            eta="auto",
            random_state=42,
            minimum_probability=0.01,
        )
        model.save(cache_path)
        lda_models[k] = model
        print(f"  Saved → {cache_path}")

# ── 6d Coherence scores ──────────────────────────────────────
print("\nComputing coherence scores...")
coherence_scores = {}
for k in K_VALUES:
    cm = CoherenceModel(
        model=lda_models[k],
        texts=df_lda["tokens_phrases"].tolist(),
        dictionary=dictionary,
        coherence="c_v",
        processes=2,
    )
    coherence_scores[k] = cm.get_coherence()
    print(f"  k={k:2d}  c_v = {coherence_scores[k]:.4f}")

best_k     = max(coherence_scores, key=coherence_scores.get)
best_model = lda_models[best_k]
print(f"\nBest k by coherence: {best_k}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(coherence_scores.keys()),
        list(coherence_scores.values()),
        marker="o", color="#4878CF", linewidth=2, markersize=8)
for k, score in coherence_scores.items():
    ax.annotate(f"{score:.3f}", (k, score),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Number of Topics (k)", fontsize=12)
ax.set_ylabel("c_v Coherence Score", fontsize=12)
ax.set_title("LDA Coherence vs k\n(higher = more interpretable)",
             fontweight="bold")
ax.set_xticks(K_VALUES)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lda_coherence.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → lda_coherence.png")

# ── 6e Topic word distributions ──────────────────────────────
print(f"\nTop 15 words per topic (k={best_k}):")
topic_word_rows = []
for tid in range(best_k):
    words = best_model.show_topic(tid, topn=15)
    print(f"  T{tid:02d}: {' | '.join(w for w,_ in words[:8])}")
    for word, weight in words:
        topic_word_rows.append(
            {"topic": tid, "word": word, "weight": round(weight, 4)}
        )
topic_word_df = pd.DataFrame(topic_word_rows)

# ── UPDATE THESE LABELS after reading your model output ──────
TOPIC_LABELS = {
    0:  "Neural Networks",
    1:  "Reinforcement Learning",
    2:  "Natural Language Processing",
    3:  "Computer Vision",
    4:  "Probabilistic / Bayesian",
    5:  "Graph Learning",
    6:  "Optimisation",
    7:  "Text Classification",
    8:  "Generative Models / GANs",
    9:  "Knowledge Representation",
    10: "Speech / Audio",
    11: "Transfer Learning",
    12: "Attention / Transformers",
    13: "Clustering / Unsupervised",
    14: "Recommender Systems",
    15: "Sequence Models / RNNs",
    16: "Meta / Multi-task Learning",
    17: "Explainability / Fairness",
    18: "Question Answering",
    19: "Time Series / Anomaly",
    20: "Object Detection",
    21: "Embeddings / Representation",
    22: "Federated / Privacy",
    23: "Semantic Parsing",
    24: "Image Segmentation",
    25: "Dialogue Systems",
    26: "Data Augmentation",
    27: "Structured Prediction",
    28: "Active / Semi-supervised",
    29: "Kernel / SVM Methods",
}
for i in range(best_k):
    if i not in TOPIC_LABELS:
        TOPIC_LABELS[i] = f"Topic {i}"

# Bar chart: top-10 words for first 12 topics
n_show = min(12, best_k)
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
axes = axes.flatten()
colors = sns.color_palette("tab20", best_k)

for i in range(n_show):
    data = topic_word_df[topic_word_df["topic"] == i].head(10)
    axes[i].barh(data["word"][::-1], data["weight"][::-1], color=colors[i])
    axes[i].set_title(
        f"T{i}: {TOPIC_LABELS.get(i,'')[:28]}",
        fontsize=9, fontweight="bold"
    )
    axes[i].set_xlabel("Weight", fontsize=8)
    axes[i].tick_params(labelsize=8)

plt.suptitle(f"Top 10 Words per Topic — LDA k={best_k}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("lda_topic_words.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → lda_topic_words.png")

# ── 6f Infer topic distributions for all documents ───────────
print("\nInferring topic distributions...")
topic_dist_matrix = np.zeros((len(df_lda), best_k))
dominant_topics   = []

for i, bow in enumerate(tqdm(bow_corpus, desc="Inferring")):
    dist = best_model.get_document_topics(bow, minimum_probability=0.0)
    for tid, prob in dist:
        topic_dist_matrix[i, tid] = prob
    dominant_topics.append(int(np.argmax(topic_dist_matrix[i])))

df_lda["dominant_topic"] = dominant_topics
df_lda["topic_label"]    = df_lda["dominant_topic"].map(TOPIC_LABELS)

print("\nDominant topic counts:")
for tid, count in df_lda["dominant_topic"].value_counts().sort_index().items():
    print(f"  T{tid:02d} {TOPIC_LABELS.get(tid,''):<30} {count:,}")

# ── 6g Topic prevalence over time ────────────────────────────
years_lda = sorted(df_lda["year"].unique())
topic_time = np.zeros((len(years_lda), best_k))

for i, yr in enumerate(years_lda):
    mask = df_lda["year"] == yr
    if mask.sum() > 0:
        topic_time[i] = topic_dist_matrix[mask].mean(axis=0)

topic_time_df = pd.DataFrame(
    topic_time,
    index=years_lda,
    columns=[TOPIC_LABELS.get(t, f"T{t}") for t in range(best_k)]
)

# Stacked area chart
fig, ax = plt.subplots(figsize=(14, 7))
topic_time_df.plot.area(
    ax=ax, colormap="tab20", alpha=0.85, linewidth=0
)
for yr, label in {2012: "Deep Learning", 2017: "Transformers"}.items():
    if yr in years_lda:
        ax.axvline(yr, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(yr + 0.1, ax.get_ylim()[1] * 0.95,
                label, fontsize=8, color="red", va="top")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Mean topic proportion", fontsize=12)
ax.set_title(
    f"Topic Prevalence Over Time — LDA k={best_k}\n"
    "(stacked area = research focus composition)",
    fontweight="bold", fontsize=13
)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
          fontsize=7, ncol=1)
plt.tight_layout()
plt.savefig("lda_topic_area.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → lda_topic_area.png")

# Top 8 most dynamic topics (highest temporal variance)
top8 = topic_time_df.var().nlargest(8).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
colors8 = sns.color_palette("husl", 8)

for i, topic_name in enumerate(top8):
    ax  = axes[i]
    y   = topic_time_df[topic_name]
    ax.plot(years_lda, y, color=colors8[i], linewidth=2)
    ax.fill_between(years_lda, y, alpha=0.2, color=colors8[i])
    peak_yr  = years_lda[int(np.argmax(y))]
    peak_val = float(np.max(y))
    ax.axvline(peak_yr, color="gray", linestyle=":", alpha=0.7)
    ax.annotate(
        f"Peak: {peak_yr}",
        xy=(peak_yr, peak_val),
        xytext=(5, -12),
        textcoords="offset points",
        fontsize=7, color="gray"
    )
    ax.set_title(topic_name[:35], fontsize=8, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Mean proportion", fontsize=8)
    ax.tick_params(labelsize=7)

plt.suptitle(
    "Top 8 Topics by Temporal Variance\n"
    "(most interesting rise/fall dynamics)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("lda_topic_trends.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → lda_topic_trends.png")

# ── 6h Granularity comparison heatmaps (k=10 vs 20 vs 30) ───
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, k in zip(axes, K_VALUES):
    model_k = lda_models[k]
    period_topic_mat = np.zeros((len(PERIOD_LABELS), k))

    for pi, period in enumerate(PERIOD_LABELS):
        mask        = df_lda["period"].astype(str) == period
        period_bows = [bow_corpus[i] for i in np.where(mask)[0]]
        vecs        = []
        for bow in period_bows:
            dist = model_k.get_document_topics(bow, minimum_probability=0.0)
            vec  = np.zeros(k)
            for tid, prob in dist:
                vec[tid] = prob
            vecs.append(vec)
        if vecs:
            period_topic_mat[pi] = np.array(vecs).mean(axis=0)

    im = ax.imshow(period_topic_mat, aspect="auto",
                   cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(PERIOD_LABELS)))
    ax.set_yticklabels(PERIOD_LABELS, fontsize=9)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"T{i}" for i in range(k)],
                        rotation=90, fontsize=7)
    ax.set_title(
        f"k={k}  coherence={coherence_scores[k]:.3f}",
        fontweight="bold"
    )
    ax.set_xlabel("Topic")
    if k == K_VALUES[0]:
        ax.set_ylabel("Period")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean proportion")

plt.suptitle(
    "Topic Distribution per Period: k=10 vs k=20 vs k=30\n"
    "(darker = topic more dominant in that period)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("lda_granularity_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → lda_granularity_comparison.png")

# ── 6i pyLDAvis ──────────────────────────────────────────────
try:
    print(f"\nGenerating pyLDAvis for k={best_k}...")
    vis = gensimvis.prepare(
        best_model, bow_corpus, dictionary,
        mds="mmds", sort_topics=False
    )
    pyLDAvis.save_html(vis, "lda_interactive.html")
    print("Saved → lda_interactive.html  (open in browser)")
except Exception as e:
    print(f"pyLDAvis skipped: {e}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"\nCorpus          : {len(df):,} AI/ML/NLP papers")
print(f"Year range      : {df['year'].min()}–{df['year'].max()}")
print(f"Best LDA k      : {best_k}  (c_v={coherence_scores[best_k]:.4f})")
print(f"\nTF-IDF PCA      : PC1={ev_tfidf[0]*100:.2f}%  "
      f"PC2={ev_tfidf[1]*100:.2f}%")
print(f"SBERT PCA       : PC1={ev_emb[0]*100:.2f}%  "
      f"PC2={ev_emb[1]*100:.2f}%")

print("\nOutput files:")
files = [
    ("eda_plots.png",                 "6-panel EDA"),
    ("tfidf_top_terms.png",           "TF-IDF top terms per period"),
    ("wordclouds.png",                "Word clouds per period"),
    ("tfidf_similarity_heatmap.png",  "TF-IDF period similarity"),
    ("embeddings_vs_tfidf_pca.png",   "4-panel TF-IDF vs SBERT PCA"),
    ("period_similarity_comparison.png","Heatmap comparison"),
    ("drift_magnitude.png",           "Year-on-year drift"),
    ("lda_coherence.png",             "LDA k selection"),
    ("lda_topic_words.png",           "Topic word bars"),
    ("lda_topic_area.png",            "Stacked area — topic prevalence"),
    ("lda_topic_trends.png",          "Top 8 rising/falling topics"),
    ("lda_granularity_comparison.png","k=10 vs 20 vs 30 heatmap"),
    ("lda_interactive.html",          "Interactive pyLDAvis"),
]
for fname, desc in files:
    status = "✓" if os.path.exists(fname) else "·"
    print(f"  {status}  {fname:<42} {desc}")
