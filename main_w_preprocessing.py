## --- IMPORTS
# combined_analysis.py
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

# NLTK & sentiment
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# stats
from scipy import stats

# NLTK downloads (only once) 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

## --- READ IN DATA
DATA_DIR = Path("Dataset")
df_fake = pd.read_csv(DATA_DIR / "Fake.csv")
df_true = pd.read_csv(DATA_DIR / "True.csv")

# Label datasets (single canonical column: label: 0=fake, 1=true)
df_fake = df_fake.copy()
df_true = df_true.copy()
df_fake['label'] = 0
df_true['label'] = 1

## --- PREPROCESSING DATA

# Combine, shuffle, dedupe, drop missing (do this early to avoid duplicated computation)
merge_csv = pd.concat([df_fake, df_true], ignore_index=True)
merge_csv = merge_csv.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop duplicates on title+text and drop rows missing either title or text
if 'title' in merge_csv.columns and 'text' in merge_csv.columns:
    merge_csv.drop_duplicates(subset=["title", "text"], inplace=True)
    merge_csv.dropna(subset=["title", "text"], inplace=True)
else:
    # If titles aren't present, at least dedupe on text
    merge_csv.drop_duplicates(subset=["text"], inplace=True)
    merge_csv.dropna(subset=["text"], inplace=True)

print(f"After merge/dedupe: {merge_csv.shape[0]} rows")

# Helper function: convert date & extract month/year
if 'date' in merge_csv.columns:
    merge_csv['date'] = pd.to_datetime(merge_csv['date'], errors='coerce')
    merge_csv['year'] = merge_csv['date'].dt.year
    merge_csv['month'] = merge_csv['date'].dt.month
else:
    merge_csv['year'] = np.nan
    merge_csv['month'] = np.nan

# Preprocessing for tokens (from new code)
stop_words = set(stopwords.words('english'))
punct = set(string.punctuation)

def preprocess_text_tokens(text: str):
    """Lowercase, tokenize, remove punctuation/stopwords, keep alpha tokens."""
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# Add token columns (title and text) — safe even if title missing
merge_csv['title_tokens'] = merge_csv['title'].apply(preprocess_text_tokens) if 'title' in merge_csv.columns else [[]]*len(merge_csv)
merge_csv['text_tokens'] = merge_csv['text'].apply(preprocess_text_tokens)

# Add clean_text string for ML tools (new code)
merge_csv['clean_text'] = merge_csv['text_tokens'].apply(lambda x: " ".join(x))

# Feature engineering: lengths (new code)
merge_csv['title_length'] = merge_csv['title'].apply(lambda x: len(str(x).split())) if 'title' in merge_csv.columns else 0
merge_csv['text_length'] = merge_csv['text'].apply(lambda x: len(str(x).split()))

# Compute original features (word_count, question marks, negativity, opposition words, pronouns, lexical diversity) ----
# word_count using NLTK tokenization (preserves original script's approach)
merge_csv['word_count'] = merge_csv['text'].apply(lambda x: len(word_tokenize(str(x))))

# question marks
merge_csv['question_marks'] = merge_csv['text'].apply(lambda x: str(x).count('?'))

# sentiment negativity (SIA)
sia = SentimentIntensityAnalyzer()
merge_csv['negativity'] = merge_csv['text'].apply(lambda x: sia.polarity_scores(str(x))['neg'])

# opposition words
opposition_words = {'but', 'however', 'although', 'though', 'yet', 'nevertheless', 'nonetheless', 'despite', 'instead'}
merge_csv['opposition_count'] = merge_csv['text'].apply(lambda x: sum(1 for w in word_tokenize(str(x).lower()) if w in opposition_words))

# pronoun counts (first person vs second+third)
first_person_pronouns = {'i', 'we', 'me', 'us', 'my', 'our', 'mine', 'ours'}
second_third_person_pronouns = {'you', 'he', 'she', 'they', 'him', 'her', 'them', 'your', 'his', 'its', 'their', 'yours', 'theirs'}

def count_pronouns(text):
    first_person_count = 0
    second_third_person_count = 0
    for word in word_tokenize(str(text).lower()):
        if word in first_person_pronouns:
            first_person_count += 1
        elif word in second_third_person_pronouns:
            second_third_person_count += 1
    return pd.Series([first_person_count, second_third_person_count])

merge_csv[['first_person_count', 'second_third_person_count']] = merge_csv['text'].apply(count_pronouns)

# lexical diversity function (optimized from original code)
def lexical_diversity_optimized(text):
    tokens = word_tokenize(str(text).lower())
    words = [w for w in tokens if w.isalpha()]
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

merge_csv['lexical_diversity'] = merge_csv['text'].apply(lexical_diversity_optimized)

# Statistical test used in original script (lexical diversity t-test) 
lex_div_fake = merge_csv[merge_csv['label'] == 0]['lexical_diversity'].dropna()
lex_div_true = merge_csv[merge_csv['label'] == 1]['lexical_diversity'].dropna()
t_statistic, p_value = stats.ttest_ind(lex_div_fake, lex_div_true, equal_var=False)  # safer to use Welch's
print(f"Lexical diversity t-test -> t: {t_statistic:.4f}, p: {p_value:.4f}")
if p_value < 0.05:
    print("Difference in lexical diversity is statistically significant (p < 0.05).")
else:
    print("Difference is not statistically significant (p >= 0.05).")

# Prepare for plotting/analysis: create 'is_fake' if some functions expect it (1=fake)
merge_csv['is_fake'] = merge_csv['label'].apply(lambda x: 1 if x == 0 else 0)  # optional: matches original where 1 = fake

# Correlation matrix (combines original and new numeric features)
numerical_features = [
    'is_fake', # numeric target for correlation (1==fake in this column)
    'word_count',
    'negativity',
    'first_person_count',
    'second_third_person_count',
    'lexical_diversity',
    'title_length',
    'text_length',
    'year',
    'month'
]
# keep only features that actually exist
numerical_features = [f for f in numerical_features if f in merge_csv.columns]
df_corr = merge_csv[numerical_features].copy()
corr_matrix = df_corr.corr()
print("Correlation matrix:")
print(corr_matrix.round(2))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Linguistic & Meta Features')
plt.show()

# Subject categorical analysis (if present)
if 'subject' in merge_csv.columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(y='subject', hue='is_fake', data=merge_csv, palette='coolwarm', order=merge_csv['subject'].value_counts().index)
    plt.title('Relationship Between Article Subject and Authenticity')
    plt.xlabel('Number of Articles')
    plt.ylabel('Subject')
    plt.legend(title='Category', labels=['True News (0)', 'Fake News (1)'])
    plt.tight_layout()
    plt.show()

# Time-series analysis (if date present)
if 'date' in merge_csv.columns:
    df_plot = merge_csv.set_index('date')
    fake_counts_by_month = df_plot[df_plot['is_fake'] == 1]['is_fake'].resample('M').size()
    true_counts_by_month = df_plot[df_plot['is_fake'] == 0]['is_fake'].resample('M').size()

    plt.figure(figsize=(15, 7))
    fake_counts_by_month.plot(label='Fake News', marker='o', linestyle='--')
    true_counts_by_month.plot(label='True News', marker='o', linestyle='-')
    plt.title('Article Publication Trends Over Time')
    plt.ylabel('Number of Articles Published')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

## --- VISUALIZATIONS

# Title/text length comparisons (bar plots)
if 'title_length' in merge_csv.columns and 'text_length' in merge_csv.columns:
    avg_title_length = merge_csv.groupby('is_fake')['title_length'].mean()
    avg_text_length = merge_csv.groupby('is_fake')['text_length'].mean()

    print("\nAverage Title Length (by is_fake):")
    print(avg_title_length.rename({0: 'True News', 1: 'Fake News'} if 0 in avg_title_length.index else avg_title_length))

    print("\nAverage Text Length (by is_fake):")
    print(avg_text_length.rename({0: 'True News', 1: 'Fake News'} if 0 in avg_text_length.index else avg_text_length))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Comparison of Average Lengths for Fake vs. True News', fontsize=16)

    # bar plots (Seaborn will pick default colors)
    sns.barplot(x=avg_title_length.index, y=avg_title_length.values, ax=ax1)
    ax1.set_title('Average Title Length')
    ax1.set_ylabel('Number of Words')
    ax1.set_xticklabels(['True News' if i==0 else 'Fake News' for i in avg_title_length.index], rotation=0)

    sns.barplot(x=avg_text_length.index, y=avg_text_length.values, ax=ax2)
    ax2.set_title('Average Text Length')
    ax2.set_ylabel('Number of Words')
    ax2.set_xticklabels(['True News' if i==0 else 'Fake News' for i in avg_text_length.index], rotation=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# merge_csv.to_csv(DATA_DIR / "merged_processed.csv", index=False)
