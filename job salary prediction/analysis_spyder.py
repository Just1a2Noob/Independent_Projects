import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re
import nltk 
from nltk.corpus import stopwords
import spacy
import subprocess
import sys
import contractions
# %%

df_train = pd.read_csv("Train_rev1.csv")
df_val = pd.read_csv("Valid_rev1.csv")
df_test = pd.read_csv("Test_rev1.csv")
# %%

nan_values = pd.DataFrame(df_train.isnull().sum())

# %%

# Creates a text length distribution graph

# Calculate text lengths
df_train['FullDescription_len'] = df_train['FullDescription'].apply(lambda x: len(x.split()))

# Plot histograms
plt.figure(figsize=(10, 5))
sns.histplot(df_train['FullDescription_len'], color='blue', label='Description Text', kde=True)

# Calculate averages
avg_len = df_train['FullDescription_len'].mean()

# Add average lines
plt.axvline(avg_len, color='blue', linestyle='dashed', linewidth=2, label=f'Avg Description ({avg_len:.2f})')

# Final touches
plt.title('Text Length Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# %%

# Creates Normalized Salary Distribution

# Plot histograms
plt.figure(figsize=(10, 5))
sns.histplot(df_train['SalaryNormalized'], color='blue', label='Normalized Salary', kde=True)

# Calculate averages
avg_salary = df_train['SalaryNormalized'].mean()

# Add average lines
plt.axvline(avg_salary, color='blue', linestyle='dashed', linewidth=2, label=f'Avg Normalized Salary ({avg_salary:.2f})')

# Final touches
plt.title('Normalized Salary Distribution')
plt.xlabel('Normalized Salary')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# %%
# Download the English model
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

nltk.download('stopwords')

# Load SpaCy Model for Advanced NLP Tasks 
nlp = spacy.load("en_core_web_sm")

# %%

# Lowercase 
def to_lowercase(text):
        return text.lower()

# Remove Punctuation 
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Remove Special Characters and Numbers
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z\s]', '', text)

# Remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

# Remove Extra Spaces
def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

# Expanding Contractions 
def expand_contractions(text):
    return contractions.fix(text)

# Stop Words Removal 
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = text.split()
    return " ".join([word for word in tokens if word not in stop_words])

# Tokenization and Lemmatization (Using SpaCy)
def spacy_tokenization_lemmatization(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def full_pipeline(text):
    text = to_lowercase(text)
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = remove_urls(text)
    text = remove_extra_spaces(text)
    text = remove_stopwords(text)
    tokens = spacy_tokenization_lemmatization(text)
    text = " ".join(tokens)
    return text

# %%

df_train['processed_description'] = df_train['FullDescription'].apply(full_pipeline)











