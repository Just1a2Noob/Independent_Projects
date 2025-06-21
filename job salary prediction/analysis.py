import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Disclaimer

    Below is my own work, and have been found unsuccessful. So I will provide a good solution from kaggle: [Job Salary Prediction By JST](https://www.kaggle.com/code/jillanisofttech/job-salary-prediction-by-jst#Now-Check-missing-values). 

    What makes the solution interesting is how it handles missing values. Below is the block of code that in my opinion is the key to its great solution: 

    ```python
    for label,content in df_train.items():
        if not pd.api.types.is_numeric_dtype(content):
            # Add binary column to indicate whether sample had missing value
            df_train[label+"is_missing"]=pd.isnull(content)
            # Turn categories into numbers and add+1
            df_train[label] = pd.Categorical(content).codes+1
    ```

    What this code essentially does it check whether the content is a numeric or not, if its not a numeric it processes it. Then it creates a new dataframe that acts as *'memory'* which contains either true or false values. Why is this important? Because sometimes the fact that a value is missing is itself meaningful information. For instance, if you're analyzing survey data, people who didn't answer a particular question might share certain characteristics.

    This particular line of thinking did not occur to me at all, I had the assumption of all missing data, has no information.
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""# Initial Setup""")
    return


@app.cell
def _(pd):
    df_train = pd.read_csv("Train_rev1.csv")
    df_val = pd.read_csv("Valid_rev1.csv")
    df_test = pd.read_csv("Test_rev1.csv")
    return (df_train,)


@app.cell
def _(df_train):
    df_train
    return


@app.cell
def _(df_train):
    df_train.isnull().sum()
    return


@app.cell
def _(mo):
    mo.md(r"""# Explaratory Data Analysis (EDA)""")
    return


@app.cell
def _(df_train, plt, sns):
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
    return


@app.cell
def _(df_train, plt, sns):
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
    return


@app.cell
def _(mo):
    mo.md(r"""# Preprocessing Data""")
    return


@app.cell
def _(df_train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    import re
    import string


    df_train.fillna({'Title': "", "ContractType": "None", "ContractTime": "None"}, inplace=True)

    tfid = TfidfVectorizer()

    # Text preprocessing function
    def preprocess_text(text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    vectorizer = TfidfVectorizer(
        max_features=1000,          # Limit vocabulary size
        min_df=1,                   # Ignore terms appearing in less than 1 document
        max_df=0.8,                 # Ignore terms appearing in more than 80% of documents
        stop_words='english',       # Remove English stop words
        ngram_range=(1, 2),         # Use unigrams and bigrams
        lowercase=True,
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
    )

    processed_texts = [preprocess_text(text) for text in df_train['Title']]
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return KMeans, OneHotEncoder, tfidf_matrix, vectorizer


@app.cell
def _(df_train, np, pd, tfidf_matrix, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()

    # Get top features for documents (works with DataFrame or list)
    def get_top_features_df(texts, tfidf_matrix, feature_names, n_features=5, text_column=None):
        """
        Get top TF-IDF features for documents

        Args:
            texts: DataFrame with text column or list of texts
            tfidf_matrix: TF-IDF matrix (sparse or dense)
            feature_names: Feature names from vectorizer
            n_features: Number of top features to return
            text_column: Column name if texts is DataFrame

        Returns:
            DataFrame with document index, text preview, and top features
        """
        if hasattr(tfidf_matrix, 'toarray'):
            tfidf_array = tfidf_matrix.toarray()
        else:
            tfidf_array = tfidf_matrix

        results = []

        # Handle DataFrame input
        if isinstance(texts, pd.DataFrame):
            if text_column is None:
                # Assume first column contains text
                text_column = texts.columns[0]
            text_list = texts[text_column].tolist()
            indices = texts.index.tolist()
        else:
            # Handle list input
            text_list = texts
            indices = list(range(len(texts)))

        for i, (idx, text) in enumerate(zip(indices, text_list)):
            doc_scores = tfidf_array[i]
            top_indices = np.argsort(doc_scores)[::-1][:n_features]

            top_features = []
            for feature_idx in top_indices:
                if doc_scores[feature_idx] > 0:
                    top_features.append({
                        'feature': feature_names[feature_idx],
                        'score': doc_scores[feature_idx]
                    })

            results.append({
                'doc_index': idx,
                'text_preview': str(text)[:50] + '...' if len(str(text)) > 50 else str(text),
                'top_features': top_features,
                'feature_count': len(top_features)
            })

        return pd.DataFrame(results)

    get_top_features_df(df_train['Category'], tfidf_scores, feature_names)
    return


@app.cell
def _(OneHotEncoder, df_train):
    ohe = OneHotEncoder()
    ohe_category = ohe.fit_transform(df_train[['Category']])
    ohe_con_type = ohe.fit_transform(df_train[['ContractType']])
    ohe_con_time = ohe.fit_transform(df_train[['ContractTime']])
    return ohe_category, ohe_con_time, ohe_con_type


@app.cell
def _(KMeans, ohe_category, ohe_con_time, ohe_con_type, tfidf_matrix):
    from sklearn.metrics import silhouette_score
    from scipy.sparse import hstack

    # Concatenate all sparse matrices horizontally
    X = hstack([tfidf_matrix, ohe_category, ohe_con_type, ohe_con_time])

    clusterer = KMeans(n_clusters=8, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    # 6 = 0.718
    # 10 = 0.01407
    return clusterer, silhouette_avg


@app.cell
def _(silhouette_avg):
    print(silhouette_avg)
    return


@app.cell
def _(clusterer):
    clusterer.cluster_centers_
    return


if __name__ == "__main__":
    app.run()
