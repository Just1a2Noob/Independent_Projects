import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    return mo, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Initial Setup""")
    return


@app.cell
def _(pd):
    df_train = pd.read_csv("Train_rev1.csv")
    df_train
    return (df_train,)


@app.cell
def _(pd):
    df_val = pd.read_csv("Valid_rev1.csv")
    df_val
    return


@app.cell
def _(pd):
    df_test = pd.read_csv("Test_rev1.csv")
    df_test
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exploratory Data Analysis (EDA)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Null Values""")
    return


@app.cell
def _(df_train):
    df_train.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Classes in each in column""")
    return


@app.cell
def _(df_train):
    df_train['Title'].value_counts()
    return


@app.cell
def _(df_train):
    df_train['Category'].value_counts()
    return


@app.cell
def _(df_train):
    df_train['SourceName'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Graphs""")
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
def _(df_train):
    # Abnormal data rows for description
    df_train[df_train['FullDescription_len'] >= 1000]
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
def _(df_train, mo):
    mo.ui.data_explorer(df_train)
    return


@app.cell
def _(df_train):
    df_train[df_train['SalaryNormalized'] >= 120000]
    return


if __name__ == "__main__":
    app.run()
