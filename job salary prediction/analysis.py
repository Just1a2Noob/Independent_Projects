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
    return mo, np, pd, plt, re, sns


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
def _(mo):
    mo.md(r"""## Salary Preprocessing""")
    return


@app.cell
def _(pd, re):
    from typing import List, Tuple, Optional

    def extract_salary_ranges(df, column_name, convert_to_annual=True):
        """
        Extract and normalize salary ranges handling different time periods.
    
        Parameters:
        df (pandas.DataFrame): The dataframe containing salary data
        column_name (str): The name of the column containing salary strings
        convert_to_annual (bool): Whether to convert all values to annual equivalents
    
        Returns:
        tuple: Two lists (min_salaries, max_salaries) containing extracted salary values
        """
    
        def detect_time_period(text: str) -> str:
            """Detect the time period from context clues."""
            text_lower = text.lower()
        
            if any(word in text_lower for word in ['/hr', 'per hour', 'hourly', 'ph']):
                return 'hourly'
            elif any(word in text_lower for word in ['/week', 'per week', 'weekly', 'pw']):
                return 'weekly'
            elif any(word in text_lower for word in ['/month', 'per month', 'monthly', 'pm']):
                return 'monthly'
            elif any(word in text_lower for word in ['/year', 'per year', '/annum', 'annum', 'annual', 'pa']):
                return 'annual'
            else:
                # Default logic based on number size
                numbers = re.findall(r'\d+\.?\d*', text)
                if numbers:
                    avg_num = sum(float(n) for n in numbers) / len(numbers)
                    if avg_num < 50:
                        return 'hourly'
                    elif avg_num < 2000:
                        return 'weekly'
                    elif avg_num < 10000:
                        return 'monthly'
                    else:
                        return 'annual'
                return 'unknown'
    
        def convert_to_annual_salary(amount: float, period: str) -> int:
            """Convert amount to annual equivalent."""
            if not convert_to_annual:
                return int(amount)
            
            conversion_factors = {
                'hourly': 2080,    # 40 hours/week * 52 weeks
                'weekly': 52,      # 52 weeks/year
                'monthly': 12,     # 12 months/year
                'annual': 1,       # Already annual
                'unknown': 1       # Assume annual
            }
        
            return int(amount * conversion_factors.get(period, 1))
    
        def extract_numbers_with_decimals(text: str) -> List[float]:
            """Extract all numbers including decimals."""
            # Find numbers with optional decimals
            pattern = r'\d+\.?\d*'
            numbers = re.findall(pattern, text)
            return [float(n) for n in numbers if float(n) > 0]
    
        def find_salary_range(text: str) -> Tuple[Optional[int], Optional[int]]:
            """Extract salary range from text."""
            if not text or pd.isna(text):
                return None, None
        
            # Detect time period
            period = detect_time_period(text)
        
            # Extract all numbers
            numbers = extract_numbers_with_decimals(text)
        
            if not numbers:
                return None, None
        
            # Filter out obviously wrong numbers (like years, percentages)
            if period == 'annual':
                # For annual, keep numbers > 1000
                filtered_numbers = [n for n in numbers if n >= 1000 and n <= 500000]
            elif period == 'monthly':
                # For monthly, keep reasonable monthly salaries
                filtered_numbers = [n for n in numbers if n >= 100 and n <= 50000]
            elif period == 'weekly':
                # For weekly, keep reasonable weekly amounts
                filtered_numbers = [n for n in numbers if n >= 50 and n <= 5000]
            elif period == 'hourly':
                # For hourly, keep reasonable hourly rates
                filtered_numbers = [n for n in numbers if n >= 3 and n <= 100]
            else:
                filtered_numbers = numbers
        
            if not filtered_numbers:
                return None, None
        
            # Determine min and max
            if len(filtered_numbers) == 1:
                min_val = max_val = filtered_numbers[0]
            else:
                # Look for range indicators
                text_lower = text.lower()
                has_range = any(indicator in text_lower for indicator in ['-', '–', 'to', 'from'])
            
                if has_range:
                    # Take smallest and largest as range
                    min_val = min(filtered_numbers)
                    max_val = max(filtered_numbers)
                else:
                    # If no clear range, take first two numbers
                    min_val = filtered_numbers[0]
                    max_val = filtered_numbers[1] if len(filtered_numbers) > 1 else filtered_numbers[0]
        
            # Convert to annual equivalents
            min_annual = convert_to_annual_salary(min_val, period)
            max_annual = convert_to_annual_salary(max_val, period)
        
            return min_annual, max_annual
    
        # Process each row
        min_salaries = []
        max_salaries = []
    
        for salary_str in df[column_name]:
            min_sal, max_sal = find_salary_range(salary_str)
            min_salaries.append(min_sal)
            max_salaries.append(max_sal)
    
        return min_salaries, max_salaries
    return (extract_salary_ranges,)


@app.cell
def _(df_train, extract_salary_ranges):
    # NLP approach
    min_sal, max_sal = extract_salary_ranges(df_train, 'SalaryRaw')
    df_train['salary_min'] = min_sal
    df_train['salary_max'] = max_sal
    df_train[['Id', 'salary_min', 'salary_max']]
    return


@app.cell
def _(df_train):
    df_train[['salary_min', 'salary_max']].isnull().sum()
    return


@app.cell
def _(df_train):
    df_train.describe()
    return


@app.cell
def _(df_train, plt, sns):
    salary_min_mean = 159595.28
    salary_max_mean = 191523.37

    df_train.dropna(subset=['salary_min'], how='any', axis=0, inplace=True)

    # Plot histograms
    plt.figure(figsize=(10, 5))
    sns.histplot(df_train['salary_min'], color='blue', label='Min Salary', kde=True)
    sns.histplot(df_train['salary_max'], color='red', label='Max Salary', kde=True)

    # Add average lines
    plt.axvline(salary_min_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Avg Source ({salary_min_mean:.2f})')
    plt.axvline(salary_max_mean, color='red', linestyle='dashed', linewidth=2, label=f'Avg Plagiarism ({salary_max_mean:.2f})')

    # Final touches
    plt.title('SalaryRaw Distribution')
    plt.xlabel('# Salary in thousands')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Clustering salary based on category""")
    return


@app.cell
def _(df_train):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import OneHotEncoder

    # Encoding categories
    enc = OneHotEncoder(sparse_output=False)
    enc_category = enc.fit_transform(df_train[['Category']])

    kmeans = KMeans().fit_predict(df_train[['salary_min', 'salary_max']])
    return (kmeans,)


@app.cell
def _(kmeans, np):
    unique, counts = np.unique(kmeans, return_counts=True)
    print(np.asarray((unique, counts)).T)
    return


@app.cell
def _(df_train, kmeans):
    df_train['cluster'] = kmeans
    df_train[['Category', 'cluster', 'SalaryNormalized']]
    return


@app.cell
def _(df_train):
    df_train.Category.unique()
    return


@app.cell
def _(df_train):
    df_train['salary_mean'] = (df_train['salary_min'] + df_train['salary_max'])/2
    df_train['SalaryRaw']
    return


if __name__ == "__main__":
    app.run()
