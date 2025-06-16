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
    return mo, pd, plt, re, sns


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
    mo.md(r"""## Salary""")
    return


@app.cell
def _(pd, re):
    def extract_salary_ranges(df, column_name):
        """
        Extract minimum and maximum salaries using NLP-like approach.
        Instead of complex regex, we'll tokenize and analyze numerically.
    
        Parameters:
        df (pandas.DataFrame): The dataframe containing salary data
        column_name (str): The name of the column containing salary strings
    
        Returns:
        tuple: Two lists (min_salaries, max_salaries) containing extracted salary values
        """
    
        def extract_all_numbers_with_context(text: str): #-> List[Tuple[int, str, int, int]]:
            """
            Extract all numbers from text with their context.
            Returns: List of (number_value, context_around_number, start_pos, end_pos)
            """
            if not text:
                return []
        
            # Find all numbers (with commas, decimals)
            number_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+'
        
            numbers_with_context = []
            for match in re.finditer(number_pattern, text):
                num_str = match.group()
                start, end = match.span()
            
                # Get context around the number (10 chars before and after)
                context_start = max(0, start - 10)
                context_end = min(len(text), end + 10)
                context = text[context_start:context_end].lower()
            
                # Convert to integer
                try:
                    number = int(num_str.replace(',', '').replace('.', ''))
                    numbers_with_context.append((number, context, start, end))
                except ValueError:
                    continue
        
            return numbers_with_context
    
        def classify_number_as_salary(number: int, context: str): #-> Tuple[int, float]:
            """
            Classify if a number represents a salary and return confidence score.
            Returns: (adjusted_number, confidence_score)
            """
            confidence = 0.0
            adjusted_number = number
        
            # Rule 1: Handle K multiplier
            if 'k' in context and number < 1000:
                adjusted_number = number * 1000
                confidence += 0.8
        
            # Rule 2: Realistic salary range
            if 10000 <= adjusted_number <= 500000:
                confidence += 0.9
            elif 1000 <= adjusted_number <= 9999:
                # Could be salary in K format
                if 'k' not in context:
                    # Maybe it should be multiplied by 1000
                    test_val = adjusted_number * 1000
                    if 10000 <= test_val <= 500000:
                        adjusted_number = test_val
                        confidence += 0.6
            elif adjusted_number < 1000:
                # Very likely needs K multiplier
                test_val = adjusted_number * 1000
                if 10000 <= test_val <= 500000:
                    adjusted_number = test_val
                    confidence += 0.7
        
            # Rule 3: Context clues
            salary_keywords = ['salary', 'annum', 'annual', 'yearly', 'pa', 'basic']
            for keyword in salary_keywords:
                if keyword in context:
                    confidence += 0.3
                    break
        
            # Rule 4: Avoid obviously non-salary numbers
            avoid_keywords = ['tips', 'iro', 'comm', 'bens', '3k per annum']
            for keyword in avoid_keywords:
                if keyword in context:
                    confidence -= 0.5
        
            return adjusted_number, confidence
    
        def find_salary_range(text: str): #-> Tuple[Optional[int], Optional[int]]:
            """
            Find the most likely salary range from text.
            """
            if not text or pd.isna(text):
                return None, None
        
            text_lower = text.lower()
        
            # Handle "upto" cases first
            if 'upto' in text_lower or 'up to' in text_lower:
                numbers = extract_all_numbers_with_context(text)
                salary_candidates = []
            
                for num, context, start, end in numbers:
                    adj_num, confidence = classify_number_as_salary(num, context)
                    if confidence > 0.3:
                        salary_candidates.append((adj_num, confidence))
            
                if salary_candidates:
                    # Take the highest confidence salary
                    best_salary = max(salary_candidates, key=lambda x: x[1])[0]
                    return None, best_salary
        
            # Extract all numbers with context
            numbers = extract_all_numbers_with_context(text)
        
            if not numbers:
                return None, None
        
            # Classify each number as potential salary
            salary_candidates = []
            for num, context, start, end in numbers:
                adj_num, confidence = classify_number_as_salary(num, context)
                if confidence > 0.3:  # Only consider decent confidence
                    salary_candidates.append((adj_num, confidence, start))
        
            if not salary_candidates:
                return None, None
        
            # Sort by position in text (to maintain order)
            salary_candidates.sort(key=lambda x: x[2])
        
            if len(salary_candidates) == 1:
                return salary_candidates[0][0], salary_candidates[0][0]
        
            # Look for range indicators
            range_indicators = ['-', '–', 'to', 'between']
            has_range_indicator = any(indicator in text_lower for indicator in range_indicators)
        
            if has_range_indicator and len(salary_candidates) >= 2:
                # Take first and second highest confidence salaries
                sorted_by_confidence = sorted(salary_candidates, key=lambda x: x[1], reverse=True)
            
                # If we have clear winners, use them
                if len(sorted_by_confidence) >= 2:
                    sal1 = sorted_by_confidence[0][0]
                    sal2 = sorted_by_confidence[1][0]
                    return min(sal1, sal2), max(sal1, sal2)
        
            # Fallback: take the two most confident salaries
            if len(salary_candidates) >= 2:
                sorted_by_confidence = sorted(salary_candidates, key=lambda x: x[1], reverse=True)
                sal1 = sorted_by_confidence[0][0]
                sal2 = sorted_by_confidence[1][0]
                return min(sal1, sal2), max(sal1, sal2)
        
            return salary_candidates[0][0], salary_candidates[0][0]
    
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
    df_train[df_train['Id'] == 34781787]
    return


@app.cell
def _(df_train):
    df_train.describe()
    return


@app.cell
def _(df_train, plt, sns):
    salary_min_mean = 159595.28
    salary_max_mean = 191523.37

    df_train['salary_min'].fillna(salary_min_mean, inplace=True)
    df_train['salary_min'].fillna(salary_max_mean, inplace=True)

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


if __name__ == "__main__":
    app.run()
