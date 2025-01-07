# About Dataset

## Data Source
This dataset is a synthetic version inspired by the original Credit Risk dataset on Kaggle and enriched with additional variables based on Financial Risk for Loan Approval data. SMOTENC was used to simulate new data points to enlarge the instances. The dataset is structured for both categorical and continuous features.

## Metadata
| Column | Description | Type |
| --------------- | --------------- | --------------- |
|person_age |Age of the person | Float |
|person_gender	| Gender of the person | Categorical|
|person_education	| Highest education level | Categorical|
|person_income	|  Annual income |Float|
|person_emp_exp	| Years of employment experience | Integer|
|person_home_ownership	| Home ownership status (e.g., rent, own, mortgage)| Categorical|
|loan_amnt	| Loan amount requested | Float|
|loan_intent | Purpose of the loan |Categorical|
|loan_int_rate	| Loan interest rat |	Float|
|loan_percent_income | Loan amount as a percentage of annual income | Float|
|cb_person_cred_hist_length | Length of credit history in years | Float|
|credit_score | Credit score of the person | Integer|
|previous_loan_defaults_on_file | Indicator of previous loan defaults | Categorical|
|loan_status (target variable) | Loan approval status: 1 = approved; 0 = rejected | Integer|


## Data Usage
The dataset can be used for multiple purposes:
- Exploratory Data Analysis (EDA): Analyze key features, distribution patterns, and relationships to understand credit risk factors.
- Classification: Build predictive models to classify the loan_status variable (approved/not approved) for potential applicants.
- Regression: Develop regression models to predict the credit_score variable based on individual and loan-related attributes.
- Mind the data issue from the original data, such as the instance > 100-year-old as age.

This dataset provides a rich basis for understanding financial risk factors and simulating predictive modeling processes for loan approval and credit scoring.

