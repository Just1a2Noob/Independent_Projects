import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Initial Setup""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    return math, mo, np, pd


@app.cell
def _(pd):
    train = pd.read_csv('training_data.csv')
    test = pd.read_csv('test_data.csv')
    val = pd.read_csv('cross_validation_dataset.csv')
    return test, train, val


@app.cell
def _(train):
    train
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Problem: Predict the outcome in a game of chess given the ID of both players and the starting position they played?

    Mathematically defined: $P(ID_w, ID_b|Y)$

    Where,

    - $ID_w$: White player ID
    - $ID_b$: Black player ID
    - Y: 1 (white wins), 0.5 (draw), 0 (black wins)

    Things to do: 

    1. Given a player ID and starting position, what is the probability of the player winning? $P(ID_w|Y=1) + P(ID_w \ \text{or} \ ID_b|Y=0.5) + P(ID_b|Y=0)$
    2. How much does position affect the probabilities of $Y$?
    3. How to evaluate the player, regardless of starting position? (Elo Score)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Explaratory Data Analysis (EDA)""")
    return


@app.cell
def _(train):
    white_winrate = train[train['Score'] == 1].size / train.size
    black_winrate = train[train['Score'] == 0].size / train.size
    draw = train[train['Score'] == 0.5].size / train.size

    print(f"White position winrate : {white_winrate}")
    print(f"Black position winrate : {black_winrate}")
    print(f"Draw probability : {draw}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    From the above we can see that White seems to have a significant advantage over black with a significant difference of 10% meaning or white is 1/3rd more likely to win compared to black.

    But surprisingly draw occurs alot, I infer that due to player skill is equal resulting in draws.
    """
    )
    return


@app.cell
def _(train):
    def player_stats(id, df=train):
        # Sees the overall player stats

        white_matches = df.loc[df['White Player #'] == id].size
        white_wins = df.loc[(df['White Player #'] == id) & (df['Score'] == 1)].size

        black_matches = df.loc[df['Black Player #'] == id].size
        black_wins = df.loc[(df['Black Player #'] == id) & (df['Score'] == 1)].size

        draw_matches = df.loc[(df['Black Player #'] == id) & (df['Score'] == 0.5)].size + df.loc[(train['White Player #'] == id) & (df['Score'] == 0.5)].size

        total_matches = white_matches + black_matches + draw_matches

        print(f"Statistics of player ID {id} \n")
        print(f"Total matches: {total_matches}\n")
        print(f"Winrate as white: {white_wins/total_matches}")
        print(f"Matches as white: {white_matches}\n")
        print(f"Winrate as black: {black_wins/total_matches}")
        print(f"Matches as black: {black_matches}\n")
    return


@app.cell
def _(mo):
    mo.md(r"""# Creating The Model""")
    return


@app.cell
def _(pd, train):
    def create_player_rating_df(df, initial_rating=1000):
        # Extract unique player IDs from both white and black player columns
        white_players = set(df['White Player #'].dropna())
        black_players = set(df['Black Player #'].dropna())

        # Combine and get all unique player IDs
        all_players = white_players.union(black_players)

        # Sort for consistent ordering
        unique_players = sorted(list(all_players))

        # Create the rating dataframe
        rating_df = pd.DataFrame({
            'player_id': unique_players,
            'rating': [initial_rating] * len(unique_players)
        })

        return rating_df

    player_ratings = create_player_rating_df(train)
    return (player_ratings,)


@app.cell
def _(math, player_ratings, train):
    def proba_win(p1_rating, p2_rating):
        # Calculate and return the expected score
        return 1.0 / (1 + math.pow(10, (p2_rating - p1_rating) / 400))

    def rating_update(p1_rating, p2_rating, result, K=130):
        # Results are in favor of player 1

        P1 = proba_win(p1_rating, p2_rating)
        P2 = proba_win(p2_rating, p1_rating)

        R1 = round(p1_rating + K*(result - P1))
        R2 = round(p2_rating + K*((1-result) - P2))

        return R1, R2

    def elo_system(df_player=player_ratings, df_matches=train):
        df_player = df_player.copy()
        for idx, row in df_matches.iterrows():
            # ID of player
            p1 = row['White Player #']
            # Rating score of related player
            p1_ratings = df_player.loc[df_player['player_id'] == p1, 'rating'].iloc[0]

            p2 = row['Black Player #']
            p2_ratings = df_player.loc[df_player['player_id'] == p2, 'rating'].iloc[0]

            results = row["Score"]

            R_p1, R_p2 = rating_update(p1_ratings, p2_ratings, results)

            df_player.loc[(df_player['player_id'] == p1), 'rating'] = R_p1
            df_player.loc[(df_player['player_id'] == p2), 'rating'] = R_p2

        return df_player
    return elo_system, proba_win


@app.cell
def _(elo_system):
    elo_rankings = elo_system()
    return (elo_rankings,)


@app.cell
def _(mo):
    mo.md(r"""# Analyzing Model Results""")
    return


@app.cell
def _(elo_rankings):
    elo_rankings[elo_rankings['rating'] > 0]
    return


@app.cell
def _(elo_rankings):
    import seaborn as sns

    sns.displot(elo_rankings, x='rating', kde=True, bins=10)
    return


@app.cell
def _(elo_rankings, mo):
    import altair as alt

    chart = mo.ui.altair_chart(alt.Chart(elo_rankings).mark_point().encode(
        x='player_id',
        y='rating'
    ))
    return (chart,)


@app.cell
def _(chart, mo):
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(mo):
    mo.md(r"""# Evaluating using Test set""")
    return


@app.cell
def _(elo_rankings, proba_win):
    def predict_match(p1, p2, probs=False):
        R_p1 = elo_rankings.loc[elo_rankings['player_id'] == p1,'rating'].iloc[0]
        R_p2 = elo_rankings.loc[elo_rankings['player_id'] == p2, 'rating'].iloc[0]
        P1 = proba_win(R_p1, R_p2)

        if probs == True:
            return P1

        if P1 >= 0.43 and P1 <= 0.53:
            return 0.5
        if P1 > 0.53:
            return 1
        else:
            return 0
    return (predict_match,)


@app.cell
def _(pd, predict_match):
    def evaluate(df):
        df_copy = df.copy()
        values = []

        for i in range(len(df_copy)):
            p1 = df_copy.iloc[i]['White Player #']
            p2 = df_copy.iloc[i]['Black Player #']
            predict = predict_match(p1, p2)
            values.append(predict)

        df_copy['prediction'] = values
        return df_copy

    # Comprehensive accuracy function with detailed output
    def detailed_accuracy(df, true_col, pred_col):
        # Create comparison column
        df_copy = df.copy()
        df_copy['correct'] = df_copy[true_col] == df_copy[pred_col]

        # Calculate metrics
        correct_predictions = df_copy['correct'].sum()
        total_predictions = len(df_copy)
        accuracy = correct_predictions / total_predictions

        # Get unique classes
        all_classes = pd.concat([df_copy[true_col], df_copy[pred_col]]).unique()

        # Per-class accuracy
        class_accuracy = {}
        for cls in all_classes:
            class_mask = df_copy[true_col] == cls
            if class_mask.sum() > 0:  # Avoid division by zero
                class_correct = ((df_copy[true_col] == cls) & (df_copy[pred_col] == cls)).sum()
                class_total = class_mask.sum()
                class_accuracy[cls] = class_correct / class_total

        return {
            'overall_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'class_accuracy': class_accuracy,
            'comparison_df': df_copy
        }
    return detailed_accuracy, evaluate


@app.cell
def _(mo):
    mo.md(r"""## Results""")
    return


@app.cell
def _(evaluate, val):
    eval = evaluate(val)
    eval
    return (eval,)


@app.cell
def _(detailed_accuracy, eval):
    result = detailed_accuracy(eval, 'Score', 'prediction')
    print(f"Overall Accuracy: {result['overall_accuracy']:.4f} ({result['overall_accuracy']*100:.2f}%)")
    print(f"Correct: {result['correct_predictions']}/{result['total_predictions']}")
    print("\nPer-class accuracy:")
    for cls, acc in result['class_accuracy'].items():
        print(f"Class {cls}: {acc:.4f} ({acc*100:.2f}%)")
    print("\n" + "="*60 + "\n")
    return


@app.cell
def _(mo):
    mo.md(r"""# Using Machine Learning""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Preparing Data""")
    return


@app.cell
def _(elo_rankings, pd, train):
    from sklearn.preprocessing import LabelEncoder

    def prepare_data(df, target=None):
        result_df = df.copy()
    
        # Create mapping dictionary from reference dataframe
        id_to_score = elo_rankings.set_index('player_id')['rating'].to_dict()
    
        # Replace all columns that contain player IDs
        for col in result_df.columns:
            # Check if column contains values that exist in our mapping
            if result_df[col].dtype == 'object' or pd.api.types.is_integer_dtype(result_df[col]):
                result_df[col] = result_df[col].map(id_to_score).fillna(result_df[col])

        if target is not None:
            y_discrete = LabelEncoder().fit_transform(target)
            return result_df, y_discrete
        
        return result_df

    df_ml, y = prepare_data(train, train['Score'])
    df_ml.drop('Month #', axis=1, inplace=True)
    return df_ml, prepare_data, y


@app.cell
def _(df_ml):
    df_ml
    return


@app.cell
def _(df_ml, y):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.metrics import matthews_corrcoef

    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=20,          
        min_samples_split=15,      
        min_samples_leaf=10,       
        max_features=None,         
        random_state=42,
        class_weight='balanced'
    )

    clf.fit(
        X=df_ml[['White Player #', 'Black Player #']],
        y=y
    )
    return accuracy_score, classification_report, clf, matthews_corrcoef


@app.cell
def _(
    accuracy_score,
    classification_report,
    clf,
    matthews_corrcoef,
    prepare_data,
    val,
):
    df_test, y_test = prepare_data(val, val['Score'])

    y_pred = clf.predict(df_test[['White Player #', 'Black Player #']])


    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nMatthew's Correlation Coefficient")
    print(matthews_corrcoef(y_test, y_pred))
    return


@app.cell
def _(mo):
    mo.md(r"""# Submission""")
    return


@app.cell
def _(clf, np, prepare_data, test):
    def submit():
        df = prepare_data(test.copy())
        y_pred = clf.predict(df[['White Player #', 'Black Player #']])

        reverse_mapping = {0: 0, 1: 0.5, 2: 1}
        y_cont = np.array([reverse_mapping[label] for label in y_pred])
        test['Score'] = y_cont
        return test
    return (submit,)


@app.cell
def _(submit):
    submit()
    return


if __name__ == "__main__":
    app.run()
