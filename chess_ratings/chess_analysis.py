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
    return mo, pd


@app.cell
def _(pd):
    train = pd.read_csv('training_data.csv')
    test = pd.read_csv('test_data.csv')
    val = pd.read_csv('cross_validation_dataset.csv')
    return test, train


@app.cell
def _(train):
    train.head()
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
    def player_stats(id):
        white_player_wr = train.loc[(train['White Player #'] == id) & (train['Score'] == 1)].size
        black_player_wr = train.loc[(train['Black Player #'] == id) & (train['Score'] == 1)].size
        draw_player = train.loc[(train['Black Player #'] == id) & (train['Score'] == 0.5)].size + train.loc[(train['White Player #'] == id) & (train['Score'] == 0.5)].size
        total_matches = white_player_wr + black_player_wr + draw_player
        points = white_player_wr + black_player_wr + draw_player*0.5

        print(f"Statistics of player ID {id} \n")
        print(f"Total matches: {total_matches}\n")
        print(f"Winrate as white: {white_player_wr/total_matches}\n")
        print(f"Winrate as black: {black_player_wr/total_matches}\n")
        print(f"Total points: {points}")
        print(f"Points/Total Matches: {points/total_matches}")
    return (player_stats,)


@app.cell
def _(player_stats):
    player_stats(7321)
    return


@app.cell
def _(mo):
    mo.md(r"""# Machine Learning Models""")
    return


@app.cell
def _(test, train):
    from sklearn.svm import SVC
    from sklearn.metrics import matthews_corrcoef

    X = ['White Player #', 'Black Player #']
    Y = ['Score']

    clf = SVC(random_state=42)

    clf.fit(X=train[X], y=train[Y])
    y_pred = clf.predict(test[X])
    MCC = matthews_corrcoef(y_pred, test[Y])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
