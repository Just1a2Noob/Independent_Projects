# Dataset Description

The dataset of chess results represents 105 months' worth of actual game-by-game results among 8,631 of the world's top 13,000 chess players, from sometime in the last 12 years. Out of the 8,631 players included in the dataset, only 70% of the games among those players have been included. Therefore, these players actually play with approximately twice the frequency that you see in this dataset.

The players are uniquely identified by ID #’s ranging from 1 to 8,631.  The dataset is divided into a training dataset, representing a consecutive stretch of 100 months of game-by-game results among those top players, and a test dataset, representing the next 5 months of games played among those players (obviously the actual game-by-game results on the test dataset have been withheld). 

You should use training_dataset.csv to train your models. It includes 65,053 rows of data, representing 65,053 distinct games played from months 1 through 100, with the following columns:

Month # (from 1 to 100)

White Player # (from 1 to 8,631)

Black Player # (from 1 to 8,631)

Score (either 0, 0.5, or 1)

“White Player” represents the ID # of the player who had the white pieces, and “Black Player” represents the ID # of the player who had the black pieces.  The possible values for Score represent the three possible outcomes of a chess game (1=White wins, 0.5=draw, 0=Black wins). 

In chess, the player with the white pieces gets to move first and therefore has a slight advantage.  For instance, in the 65,053 games listed in the training dataset, White won 32.5% of the games, Black won 23.4% of the games, and 44.1% of the games were drawn (draws are very common among top players)

The test_dataset.csv should be used to frame submissions. It includes 7,809 rows of data, representing 7,809 distinct games played from months 101 through 105, with the following columns:

Month # (from 101 to 105)

White Player # (from 1 to 8,631)

Black Player # (from 1 to 8,631)

Score (either 0, 0.5, or 1)


--- 

Reference: www.kaggle.com/c/chess
