import pandas as pd
from math import log
from sklearn.preprocessing import MinMaxScaler

path = '../data/'
data = pd.read_csv(path + 'new-preprocessed_data.csv', index_col = 0)

data['schedule_date'] = pd.to_datetime(data['schedule_date'])

def calc_expected_score(team_rating, opp_team_rating):
    return 1 / (1 + 10 ** ((opp_team_rating - team_rating) / 400))

def calc_new_rating(team_rating, observed_score, expected_score, k_factor=20):
    return team_rating + k_factor * (observed_score - expected_score)

# Function to apply the corrected Elo rating calculation with MOV multiplier
def calculate_elo_ratings_with_mov(data):
    # Initialize a dictionary to store Elo ratings for each team
    elo_ratings = {}
    
    # Initialize lists to store home and away Elo ratings
    home_elo_ratings = []
    away_elo_ratings = []

    # Sort the data by date to process games in chronological order
    data = data.sort_values(by='schedule_date')
    
    # Get all unique teams
    all_teams = set(data['team_home'].unique()).union(set(data['team_away'].unique()))

    # Set initial Elo rating for all teams
    for team in all_teams:
        elo_ratings[team] = 1500

    # Iterate through each row (game) in the dataset
    for index, row in data.iterrows():
        # Get the team names
        home_team = row['team_home']
        away_team = row['team_away']

        # Reset Elo ratings to 1500 at the start of a new season for all teams
        if row['schedule_week'] == 1:
            for team in all_teams:
                elo_ratings[team] = 1500

        # Get current Elo ratings
        home_team_rating = elo_ratings[home_team]
        away_team_rating = elo_ratings[away_team]

        # Append current Elo ratings to lists
        home_elo_ratings.append(home_team_rating)
        away_elo_ratings.append(away_team_rating)

        # Calculate expected scores
        home_expected_score = calc_expected_score(home_team_rating, away_team_rating)
        away_expected_score = calc_expected_score(away_team_rating, home_team_rating)

        # Determine observed scores based on game result
        winner_point_diff = abs(row['score_home'] - row['score_away'])
        home_wins = row['score_home'] > row['score_away']

        home_observed_score = 1 if home_wins else 0
        away_observed_score = 1 - home_observed_score

        # Calculate MOV multiplier
        winner_elo_diff = home_team_rating - away_team_rating if home_wins else away_team_rating - home_team_rating
        mov_multiplier = log(winner_point_diff + 1) * (2.2 / ((winner_elo_diff * 0.001) + 2.2))

        # Calculate new Elo ratings with MOV multiplier
        new_home_rating = home_team_rating + mov_multiplier * (home_observed_score - home_expected_score) * 40
        new_away_rating = away_team_rating + mov_multiplier * (away_observed_score - away_expected_score) * 40

        # Update Elo ratings in the dictionary
        elo_ratings[home_team] = new_home_rating
        elo_ratings[away_team] = new_away_rating

    # Add Elo ratings as new columns to the dataset
    data['home_elo_rating'] = home_elo_ratings
    data['away_elo_rating'] = away_elo_ratings
    
    return data

# Apply the corrected Elo rating calculation with MOV multiplier to the dataset
data = calculate_elo_ratings_with_mov(data)


# Add custom ratings based off team stats

custom_rating_columns = ['1stD', 'TotYd', 'RushY', 'exp_pts_offense', 'TO', '1stD.1', 'TotYd.1', 'RushY.1', 'exp_pts_defense', 'TO.1']

scaler = MinMaxScaler()
data[custom_rating_columns] = scaler.fit_transform(data[custom_rating_columns])

def calculate_custom_rating(row):
    home_rating = row['1stD'] + row['TotYd'] + row['RushY'] + row['exp_pts_offense'] - (row['TO'] * 50)
    away_rating = row['1stD.1'] + row['TotYd.1'] + row['RushY.1'] - row['exp_pts_defense'] - (row['TO.1'] * 50)
    return home_rating, away_rating

ratings = data.apply(calculate_custom_rating, axis=1)
data['home_team_rating'] = ratings.apply(lambda x: x[0])
data['away_team_rating'] = ratings.apply(lambda x: x[1])


# Rolling 10 game stas rating

def calculate_10_game_rolling_rating(team, home_data, away_data, home_rating_column, away_rating_column):
    home_ratings = home_data[home_data['team_home'] == team][home_rating_column]
    away_ratings = away_data[away_data['team_away'] == team][away_rating_column]

    all_ratings = pd.concat([home_ratings, away_ratings]).sort_index()

    rolling_ratings = all_ratings.rolling(window=5).mean()

    return rolling_ratings.loc[home_ratings.index].values, rolling_ratings.loc[away_ratings.index].values

home_rolling_team_ratings = []
away_rolling_team_ratings = []
for team in data['team_home'].unique():

    home_rolling, away_rolling = calculate_10_game_rolling_rating(team, data, data, 'home_team_rating', 'away_team_rating')

    home_rolling_team_ratings.extend(home_rolling)
    away_rolling_team_ratings.extend(away_rolling)

data['home_rolling_team_rating'] = home_rolling_team_ratings
data['away_rolling_team_rating'] = away_rolling_team_ratings

data.to_csv(f'{path}new-preprocessed_data.csv', index=False)
