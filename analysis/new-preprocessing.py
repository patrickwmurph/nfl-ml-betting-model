import pandas as pd

import numpy as np

# Read the CSV file
path = '../data/'

df = pd.read_csv(path + '2023-02-12-data.csv', index_col = 0)

betting_df = pd.read_csv(path + 'historical-betting-data.csv', index_col=0)

names_df = pd.read_csv(path + 'nfl_teams.csv') 


################# Clean Data #################

# Map Team Names to Team Code
team_name_to_id_mapping = dict(zip(names_df["team_name"], names_df["team_id"]))

betting_df['Opp'] = betting_df['Opp'].replace('St Louis Rams', 'St. Louis Rams')
betting_df["Team"] = betting_df["Team"].map(team_name_to_id_mapping)
betting_df["Opp"] = betting_df["Opp"].map(team_name_to_id_mapping)
df["Team"] = df["Team"].map(team_name_to_id_mapping)
df["Opp"] = df["Opp"].map(team_name_to_id_mapping)

# Merge in Betting data
data = pd.merge(df, betting_df.drop(columns = ['Week', 'Opp']), on = ['Date', 'Team'], how = 'left')


# Fix week numbers
data['playoff'] = pd.to_numeric(data['Week'], errors='coerce').isna()
data.loc[(data['Week'] == '18'), 'Week'] = '17'
data.loc[(data['Week'] == 'Wild Card'), 'Week'] = '18'
data.loc[(data['Week'] == 'Division'), 'Week'] = '19'
data.loc[(data['Week'] == 'Conf. Champ.'), 'Week'] = '20'
data.loc[(data['Week'] == 'SuperBowl'), 'Week'] = '21'
data['Week'] = data['Week'].astype(int)
data['Date'] = pd.to_datetime(data['Date'])
data['schedule_season'] = data['Date'].apply(lambda x: x.year if x.month >= 9 else x.year - 1)


################# Create W/L and Rec Columns #################

# Create dict to map date and tm to W/L and Rec
team_wl_dict = {(row['Date'], row['Opp']): row['W/L'] for index, row in data.iterrows()}
team_rec_dict = {(row['Date'], row['Opp']): row['Rec'] for index, row in data.iterrows()}
 
# Get opponenet W/L and Rec Functions
def get_opponent_wl(row):
    return team_wl_dict.get((row['Date'], row['Team']), None)

def get_opponent_rec(row):
    return team_rec_dict.get((row['Date'], row['Team']), None)

# Adding new columns for the opponent's W/L and Rec
data['Opp_W/L'] = data.apply(get_opponent_wl, axis=1)
data['Opp_Rec'] = data.apply(get_opponent_rec, axis=1)

# Shift columns within each team Functions 
def shift_columns_within_team(team_data):
    team_data['Prev_Rec'] = team_data['Rec'].shift(1)
    team_data['Prev_W/L'] = team_data['W/L'].shift(1)
    return team_data

def shift_opp_columns_within_team(team_data):
    team_data['Prev_Opp_W/L'] = team_data['Opp_W/L'].shift(1)
    team_data['Prev_Opp_Rec'] = team_data['Opp_Rec'].shift(1)
    return team_data

# Apply functions
data = data.groupby('Team').apply(shift_columns_within_team)
data = data.sort_values(by = ['schedule_season', 'Date']).groupby('Opp').apply(shift_opp_columns_within_team)

# Fill NA caused by shift
data['Prev_W/L'] = data['Prev_W/L'].fillna('N/A')
data['Prev_Rec'] = data['Prev_Rec'].fillna(0)
data['Prev_Opp_W/L'] = data['Prev_Opp_W/L'].fillna('N/A')
data['Prev_Opp_Rec'] = data['Prev_Opp_Rec'].fillna(0)
data['Date'] = pd.to_datetime(data['Date'])

# Change W/L to Numeric
data['Prev_W/L'] = np.where(data['Prev_W/L'] == 'W', 1,0)
data['Prev_Opp_W/L'] =  np.where(data['Prev_Opp_W/L'] == 'W', 1,0)


################# Prepare Data for Rolling Ave by removing NA #################

data['TO.1'] = data['TO.1'].fillna(0)
data['TO'] = data['TO'].fillna(0)
data['PassY'] = data['PassY'].fillna(0)
data['PassY.1'] = data['PassY.1'].fillna(0)

data.rename(columns = {
    'Offense' : 'exp_pts_offense',
    'Defense' : 'exp_pts_defense',
    'Week' : 'schedule_week',
    'Date' : 'schedule_date',
    'Team' : 'team_home',
    'Opp' : 'team_away',
    'Tm' : 'score_home',
    'Opp.1' : 'score_away',
    'O/U_Points' : 'over_under_line'
}, inplace = True)

# Data
data = data.sort_values(by='schedule_date')


################# Create Rolling Average Columns #################

# Function to calculate rolling averages
def calculate_rolling_averages_combined_corrected(data):
    unique_dates = sorted(data['schedule_date'].unique())
    unique_teams = data['team_home'].unique()

    for team in unique_teams:
        for date in unique_dates:
            daily_games = data[data['schedule_date'] == date]

            for idx, game in daily_games.iterrows():
                if game['team_home'] == team:
                    team_type = 'home'
                elif game['team_away'] == team:
                    team_type = 'away'
                else:
                    continue

                team_columns = ['score_home', '1stD', 'TotYd', 'PassY', 'RushY', 'TO'] if team_type == 'home' else ['score_away', '1stD.1', 'TotYd.1', 'PassY.1', 'RushY.1', 'TO.1']
                opponent_columns = ['score_away', '1stD.1', 'TotYd.1', 'PassY.1', 'RushY.1', 'TO.1'] if team_type == 'home' else ['score_home', '1stD', 'TotYd', 'PassY', 'RushY', 'TO']

                for col in team_columns:
                    rolling_series = data.loc[(data['team_' + team_type] == team) & (data['schedule_date'] < date), col].rolling(window=10, min_periods=1).mean()
                    rolling_avg = rolling_series.iloc[-1] if not rolling_series.empty else None
                    data.at[idx, col + '_' + team_type + '_team_rolling_avg'] = rolling_avg

                for col, opp_col in zip(team_columns, opponent_columns):
                    rolling_series = data.loc[(data['team_' + team_type] == team) & (data['schedule_date'] < date), opp_col].rolling(window=10, min_periods=1).mean()
                    rolling_avg = rolling_series.iloc[-1] if not rolling_series.empty else None
                    data.at[idx, opp_col + '_' + team_type + '_opponent_rolling_avg'] = rolling_avg

    return data

# Calculate rolling averages for all teams
final_averages_all_teams = calculate_rolling_averages_combined_corrected(data)

copy_final_averages_df = final_averages_all_teams.copy()


################# Prepare Data for Model #################

# Remove duplicates by only include games for home team
copy_final_averages_df = copy_final_averages_df[copy_final_averages_df['Location'] == 1]

# Clean Spread_Value column
copy_final_averages_df['Spread_Value'] = np.where(copy_final_averages_df['Spread_Value'] == 'PK', 0, copy_final_averages_df['Spread_Value'])
copy_final_averages_df['Spread_Value'] = pd.to_numeric(copy_final_averages_df['Spread_Value'])

# Create simplified columns 
copy_final_averages_df['team_favorite_id'] = np.where(copy_final_averages_df['Spread_Value'] < 0, copy_final_averages_df['team_home'], np.where(copy_final_averages_df['Spread_Value'] > 0, copy_final_averages_df['team_away'], 'PICK'))
copy_final_averages_df['spread_favorite'] = -1 * abs(copy_final_averages_df['Spread_Value'])

# Creat columns for favorites
copy_final_averages_df.loc[copy_final_averages_df['team_favorite_id'] == copy_final_averages_df['team_home'], 'home_favorite'] = 1
copy_final_averages_df.loc[copy_final_averages_df['team_favorite_id'] == copy_final_averages_df['team_away'], 'away_favorite'] = 1
copy_final_averages_df['home_favorite'].fillna(0, inplace=True)
copy_final_averages_df['away_favorite'].fillna(0, inplace=True)
copy_final_averages_df['home_favorite'] = copy_final_averages_df['home_favorite'].astype(int)
copy_final_averages_df['away_favorite'] = copy_final_averages_df['away_favorite'].astype(int)

# Create over column
copy_final_averages_df.loc[((copy_final_averages_df['score_home'] + copy_final_averages_df['score_away']) > copy_final_averages_df['over_under_line']), 'over'] = 1
copy_final_averages_df.over.fillna(0, inplace=True)

# Create column for target
copy_final_averages_df['result'] = (copy_final_averages_df.score_home > copy_final_averages_df.score_away).astype(int)


################# Current Season Win Percentage #################

# Function to calculate win percentage
def calculate_win_percentage(wins, losses):
    if wins + losses == 0:
        return 0
    return wins / (wins + losses)

current_season_home_win_percentage = []
current_season_away_win_percentage = []

current_season_team_records = {team: {'wins': 0, 'losses': 0} for team in copy_final_averages_df['team_home'].unique().tolist() + copy_final_averages_df['team_away'].unique().tolist()}

current_season = None

for index, row in copy_final_averages_df.iterrows():
    home_team = row['team_home']
    away_team = row['team_away']
    
    if current_season != row['schedule_season']:
        current_season = row['schedule_season']
        current_season_team_records = {team: {'wins': 0, 'losses': 0} for team in current_season_team_records.keys()}
    
    home_win_percentage = calculate_win_percentage(current_season_team_records[home_team]['wins'], current_season_team_records[home_team]['losses'])
    away_win_percentage = calculate_win_percentage(current_season_team_records[away_team]['wins'], current_season_team_records[away_team]['losses'])
    
    current_season_home_win_percentage.append(home_win_percentage)
    current_season_away_win_percentage.append(away_win_percentage)
    
    if row['score_home'] > row['score_away']:
        current_season_team_records[home_team]['wins'] += 1
        current_season_team_records[away_team]['losses'] += 1
    else:
        current_season_team_records[home_team]['losses'] += 1
        current_season_team_records[away_team]['wins'] += 1

copy_final_averages_df['current_season_home_win_percentage'] = current_season_home_win_percentage
copy_final_averages_df['current_season_away_win_percentage'] = current_season_away_win_percentage


################# Last Year Win Percentage #################

last_year_home_win_percentages = []
last_year_away_win_percentages = []

current_season_team_records = {team: {'wins': 0, 'losses': 0} for team in copy_final_averages_df['team_home'].unique().tolist() + copy_final_averages_df['team_away'].unique().tolist()}
last_season_team_records = {team: {'wins': 0, 'losses': 0} for team in current_season_team_records.keys()}

current_season = None

for index, row in copy_final_averages_df.iterrows():
    home_team = row['team_home']
    away_team = row['team_away']
    
    if current_season != row['schedule_season']:
        last_season_team_records = current_season_team_records.copy()
        current_season = row['schedule_season']
        current_season_team_records = {team: {'wins': 0, 'losses': 0} for team in last_season_team_records.keys()}
    
    home_win_percentage = calculate_win_percentage(last_season_team_records[home_team]['wins'], last_season_team_records[home_team]['losses'])
    away_win_percentage = calculate_win_percentage(last_season_team_records[away_team]['wins'], last_season_team_records[away_team]['losses'])
    
    last_year_home_win_percentages.append(home_win_percentage)
    last_year_away_win_percentages.append(away_win_percentage)
    
    if row['score_home'] > row['score_away']:
        current_season_team_records[home_team]['wins'] += 1
        current_season_team_records[away_team]['losses'] += 1
    else:
        current_season_team_records[home_team]['losses'] += 1
        current_season_team_records[away_team]['wins'] += 1

copy_final_averages_df['last_year_home_win_percentage'] = last_year_home_win_percentages
copy_final_averages_df['last_year_away_win_percentage'] = last_year_away_win_percentages

copy_final_averages_df.rename(columns = {
    'current_season_home_win_percentage' : 'team_home_current_win_pct',
    'current_season_away_win_percentage' : 'team_away_current_win_pct',
    'last_year_home_win_percentage' : "team_home_lastseason_win_pct",
    'last_year_away_win_percentage' : 'team_away_lastseason_win_pct',
    'Offense' : 'exp_pts_offense',
    'Defense' : 'exp_pts_defense'
    }, inplace = True)

####### Hisotical Matchup Win Percentage #######

games_df = copy_final_averages_df[['team_home', 'team_away', 'W/L']].copy()
games_df['winner'] = games_df.apply(lambda row: row['team_home'] if row['W/L'] == 'W' else row['team_away'], axis=1)

def calculate_h2h_win_percentage(row):
    historical_games = games_df.loc[
        ((games_df['team_home'] == row['team_home']) & (games_df['team_away'] == row['team_away']) |
        (games_df['team_home'] == row['team_away']) & (games_df['team_away'] == row['team_home'])) &
        (games_df.index < row.name)
    ]
    home_wins = (historical_games['winner'] == row['team_home']).sum()
    away_wins = (historical_games['winner'] == row['team_away']).sum()
    total_games = home_wins + away_wins
    home_win_pct = home_wins / total_games if total_games > 0 else 0.0
    away_win_pct = away_wins / total_games if total_games > 0 else 0.0
    return home_win_pct, away_win_pct

games_df['home_team_h2h_win_pct'], games_df['away_team_h2h_win_pct'] = zip(*games_df.apply(calculate_h2h_win_percentage, axis=1))

copy_final_averages_df['home_team_h2h_win_pct'] = games_df['home_team_h2h_win_pct']
copy_final_averages_df['away_team_h2h_win_pct'] = games_df['away_team_h2h_win_pct']

################# Hisotrical Matchup Team Points #################

h2h_team_points_df = copy_final_averages_df.copy()

h2h_team_points_df['matchup'] = h2h_team_points_df.apply(lambda x: '-'.join(sorted([x['team_away'], x['team_home']])), axis=1)

h2h_team_points_df['away_team_hth_rolling_avg_points'] = h2h_team_points_df.groupby(['matchup', 'team_away'])['score_away'].transform(lambda x: x.shift(1).rolling(window=3).mean())

h2h_team_points_df['home_team_hth_rolling_avg_points'] = h2h_team_points_df.groupby(['matchup', 'team_home'])['score_home'].transform(lambda x: x.shift(1).rolling(window=3).mean())

h2h_team_points_df['combined_rolling_avg_points'] = h2h_team_points_df['away_team_hth_rolling_avg_points'] + h2h_team_points_df['home_team_hth_rolling_avg_points']

h2h_team_points_df.drop(columns = ['matchup'], inplace = True)


################# Prepare and Merge Moneyline data #################

money_line_df = pd.read_csv(path + 'moneyline-data.csv', index_col=0)

money_line_df.rename(columns = {'gameday' : 'schedule_date',
                                'away_team' : 'team_away',
                                'home_team' : 'team_home',
                                'away_moneyline' : 'moneyline_away',
                                'home_moneyline' : 'moneyline_home', 
                                'away_spread_odds' : 'spread_odds_away',
                                'home_spread_odds' : 'spread_odds_home', 
                                }, inplace = True)

money_line_df['schedule_date'] = pd.to_datetime(money_line_df['schedule_date'])

updates = {
    'LV': 'LVR',
    'LA': 'LAR',
    'OAK': 'LVR',
    'SD': 'LAC',
    'STL': 'LAR'
}

money_line_df['team_home'] = money_line_df['team_home'].map(updates).fillna(money_line_df['team_home'])
money_line_df['team_away'] = money_line_df['team_away'].map(updates).fillna(money_line_df['team_away'])

merged_moneyline = money_line_df[['schedule_date', 'team_home', 'team_away',
                                  'moneyline_away','moneyline_home', 'spread_line',
                                  'spread_odds_away', 'spread_odds_home','total_line', 
                                  'under_odds', 'over_odds', 'div_game', 'roof', 
                                  'surface', 'temp', 'wind', 
                                  'away_qb_id', 'home_qb_id', 'away_qb_name',
                                  'home_qb_name', 'away_coach', 'home_coach',
                                  'referee', 'stadium_id','stadium']]

averages_line_df = pd.merge(h2h_team_points_df, merged_moneyline, on = ['schedule_date', 'team_home', 'team_away'], how = 'left')

averages_line_df.to_csv('../data/preprocessed_data.csv')