import pandas as pd
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split


path = 'data/'
df = pd.read_csv(path + 'preprocessed_data.csv')
df['margin'] = df['score_away'] - df['score_home']
df.fillna(0, inplace=True)

################# Feature Selection #################

features = [
    'schedule_week',
    'Spread_Value','over_under_line', 'playoff','schedule_season',
    'Prev_Rec', 'Prev_W/L','Prev_Opp_W/L', 'Prev_Opp_Rec',
    'score_home_home_team_rolling_avg',
    '1stD_home_team_rolling_avg', 'TotYd_home_team_rolling_avg',
    'PassY_home_team_rolling_avg', 'RushY_home_team_rolling_avg',
    'TO_home_team_rolling_avg', 'score_away_home_opponent_rolling_avg',
    '1stD.1_home_opponent_rolling_avg', 'TotYd.1_home_opponent_rolling_avg',
    'PassY.1_home_opponent_rolling_avg',
    'RushY.1_home_opponent_rolling_avg', 'TO.1_home_opponent_rolling_avg',
    'score_away_away_team_rolling_avg', '1stD.1_away_team_rolling_avg',
    'TotYd.1_away_team_rolling_avg', 'PassY.1_away_team_rolling_avg',
    'RushY.1_away_team_rolling_avg', 'TO.1_away_team_rolling_avg',
    'score_home_away_opponent_rolling_avg',
    '1stD_away_opponent_rolling_avg', 'TotYd_away_opponent_rolling_avg',
    'PassY_away_opponent_rolling_avg', 'RushY_away_opponent_rolling_avg',
    'TO_away_opponent_rolling_avg', 'spread_favorite',
    'home_favorite', 'away_favorite',
    'team_home_current_win_pct', 'team_away_current_win_pct',
    'team_home_lastseason_win_pct', 'team_away_lastseason_win_pct',
    'home_team_h2h_win_pct', 'away_team_h2h_win_pct', 'home_elo_rating',
    'away_elo_rating',
    'margin'
]

################# Model Selection and Hyperparameter Tuning #################

df_features = df[features]
df_features.fillna(0, inplace=True)

# Test Feature importance

X = df_features.drop('margin', axis=1)
y = df_features['margin']

# Train randomforestregressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=1)
rf_regressor.fit(X, y)

feature_importance_regressor = rf_regressor.feature_importances_

# Df of results

feature_importance_df_regressor = pd.DataFrame({
    'Feature': features[:-1],
    'Importance': feature_importance_regressor
})

# Sort features by importance
sorted_feature_importance_regressor = feature_importance_df_regressor.sort_values(by='Importance', ascending=False)

# Pick top 20 features

features = sorted_feature_importance_regressor.head(n=20)['Feature'].values

print(features)

################# Hyperparameter Tuning #################

X = df[features]
y = df['margin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Hyperparameter Tuning

param_grid_rf_regressor = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search_rf_regressor = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=1),
                                                param_distributions=param_grid_rf_regressor,
                                                n_iter=10,
                                                scoring='neg_mean_squared_error',
                                                cv=5,
                                                n_jobs=-1,
                                                random_state=1)

# Fit the model
random_search_rf_regressor.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rf_random = random_search_rf_regressor.best_params_
best_score_rf_random = random_search_rf_regressor.best_score_

print(best_params_rf_random, best_score_rf_random)


# Randomized search for GradientBoostingRegressor
param_grid_gb_regressor = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1, 1],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

random_search_gb_regressor = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=1),
                                                param_distributions=param_grid_gb_regressor,
                                                n_iter=20,
                                                scoring='neg_mean_squared_error',
                                                cv=5,
                                                n_jobs=-1,
                                                random_state=1)

# Fit the model
random_search_gb_regressor.fit(X_train, y_train)

# Get the best hyperparameters
best_params_gb_random = random_search_gb_regressor.best_params_
best_score_gb_random = random_search_gb_regressor.best_score_

print(best_params_gb_random, best_score_gb_random)



################# Apply Model and Predict Results for 2022 Season #################

# Split data into train and test sets

target = 'margin'

train = df.loc[df['schedule_season'] < 2022]
test = df.loc[df['schedule_season'] == 2022]

X_train = train[features]
y_train = train[target].values.ravel()

X_test = test[features]
y_test = test[target].values.ravel()

# Voting Regressor with optimized models
vote = VotingRegressor(estimators=[
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(**best_params_rf_random, random_state=1)),
    ('Gradient Boosting', GradientBoostingRegressor(**best_params_gb_random, random_state=1))
])

# Fit
model = vote.fit(X_train, y_train)

# Predict the margin
predicted_margins = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, predicted_margins)
mae = mean_absolute_error(y_test, predicted_margins)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)


################# Create Confidence of Bet and Evaluate Bet Success #################

test.loc[:,'predicted_margin'] = predicted_margins
test = test[['schedule_season', 'schedule_week', 'team_home', 'team_away', 'score_home', 'score_away', 'Spread_Value','margin', 'predicted_margin','result']]
