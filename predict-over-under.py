import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


path = 'data/'
df = pd.read_csv(path + 'preprocessed_data.csv')
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
    'TO_away_opponent_rolling_avg', 
    'home_favorite', 'away_favorite',
    'team_home_current_win_pct', 'team_away_current_win_pct',
    'team_home_lastseason_win_pct', 'team_away_lastseason_win_pct',
    'home_team_h2h_win_pct', 'away_team_h2h_win_pct', 
    'away_team_hth_rolling_avg_points','home_team_hth_rolling_avg_points', 
    'combined_rolling_avg_points',
    'home_elo_rating','away_elo_rating', 'over'
]

df_features = df[features]
df_features.fillna(0, inplace=True)

# Test Feature importance

X = df_features.drop('over', axis=1)
y = df_features['over']

# Train randomforestclassifier

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X, y)

feature_importance = rf.feature_importances_

# Df of results

feature_importance_df = pd.DataFrame({
    'Feature': features[:-1],
    'Importance': feature_importance
})

# Sort features by importance
sorted_feature_importance = feature_importance_df.sort_values(by='Importance', ascending=False)

# Pick top 20 features
features = sorted_feature_importance.head(n=10)['Feature'].values

print(features)


################# Model Selection and Hyperparameter Tuning #################

## Model Selection

X = df[features]
y = df['over']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=1),
    "Random Forest": RandomForestClassifier(random_state=1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1),
    "Calibrated SVM (CCV)": CCV(base_estimator=SVC(probability=True), method='isotonic', cv=3),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=1)
}

# Train the classifiers and evaluate ROC AUC score
roc_auc_scores = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    roc_auc_scores[name] = roc_auc

roc_auc_scores

## Hyperparameter Tuning

# Randomized search for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Randomized search for Decision Tree
random_search_dt = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=1),
                                      param_distributions=param_grid_dt,
                                      n_iter=30,
                                      scoring='roc_auc',
                                      cv=5,
                                      n_jobs=-1,
                                      random_state=1)

# Fit the model
random_search_dt.fit(X_train, y_train)

best_params_dt_random = random_search_dt.best_params_
best_score_dt_random = random_search_dt.best_score_

print(best_params_dt_random, best_score_dt_random)

# Randomized search for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
random_search_rf = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=1),
                                              param_distributions=param_grid_rf,
                                              n_iter=30,
                                              scoring='roc_auc',
                                              cv=5,
                                              n_jobs=-1,
                                              random_state=1)

# Fit the model
random_search_rf.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rf_random = random_search_rf.best_params_
best_score_rf_random = random_search_rf.best_score_

print(best_params_rf_random, best_score_rf_random)


# Randomized search for Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1, 1],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}
random_search_gb = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=1),
                                              param_distributions=param_grid_gb,
                                              n_iter=20,
                                              scoring='roc_auc',
                                              cv=5,
                                              n_jobs=-1,
                                              random_state=1)

# Fit the model
random_search_gb.fit(X_train, y_train)

# Get the best hyperparameters
best_params_gb_random = random_search_gb.best_params_
best_score_gb_random = random_search_gb.best_score_

print(best_params_gb_random, best_score_gb_random)

################# Apply Model and Predict Results for 2022 Season #################

# Split data into train and test sets

target = 'over'

train = df.loc[df['schedule_season'] < 2022]
test = df.loc[df['schedule_season'] == 2022]

X_train = train[features]
y_train = train[target].values.ravel()

X_test = test[features]
y_test = test[target].values.ravel()

# Initialize Models & Classifier

decision_tree = DecisionTreeClassifier(splitter = 'best',
                                       min_samples_split = 5,
                                       min_samples_leaf = 2,
                                       max_features = None,
                                       max_depth = 15,
                                       criterion = 'entropy',
                                       random_state=1)

random_forest = RandomForestClassifier(n_estimators=50,
                                       min_samples_split =10,
                                       min_samples_leaf = 4,
                                       max_depth = 10,
                                       bootstrap = True,
                                       random_state=1)

gradient_boosting = GradientBoostingClassifier(subsample = .9,
                                               n_estimators = 100,
                                               min_samples_split = 5,
                                               min_samples_leaf = 2,
                                               max_depth = 3,
                                               learning_rate = 1,
                                               random_state=1)

# Voting Classifier with optimized models
vote = VotingClassifier(estimators=[
    ('Decision Tree', decision_tree),
    ('Random Forest', random_forest),
    ('Gradient Boosting', gradient_boosting)
], voting='soft')


# Fit
model = vote.fit(X_train, y_train)

# Predict
predicted = model.predict_proba(X_test)[:,1]

# Predict home team W/L
predictions = model.predict(X_test)

# Score
roc_auc_score(y_test, predicted)


# Create df of results
test.loc[:,'over_prob'] = predicted
test = test[['schedule_season', 'schedule_week', 'team_home', 'team_away', 'score_home', 'score_away','over','over_prob']]

## Create Confidence of Bet

bet_probaility = [.2, .8]

## Evalulate Bet Success
test['my_bet_won'] = (((test.over_prob >= bet_probaility[1]) & (test.over == 1)) | 
                      ((test.over_prob <= bet_probaility[0]) & (test.over == 0))).astype(int)
test['my_bet_lost'] = (((test.over_prob >= bet_probaility[1]) & (test.over == 0)) | 
                       ((test.over_prob <= bet_probaility[0]) & (test.over == 1))).astype(int)

print("Win Percentage: " + "{:.4f}".format(test.my_bet_won.sum() / (test.my_bet_lost.sum() + test.my_bet_won.sum())))
print("Bets Won: " + str(test.my_bet_won.sum()))
print("Bets Made: " + str((test.my_bet_lost.sum() + test.my_bet_won.sum())))
print("Possible Games: " + str(len(test)))
