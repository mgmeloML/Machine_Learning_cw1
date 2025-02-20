import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set seed
np.random.seed(123)

# Import training data
df = pd.read_csv('CW1_train.csv')

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']  # Replace with actual categorical column names

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Random 80/20 train/validation split
trn, tst = train_test_split(df, test_size=0.2, random_state=123)

X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
X_tst = tst.drop(columns=['outcome'])
y_tst = tst['outcome']


regressor = RandomForestRegressor()

rfe = RFECV(regressor, step=1, cv=5, scoring="r2", n_jobs=-1)
rfe.fit(X_trn, y_trn)

X_trn_selected = X_trn.iloc[:, rfe.support_]
X_tst_selected = X_tst.iloc[:, rfe.support_]

# Define the grid of hyperparameters
param_grid = {
    'n_estimators': [5,10,15,20],   # Number of trees
    'max_depth': [None, 10, 20],      # Depth of each tree
    'min_samples_split': [2, 5, 10]  # Minimum samples to split a node
}

grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_trn_selected, y_trn)


model = RandomForestRegressor(**param_grid)

model.fit(X_trn_selected,y_trn)
yhat_rf = model.predict(X_tst_selected)


def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

print(X_trn.columns[rfe.support_])
print(grid_search.best_params_)
print(r2_fn(yhat_rf))