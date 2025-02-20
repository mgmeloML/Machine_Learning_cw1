import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Set seed
np.random.seed(123)

# Import training data
df_trn = pd.read_csv('CW1_train.csv')
df_tst = pd.read_csv("CW1_test.csv")

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']  # Replace with actual categorical column names

# One-hot encode categorical variables
df_trn = pd.get_dummies(df_trn, columns=categorical_cols, drop_first=True)
df_tst = pd.get_dummies(df_tst, columns=categorical_cols, drop_first=True)

X_trn = df_trn.drop(columns=['outcome'])
y_trn = df_trn['outcome']
X_tst = df_tst


regressor = DecisionTreeRegressor()

rfe = RFECV(regressor, step=1, cv=5, scoring="r2", n_jobs=-1)
rfe.fit(X_trn, y_trn)

X_trn_selected = X_trn.iloc[:, rfe.support_]
X_tst_selected = X_tst.iloc[:, rfe.support_]

param_grid = {
    "max_depth": [None,5,10,15],
    "min_samples_split":[1,2,3,4],
    "min_samples_leaf": [0,1,2],
    "max_leaf_nodes": [None, 20, 30, 40, 50]
}
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_trn_selected, y_trn)

model = DecisionTreeRegressor(**grid_search.best_params_)

model.fit(X_trn_selected,y_trn)

yhat_dt = model.predict(X_tst_selected)

# Format submission
out = pd.DataFrame({'yhat': yhat_dt})
out.to_csv('CW1_submission.csv', index=False)