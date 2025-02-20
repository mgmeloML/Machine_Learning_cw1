import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
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


Scaler = MinMaxScaler()
X_trn_norm = Scaler.fit_transform(X_trn)
X_tst_norm = Scaler.fit_transform(X_tst)

regressor = MLPRegressor()


param_grid = {
    "hidden_layer_sizes": [80,100],
    "activation": ["tanh","relu"],
    "max_iter": [300,400,500],
    "shuffle": [False, True]
}

grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_trn_norm, y_trn)

model = MLPRegressor(**grid_search.best_params_)
model.fit(X_trn_norm, y_trn)

yhat_nn = model.predict(X_tst_norm)

def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

print(grid_search.best_params_)
print(r2_fn(yhat_nn))