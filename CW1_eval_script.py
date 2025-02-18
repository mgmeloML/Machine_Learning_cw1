import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

# Train a linear model
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
X_tst = tst.drop(columns=['outcome'])
y_tst = tst['outcome']

model = LinearRegression()
model.fit(X_trn, y_trn)
yhat_lm = model.predict(X_tst)

# Format submission
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission.csv', index=False)

### THIS IS HOW THE SOLUTION IS EVALUATED ###

# Read in the submission
yhat_lm = np.array(pd.read_csv('CW1_submission.csv')['yhat'])

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

# Evaluate
print(r2_fn(yhat_lm))




