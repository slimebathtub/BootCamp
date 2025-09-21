# %%
import numpy as np
import pandas as pd

def read_data(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)
# %%
train_data = read_data('../data/train.csv')
train_data.head()

# %%
test_data = read_data('../data/test.csv')
test_data.head()

# %%
women = train_data.loc[train_data.Sex == 'female']["Survived"]
# add all the True(1) caluse and divided by the total
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
# %%
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
# %%
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
print(X.head())
# %% 
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")