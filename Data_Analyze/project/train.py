# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier 

# %%
def read_data(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

data = read_data('mushrooms.csv')
data.info()

# %%
y = data["class"] # target column: edible or poisonous
x = data.drop(columns=["class"]) # features (all the other columns except target column)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42, stratify=y # y-axis that be picked up is fit with the test_size as well
)

# %%
all_categorical_columns = x.select_dtypes(include=['object']).columns

preprocess = ColumnTransformer([
    ('turn_cat', OneHotEncoder(handle_unknown='ignore'), all_categorical_columns)
])
# %%

model = DecisionTreeClassifier(random_state=42)
pipe = Pipeline([('prep', preprocess), ('model', model)])
# train the model
pipe.fit(X_train, y_train)

# test the model 
pred = pipe.predict(X_test)

# accuracy
accuracy = (pred == y_test).mean()
print(f"Model accuracy: {accuracy:.2%}")

# %%
