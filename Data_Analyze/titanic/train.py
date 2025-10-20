# %%
import numpy as np
import pandas as pd

def read_data(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)
# %%
train_data = read_data('data/train.csv').copy()
age_median = train_data["Age"].median()
fare_median = train_data["Fare"].median()
embarked_mode = train_data["Embarked"].mode()[0]

train_data["Age"].fillna(age_median, inplace=True)
train_data["Fare"].fillna(fare_median, inplace=True)
train_data["Embarked"].fillna(embarked_mode, inplace=True)

train_data.head()

# %%
test_data = read_data('data/test.csv').copy()
test_data["Age"].fillna(age_median, inplace=True)
test_data["Fare"].fillna(fare_median, inplace=True)
test_data["Embarked"].fillna(embarked_mode, inplace=True)
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

features = [ "Sex", "Age", "SibSp", "Fare", "Embarked"]
X = pd.get_dummies(train_data[features])
feature_columns = X.columns
print(X.head())
# %% 
X_test = pd.get_dummies(test_data[features])
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

# %%
# This cell is meant to test accurancy
from sklearn.model_selection import cross_val_score

cv_model = RandomForestClassifier(**model.get_params())
cv_scores = cross_val_score(cv_model, X, y, cv=5)
print(f"5-fold cross-validation accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"Training accuracy: {model.score(X, y):.3f}")


# %% 
# This cell is meant to let the user input a person and get the 
# predicted result and the actual result
# the input should consist of gender, age and fare
def predict_passenger_outcome(
    model,
    gender,
    age,
    fare,
    pclass,
    sibsp,
    parch,
    training_data,
    feature_columns,
):
    normalized_gender = gender.strip().lower()
    if normalized_gender not in ("male", "female"):
        raise ValueError("Gender must be 'male' or 'female'.")

    input_features = pd.DataFrame(
        {
            "Pclass": [int(pclass)],
            "Sex": [normalized_gender],
            "Age": [age],
            "SibSp": [int(sibsp)],
            "Parch": [int(parch)],
            "Fare": [fare],
            "Embarked": [training_data["Embarked"].mode()[0]],
        }
    )
    input_features = pd.get_dummies(input_features)
    input_features = input_features.reindex(columns=feature_columns, fill_value=0)

    prediction = int(model.predict(input_features)[0])
    proba = model.predict_proba(input_features)[0][prediction]

    comparable_passengers = training_data[
        (training_data["Sex"] == normalized_gender)
        & (training_data["Pclass"] == int(pclass))
    ][["PassengerId", "Age", "Fare", "Survived", "SibSp", "Parch"]].copy()

    actual_result = None
    matched_passenger = None
    distance = None

    if not comparable_passengers.empty:
        comparable_passengers["distance"] = (
            np.abs(comparable_passengers["Age"] - age)
            + np.abs(comparable_passengers["Fare"] - fare)
            + np.abs(comparable_passengers["SibSp"] - int(sibsp))
            + np.abs(comparable_passengers["Parch"] - int(parch))
        )
        best_match = comparable_passengers.loc[comparable_passengers["distance"].idxmin()]
        actual_result = int(best_match["Survived"])
        matched_passenger = int(best_match["PassengerId"])
        distance = float(best_match["distance"])

    print(f"Predicted survival: {"Alive "if prediction else "Dead"} (probability {proba:.2%})")
    if actual_result is not None:
        print(
            f"Closest matching passenger: PassengerId {matched_passenger} | "
            f"actual outcome {actual_result} | distance {distance:.2f}"
        )
    else:
        print("No comparable passenger found to report an actual outcome.")


try:
    gender_input = input("Enter passenger gender (male/female): ")
    age_input = float(input("Enter passenger age (in years): "))
    fare_input = float(input("Enter passenger fare: "))
    pclass_input = int(input("Enter passenger class (1, 2, or 3): "))
    sibsp_input = int(input("Enter number of siblings/spouses aboard: "))
    parch_input = int(input("Enter number of parents/children aboard: "))
    predict_passenger_outcome(
        model=model,
        gender=gender_input,
        age=age_input,
        fare=fare_input,
        pclass=pclass_input,
        sibsp=sibsp_input,
        parch=parch_input,
        training_data=train_data,
        feature_columns=feature_columns,
    )
except ValueError as error:
    print(f"Invalid input: {error}")

# %%
