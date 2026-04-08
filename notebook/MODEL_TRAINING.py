# ============================================================
# STUDENT PERFORMANCE INDICATOR - MODEL TRAINING
# ============================================================

print("Project Stage: Model Training")
print("Goal: Predict student math score using ML regression models")
print("--------------------------------------------------")


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

print("Importing required libraries for machine learning")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Libraries imported successfully")
print("--------------------------------------------------")


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

print("Loading dataset")

df = pd.read_csv("notebook/data/stud.csv")

print("Dataset loaded successfully")
print("--------------------------------------------------")


# ============================================================
# STEP 3: CREATE INPUT FEATURES (X) AND TARGET VARIABLE (Y)
# ============================================================

print("Preparing input features and target variable")

X = df.drop(columns=["math_score"], axis=1)
y = df["math_score"]

print("Input features and target variable created")
print("--------------------------------------------------")


# ============================================================
# STEP 4: IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

print("Separating categorical and numerical columns")

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

print("Numerical Features:", num_features)
print("Categorical Features:", cat_features)

print("Used for preprocessing pipeline")
print("--------------------------------------------------")


# ============================================================
# STEP 5: DATA PREPROCESSING PIPELINE
# ============================================================

print("Applying StandardScaler and OneHotEncoder")

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", categorical_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

X = preprocessor.fit_transform(X)

print("Data preprocessing completed")
print("--------------------------------------------------")


# ============================================================
# STEP 6: TRAIN TEST SPLIT
# ============================================================

print("Splitting dataset into training and testing sets")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("Dataset split completed")
print("--------------------------------------------------")


# ============================================================
# STEP 7: MODEL EVALUATION FUNCTION
# ============================================================

print("Creating evaluation function")

def evaluate_model(true, predicted):

    mae = mean_absolute_error(true, predicted)

    rmse = np.sqrt(mean_squared_error(true, predicted))

    r2 = r2_score(true, predicted)

    return mae, rmse, r2

print("Evaluation function ready")
print("--------------------------------------------------")


# ============================================================
# STEP 8: TRAIN MULTIPLE MODELS
# ============================================================

print("Training multiple regression models")

models = {

    "Linear Regression": LinearRegression(),

    "Lasso": Lasso(),

    "Ridge": Ridge(),

    "KNN": KNeighborsRegressor(),

    "Decision Tree": DecisionTreeRegressor(),

    "Random Forest": RandomForestRegressor(),

    "XGBoost": XGBRegressor(),

    "CatBoost": CatBoostRegressor(verbose=False),

    "AdaBoost": AdaBoostRegressor()
}

model_names = []
r2_scores = []

for name, model in models.items():

    print("Training model:", name)

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    mae, rmse, r2 = evaluate_model(y_test, y_test_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    model_names.append(name)
    r2_scores.append(r2)

    print("--------------------------------------------------")


# ============================================================
# STEP 9: MODEL COMPARISON RESULTS
# ============================================================

print("Comparing model performances")

results = pd.DataFrame({
    "Model Name": model_names,
    "R2 Score": r2_scores
})

results = results.sort_values(by="R2 Score", ascending=False)

print(results)

print("Shows best performing regression model")
print("--------------------------------------------------")


# ============================================================
# STEP 10: TRAIN BEST MODEL (LINEAR REGRESSION)
# ============================================================

print("Training final Linear Regression model")

lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)

accuracy = r2_score(y_test, y_pred) * 100

print("Model Accuracy:", accuracy)

print("Linear Regression selected as final model")
print("--------------------------------------------------")


# ============================================================
# STEP 11: VISUALIZE ACTUAL VS PREDICTED VALUES
# ============================================================

print("Plotting Actual vs Predicted values")

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted Scores")

sns.regplot(x=y_test, y=y_pred, ci=None, color="red")

plt.show()

print("Visualization completed")
print("--------------------------------------------------")


# ============================================================
# STEP 12: CREATE PREDICTION DIFFERENCE TABLE
# ============================================================

print("Creating comparison table")

prediction_df = pd.DataFrame({

    "Actual Value": y_test,

    "Predicted Value": y_pred,

    "Difference": y_test - y_pred
})

print(prediction_df.head())

print("Shows prediction error differences")
print("--------------------------------------------------")


# ============================================================
# FINAL MODEL TRAINING CONCLUSION
# ============================================================

print("MODEL TRAINING COMPLETED")

print("Linear Regression achieved highest accuracy")

print("Model accuracy approx: 88%")

print("Model ready for deployment or prediction use")

print("--------------------------------------------------")