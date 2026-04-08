# ============================================================
# STUDENT PERFORMANCE INDICATOR PROJECT
# ============================================================

print("Project: Student Performance Indicator")
print("Goal: Understand how different factors affect student exam scores")
print("Factors: Gender, Ethnicity, Parental Education, Lunch Type, Test Preparation")
print("--------------------------------------------------")


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

print("Importing required libraries for data analysis")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("Libraries imported successfully")
print("--------------------------------------------------")


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

print("Loading dataset into pandas dataframe")

df = pd.read_csv("notebook/data/stud.csv")

print("Dataset loaded successfully")
print("--------------------------------------------------")


# ============================================================
# STEP 3: DISPLAY FIRST 5 ROWS
# ============================================================

print("Displaying top 5 rows of dataset")

print(df.head())

print("This helps us understand dataset structure")
print("--------------------------------------------------")


# ============================================================
# STEP 4: CHECK DATASET SHAPE
# ============================================================

print("Checking number of rows and columns")

print(df.shape)

print("Dataset contains rows and columns shown above")
print("--------------------------------------------------")


# ============================================================
# STEP 5: DATASET INFORMATION
# ============================================================

print("Checking dataset structure and data types")

print(df.info())

print("This shows column names, non-null values and datatypes")
print("--------------------------------------------------")


# ============================================================
# STEP 6: CHECK MISSING VALUES
# ============================================================

print("Checking missing values in dataset")

print(df.isna().sum())

print("Shows count of missing values in each column")
print("--------------------------------------------------")


# ============================================================
# STEP 7: CHECK DUPLICATE VALUES
# ============================================================

print("Checking duplicate records")

print(df.duplicated().sum())

print("Displays total duplicate rows in dataset")
print("--------------------------------------------------")


# ============================================================
# STEP 8: CHECK UNIQUE VALUES
# ============================================================

print("Checking unique values in each column")

print(df.nunique())

print("Shows number of unique categories per column")
print("--------------------------------------------------")


# ============================================================
# STEP 9: DESCRIPTIVE STATISTICS
# ============================================================

print("Checking statistical summary of numerical columns")

print(df.describe())

print("Shows mean, std deviation, min, max and quartiles")
print("--------------------------------------------------")


# ============================================================
# STEP 10: IDENTIFY NUMERICAL & CATEGORICAL FEATURES
# ============================================================

print("Separating numerical and categorical columns")

numeric_features = [col for col in df.columns if df[col].dtype != "O"]

categorical_features = [col for col in df.columns if df[col].dtype == "O"]

print("Numerical Features:", numeric_features)
print("Categorical Features:", categorical_features)

print("This helps during preprocessing and model training")
print("--------------------------------------------------")


# ============================================================
# STEP 11: ADD TOTAL SCORE COLUMN
# ============================================================

print("Creating Total Score column")

df["total_score"] = (
    df["math_score"]
    + df["reading_score"]
    + df["writing_score"]
)

print("Total score column created")
print("--------------------------------------------------")


# ============================================================
# STEP 12: ADD AVERAGE SCORE COLUMN
# ============================================================

print("Creating Average Score column")

df["average_score"] = df["total_score"] / 3

print("Average score column created")
print("--------------------------------------------------")


# ============================================================
# STEP 13: STUDENTS WITH FULL MARKS
# ============================================================

print("Counting students scoring full marks")

math_full = df[df["math_score"] == 100].shape[0]
reading_full = df[df["reading_score"] == 100].shape[0]
writing_full = df[df["writing_score"] == 100].shape[0]

print("Full marks in Math:", math_full)
print("Full marks in Reading:", reading_full)
print("Full marks in Writing:", writing_full)

print("Displays number of top scorers in each subject")
print("--------------------------------------------------")


# ============================================================
# STEP 14: STUDENTS WITH LESS THAN 20 MARKS
# ============================================================

print("Counting students scoring below 20 marks")

math_low = df[df["math_score"] <= 20].shape[0]
reading_low = df[df["reading_score"] <= 20].shape[0]
writing_low = df[df["writing_score"] <= 20].shape[0]

print("Low scores in Math:", math_low)
print("Low scores in Reading:", reading_low)
print("Low scores in Writing:", writing_low)

print("Displays number of weak performing students")
print("--------------------------------------------------")


# ============================================================
# STEP 15: SCORE DISTRIBUTION VISUALIZATION
# ============================================================

print("Plotting average score distribution")

plt.figure(figsize=(10,5))

sns.histplot(df["average_score"], kde=True)

plt.title("Average Score Distribution")

plt.show()

print("Histogram shows distribution of student performance")
print("--------------------------------------------------")


# ============================================================
# STEP 16: GENDER VS PERFORMANCE
# ============================================================

print("Analyzing gender impact on performance")

gender_avg = df.groupby("gender").mean(numeric_only=True)

print(gender_avg)

print("Displays average marks comparison between genders")
print("--------------------------------------------------")


# ============================================================
# STEP 17: LUNCH TYPE IMPACT
# ============================================================

print("Analyzing lunch type impact on performance")

lunch_avg = df.groupby("lunch").mean(numeric_only=True)

print(lunch_avg)

print("Shows how nutrition affects exam scores")
print("--------------------------------------------------")


# ============================================================
# STEP 18: TEST PREPARATION COURSE IMPACT
# ============================================================

print("Analyzing impact of test preparation course")

prep_avg = df.groupby("test_preparation_course").mean(numeric_only=True)

print(prep_avg)

print("Shows improvement after preparation course")
print("--------------------------------------------------")


# ============================================================
# STEP 19: PARENTAL EDUCATION IMPACT
# ============================================================

print("Analyzing parental education impact")

parent_avg = df.groupby("parental_level_of_education").mean(numeric_only=True)

print(parent_avg)

print("Displays effect of parents education on student performance")
print("--------------------------------------------------")


# ============================================================
# FINAL PROJECT CONCLUSION
# ============================================================

print("PROJECT CONCLUSION")

print("Student performance depends on multiple factors")
print("Standard lunch improves performance")
print("Test preparation course improves scores")
print("Parental education has moderate impact")
print("Female students perform better overall")

print("EDA Completed Successfully")
print("--------------------------------------------------")