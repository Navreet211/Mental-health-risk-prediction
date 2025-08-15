import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load and Inspect the Data ---
try:
    # We select only the first 13 columns to ignore the empty ones at the end.
    df = pd.read_csv('Mental_Health_Lifestyle_Dataset.csv', encoding='latin-1')
    df.dropna(axis='columns', how='all', inplace=True)
    print("Columns found:", df.columns)
    print("Successfully loaded the dataset.")
except FileNotFoundError:
    print("Error: The file 'Mental_Health_Lifestyle_Dataset.csv' was not found.")
    exit()

# Remove any fully duplicate rows
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"Removed {initial_rows - len(df)} duplicate rows.")


# --- Step 2: Handle Missing Values ---
# Impute numerical columns with the median
for col in ['Work Hours per Week']:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in '{col}' with median value: {median_val}")

# No obvious categorical missing values to fill, but this is how you would do it:
# mode_val = df['some_categorical_column'].mode()[0]
# df['some_categorical_column'].fillna(mode_val, inplace=True)


# --- Step 3: Encode Categorical Features ---
print("Encoding categorical features...")

# Label Encoding for Ordinal Data (features with a clear order)
exercise_map = {'Low': 0, 'Moderate': 1, 'High': 2}
stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
risk_map = {'Low Risk': 0, 'Moderate Risk': 1, 'High Risk': 2}

df['Exercise Level'] = df['Exercise Level'].map(exercise_map)
df['Stress Level'] = df['Stress Level'].map(stress_map)
df['Risk Category'] = df['Risk Category'].map(risk_map)

# One-Hot Encoding for Nominal Data (features without order)
# This creates new binary columns for each category
df = pd.get_dummies(df, columns=['Country', 'Gender', 'Diet Type', 'Mental Health Condition'], drop_first=True)


# --- Step 4: Normalize Numerical Features ---
print("Normalizing numerical features...")

# Identify all numerical columns that need to be scaled
numerical_cols = ['Age', 'Sleep Hours', 'Work Hours per Week', 'Screen Time per Day (Hours)', 'Social Interaction Score', 'Happiness Score']

# Initialize the StandardScaler and fit-transform the data
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# --- Final Output ---
print("\nPreprocessing complete!")
print("Here are the first 5 rows of the processed data:")
print(df.head())

# Save the cleaned data to a new CSV file for the next steps
df.to_csv('cleaned_mental_health_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_mental_health_data.csv'")