import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the Cleaned Data ---
try:
    df = pd.read_csv('cleaned_mental_health_data.csv')
    print("Cleaned dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'cleaned_mental_health_data.csv' not found. Please run the preprocess.py script first.")
    exit()

# Set a professional style for the plots
sns.set_theme(style="whitegrid", palette="viridis")

# ----------------------------------------------------
# a) Univariate Analysis (Analyzing Single Variables)
# ----------------------------------------------------
print("\nGenerating Univariate Analysis plots...")

# Plot 1: Distribution of the target variable 'Risk Category'
plt.figure(figsize=(8, 6))
sns.countplot(x='Risk Category', data=df)
plt.title('Distribution of Mental Health Risk Categories')
plt.xlabel('Risk Category (0=Low, 1=Moderate, 2=High)')
plt.ylabel('Number of Individuals')
plt.show()

# Plot 2: Distribution of 'Age'
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution of Participants')
plt.xlabel('Age (Normalized)')
plt.ylabel('Frequency')
plt.show()

# Plot 3: Distribution of 'Stress Level'
plt.figure(figsize=(8, 6))
sns.countplot(x='Stress Level', data=df)
plt.title('Distribution of Stress Levels')
plt.xlabel('Stress Level (0=Low, 1=Moderate, 2=High)')
plt.ylabel('Number of Individuals')
plt.show()


# ----------------------------------------------------
# b) Bivariate Analysis (Analyzing Two Variables)
# ----------------------------------------------------
print("\nGenerating Bivariate Analysis plots...")

# Plot 1: 'Stress Level' vs. 'Risk Category'
plt.figure(figsize=(10, 7))
sns.countplot(data=df, x='Risk Category', hue='Stress Level', palette='magma')
plt.title('Risk Category by Stress Level')
plt.xlabel('Risk Category (0=Low, 1=Moderate, 2=High)')
plt.ylabel('Number of Individuals')
plt.legend(title='Stress Level')
plt.show()

# Plot 2: 'Sleep Hours' vs. 'Happiness Score'
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Sleep Hours', y='Happiness Score')
plt.title('Sleep Hours vs. Happiness Score')
plt.xlabel('Sleep Hours (Normalized)')
plt.ylabel('Happiness Score (Normalized)')
plt.show()


# ----------------------------------------------------
# c) Multivariate Analysis (Analyzing Multiple Variables)
# ----------------------------------------------------
print("\nGenerating Multivariate Analysis plots...")

# Plot 1: Correlation Heatmap
# We select the most relevant numerical columns for a readable heatmap
cols_for_heatmap = [
    'Age', 'Exercise Level', 'Sleep Hours', 'Stress Level',
    'Work Hours per Week', 'Screen Time per Day (Hours)',
    'Social Interaction Score', 'Happiness Score', 'Risk Category'
]
plt.figure(figsize=(12, 10))
correlation_matrix = df[cols_for_heatmap].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Key Features')
plt.show()

# Plot 2: Happiness Score vs. Risk Category, faceted by Stress Level
# This lets us see the interaction between three variables.
sns.catplot(data=df, x='Risk Category', y='Happiness Score', hue='Stress Level',
            kind='box', palette='plasma', height=6, aspect=1.5)
plt.suptitle('Happiness Score vs. Risk Category by Stress Level', y=1.02) # y=1.02 raises title
plt.xlabel('Risk Category (0=Low, 1=Moderate, 2=High)')
plt.ylabel('Happiness Score (Normalized)')
plt.show()


print("\n--- Analysis script finished ---")