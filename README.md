
 🎯 Project Objective
To build a machine learning model that classifies individuals into **Low**, **Moderate**, or **High** mental health risk categories based on factors such as:
- Age
- Exercise Level
- Sleep Hours
- Stress Level
- Work Hours per Week
- Screen Time
- Social Interaction Score
- Happiness Score
- Diet Type
- Mental Health Condition

---

🛠️ Technologies Used
- **Python**
- **Pandas**, **NumPy** – Data handling
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Machine learning (Decision Tree Classifier)
- **Jupyter Notebook / Google Colab** – Development environment

---

 📊 Steps Performed-

 1️⃣ Data Preprocessing
- Removed duplicates and handled missing values.
- Encoded categorical features:
  - **Ordinal encoding** for ordered features (e.g., Stress Level).
  - **One-hot encoding** for nominal features (e.g., Country, Gender).
- Normalized numerical features with `StandardScaler`.
- Saved the processed dataset as `cleaned_mental_health_data.csv`.

 2️⃣ Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Distribution of each feature and target variable.
- **Bivariate Analysis:** Relationships between features and risk category.
- **Multivariate Analysis:** Correlation heatmaps and multi-feature plots.

 3️⃣ Model Training & Prediction
- Used **Decision Tree Classifier** with `criterion='entropy'` and `max_depth=5`.
- Split dataset into 80% training and 20% testing sets with stratified sampling.

 4️⃣ Model Evaluation
- **Classification Report**: Precision, Recall, F1-score.
- **Confusion Matrix**: Visualized prediction accuracy per class.
- **ROC Curve**: Multi-class ROC with AUC scores.

---

#📈 Results
- **Accuracy:** ~80%
- **High Risk** category detected with 100% precision and recall.
- Moderate misclassification between **Low** and **Moderate** categories.

-
cd mental-health-risk-prediction

