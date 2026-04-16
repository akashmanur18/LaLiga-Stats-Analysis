

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score,mean_absolute_error

# -------------------------
# Step 1: Load & Clean Data
# -------------------------
df = pd.read_csv("laliga_stats.csv")

print("Initial Shape:", df.shape)
print(df.head())

# Handle missing values
df.fillna({
    "goals": 0,
    "assists": 0,
    "passes_completed": 0,
    "yellow_cards": 0,
    "red_cards": 0
}, inplace=True)

# Normalize team names
df['team'] = df['team'].str.strip().str.title()

# Ensure datatypes
df['matches_played'] = df['matches_played'].astype(int)
df['goals'] = df['goals'].astype(int)
df['assists'] = df['assists'].astype(int)
df['passes_completed'] = df['passes_completed'].astype(int)

# -------------------------
# Step 2: Exploratory Data Analysis (EDA)
# -------------------------
# Top 5 goal scorers
top_scorers = df.sort_values("goals", ascending=False).head(5)
print("Top Scorers:\n", top_scorers[['player_name', 'team', 'goals']])

# Top 5 assist leaders
top_assists = df.sort_values("assists", ascending=False).head(5)
print("Top Assists:\n", top_assists[['player_name', 'team', 'assists']])

# Top 5 pass leaders
top_passes = df.sort_values("passes_completed", ascending=False).head(5)
print("Top Pass Leaders:\n", top_passes[['player_name', 'team', 'passes_completed']])

# Team-wise total goals
team_goals = df.groupby('team')['goals'].sum().sort_values(ascending=False)
print("Team Goals:\n", team_goals.head())

# ==============================================
# Extended Visualizations for La Liga Analysis
# ==============================================

# 1. Top 10 Goal Scorers
top10_goals = df.sort_values("goals", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="goals", y="player_name", data=top10_goals, palette="Blues_r")
plt.title("Top 10 Goal Scorers", fontsize=14)
plt.xlabel("Goals")
plt.ylabel("Player")
plt.show()

# 2. Top 10 Assist Leaders
top10_assists = df.sort_values("assists", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="assists", y="player_name", data=top10_assists, palette="Greens_r")
plt.title("Top 10 Assist Leaders", fontsize=14)
plt.xlabel("Assists")
plt.ylabel("Player")
plt.show()

# 3. Top 10 Pass Leaders
top10_passes = df.sort_values("passes_completed", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="passes_completed", y="player_name", data=top10_passes, palette="Oranges_r")
plt.title("Top 10 Pass Leaders", fontsize=14)
plt.xlabel("Passes Completed")
plt.ylabel("Player")
plt.show()

# 4. Team-wise Total Goals
team_goals = df.groupby("team")["goals"].sum().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=team_goals.values, y=team_goals.index, palette="coolwarm")
plt.title("Team-wise Total Goals", fontsize=14)
plt.xlabel("Goals")
plt.ylabel("Team")
plt.show()

# 5. Goals vs Assists Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x="goals", y="assists",  size="matches_played", data=df, alpha=0.7)
plt.title("Goals vs Assists by Position", fontsize=14)
plt.xlabel("Goals")
plt.ylabel("Assists")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.show()

# 6. Distribution of Player Ratings (proxy: goals + assists)
df["contribution"] = df["goals"] + df["assists"]
plt.figure(figsize=(8,6))
sns.histplot(df["contribution"], bins=20, kde=True, color="purple")
plt.title("Distribution of Player Contributions (Goals + Assists)", fontsize=14)
plt.xlabel("Goals + Assists")
plt.ylabel("Number of Players")
plt.show()

# 7. Discipline: Yellow & Red Cards by Team
discipline = df.groupby("team")[["yellow_cards", "red_cards"]].sum().sort_values("yellow_cards", ascending=False)
discipline.plot(kind="bar", figsize=(12,6), color=["gold", "red"])
plt.title("Discipline: Yellow & Red Cards by Team", fontsize=14)
plt.ylabel("Cards")
plt.show()

# 8. Heatmap of Correlations
plt.figure(figsize=(8,6))
sns.heatmap(df[["goals", "assists", "passes_completed", "matches_played", "yellow_cards"]].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

# ==============================
# Logistic Regression (Top Scorer Prediction)
# ==============================
X = df[["shots", "passes_completed", "minutes_played"]]
y = df["top_scorer"]

# Train-test split (stratified to balance classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions
y_pred = log_model.predict(X_test)

# Evaluation
print("=== Logistic Regression (Top Scorer Prediction) ===")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ==============================
# Linear Regression (Team Goals Prediction)
# ==============================
# Aggregate by team
team_stats = df.groupby("team")[["shots", "possession", "goals"]].sum().reset_index()

# Features & target
X_lin = team_stats[["shots", "possession"]]
y_lin = team_stats["goals"]

# Train-test split
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)

# Train Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)

# Predictions
y_pred_lin = lin_model.predict(X_test_lin)

# Evaluation
print("\n=== Linear Regression (Team Goals Prediction) ===")
print("MAE:", mean_absolute_error(y_test_lin, y_pred_lin))
print("RMSE:", np.sqrt(mean_squared_error(y_test_lin, y_pred_lin)))
print("R² Score:", r2_score(y_test_lin, y_pred_lin))

df.to_csv('cleaned_laliga_stats.csv',index=False)
