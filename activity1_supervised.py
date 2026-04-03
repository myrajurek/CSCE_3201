import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
BASE_DIR = Path(__file__).parent
data_path = BASE_DIR/"metacritic_games.csv"
df = pd.read_csv(data_path) 

# get columns
print("Columns in dataset:", df.columns)

# Select relevant columns
df = df[['metascore', 'userscore', 'platforms', 'genres', 'releaseDate']]

# Drop missing values
df = df.dropna()

# Function to plot and close automatically
def plot_and_close(fig):
    plt.show()
    plt.close(fig)

# Outcome variable distribution (Meta Score)
fig1 = plt.figure()
plt.hist(df['metascore'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Meta Scores")
plt.xlabel("Meta Score")
plt.ylabel("Number of Games")
plot_and_close(fig1)

# Input variable distribution (User Score)
fig2 = plt.figure()
plt.hist(df['userscore'], bins=20, color='salmon', edgecolor='black')
plt.title("Distribution of User Scores")
plt.xlabel("User Score")
plt.ylabel("Number of Games")
plot_and_close(fig2)

# Input variable distribution (Genres)
genre_series = df['genres'].str.split(",", expand=True).stack()
genre_counts = genre_series.value_counts()
fig3 = plt.figure()
plt.barh(genre_counts.index, genre_counts.values, color='lightgreen')
plt.title("Distribution of Game Genres")
plt.xlabel("Number of Games")
plt.ylabel("Genre")
plot_and_close(fig3)

# Encode categorical variables
df = df.join(df['genres'].str.get_dummies(sep=","))
df = pd.get_dummies(df, columns=['platforms'], drop_first=True)

# Separate features and target
X = df.drop(['metascore', 'userscore', 'genres', 'releaseDate'], axis=1)  # remove non-numeric columns
y = df['metascore']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", round(mse, 2))
print("R-squared:", round(r2, 2))

print("\nReflection:", flush=True)
print("The biggest challenge was cleaning the dataset and dealing with categorical variables like genres and platforms.", flush=True)
print("Encoding them properly was important to make the model work.", flush=True)
print("I learned that Random Forest can handle complex datasets well without much tuning, and visualizing the data first helps understand patterns and potential predictors.", flush=True)

# Keep script open so you can see all output in terminal
input("\nPress Enter to exit…")