import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("metacritic_games.csv") 

# Select relevant columns
df = df[['metascore', 'userscore', 'platforms', 'genres', 'releaseDate']]

# Clean data: Convert userscore to numeric and drop missing values
df['userscore'] = pd.to_numeric(df['userscore'], errors='coerce')
df = df.dropna()

# Function to plot and close automatically
def plot_and_close(fig):
    plt.show()
    plt.close(fig)

# --- TASK 1: DISTRIBUTIONS (Required by your teacher) ---

# Distribution of Outcome (Meta Score)
fig1 = plt.figure()
plt.hist(df['metascore'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Meta Scores")
plt.xlabel("Meta Score")
plt.ylabel("Number of Games")
plot_and_close(fig1)

# Distribution of Input (User Score)
fig2 = plt.figure()
plt.hist(df['userscore'], bins=20, color='salmon', edgecolor='black')
plt.title("Distribution of User Scores")
plt.xlabel("User Score")
plt.ylabel("Number of Games")
plot_and_close(fig2)

# --- TASK 2: MACHINE LEARNING IMPLEMENTATION ---

# Encode categorical variables
df_encoded = df.join(df['genres'].str.get_dummies(sep=","))
df_encoded = pd.get_dummies(df_encoded, columns=['platforms'], drop_first=True)

# Separate features and target
X = df_encoded.drop(['metascore', 'genres', 'releaseDate'], axis=1)
y = df_encoded['metascore']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# --- NEW VISUALIZATION: LINEAR RELATIONSHIP ---

# This plot shows how well the Linear Model performed
fig4 = plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='purple')
# Add a diagonal line representing "Perfect Prediction"
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Linear Regression: Actual vs. Predicted Scores")
plt.xlabel("Actual Metascore")
plt.ylabel("Predicted Metascore")
plot_and_close(fig4)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation (Linear Regression):")
print("Mean Squared Error:", round(mse, 2))
print("R-squared:", round(r2, 2))



# ------------------------ Reflection ----------------------------------------------------------------------------------------------------

# The biggest challenge was cleaning the dataset, specifically converting the "tbd" strings in the userscore column 
# into numeric values so the model wouldn't crash. I also had to use one-hot encoding for the genres and platforms to
# transform that categorical text into numbers the linear equation could actually process. Through this project, I learned 
# how much the distribution of data, like the clusters seen in my histograms, impacts the final prediction. 
# Using Linear Regression showed me the direct relationship between user sentiment and critic scores in a way that was easy to interpret.
# Overall, this taught me that Supervised Learning is just as much about careful data preparation as it is about the actual math.

# -------------------------------------------------------------------------------------------------------------------------------------------
