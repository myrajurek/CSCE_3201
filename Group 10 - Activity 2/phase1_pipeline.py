# =============================================================================
# phase1_pipeline.py
# Phase 1 – Metacritic Games: Metascore Prediction
#
# PURPOSE:
#   This file faithfully reproduces the Phase 1 supervised learning workflow
#   exactly as submitted in Activity 1.  It:
#     1. Loads the Metacritic games CSV dataset
#     2. Cleans the data (handles "tbd" userscore strings)
#     3. Encodes categorical features with the same logic as Activity 1
#     4. Trains a Linear Regression model to predict metascore
#     5. Evaluates with MSE and R²
#     6. Saves the trained model + feature column list to disk
#        so the Phase 2 agent can reload and use them at runtime.
#
# DATASET  : metacritic_games.csv
# TARGET   : metascore  (continuous critic score, 0–100)
# FEATURES : userscore, platforms (one-hot), genres (multi-hot)
# MODEL    : LinearRegression (scikit-learn)
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------------------------------------------------------
# SECTION 1 – LOAD DATASET
# Reads the raw CSV and selects the five columns used in Activity 1.
# -----------------------------------------------------------------------------

def load_dataset(path="metacritic_games.csv"):
    """
    Load the Metacritic games CSV.
    Only the five columns from Activity 1 are kept:
        metascore   – target variable (critic score 0–100)
        userscore   – main numeric predictor (user rating)
        platforms   – which console(s) the game appeared on (categorical)
        genres      – comma-separated genre tags (categorical, multi-label)
        releaseDate – kept here, dropped during feature engineering
    """
    df = pd.read_csv(path)
    df = df[['metascore', 'userscore', 'platforms', 'genres', 'releaseDate']]
    print(f"[Dataset] Loaded {len(df)} rows × {df.shape[1]} columns.")
    return df


# -----------------------------------------------------------------------------
# SECTION 2 – CLEAN DATA
# Exactly mirrors Activity 1's cleaning step:
#   • pd.to_numeric(errors='coerce') converts "tbd" → NaN
#   • dropna() removes all incomplete rows
# -----------------------------------------------------------------------------

def clean_data(df):
    """
    The userscore column contains the string "tbd" for unreleased/unrated games.
    pd.to_numeric with errors='coerce' silently converts those to NaN so the
    model does not crash. dropna() then removes rows with any missing value.
    This is identical to the cleaning block in Activity 1.
    """
    df = df.copy()
    df['userscore'] = pd.to_numeric(df['userscore'], errors='coerce')
    before = len(df)
    df = df.dropna()
    print(f"[Cleaning] Removed {before - len(df)} rows with missing values. "
          f"Remaining: {len(df)} rows.")
    return df


# -----------------------------------------------------------------------------
# SECTION 3 – FEATURE ENGINEERING
# Reproduces the exact encoding from Activity 1:
#   • Genre multi-hot  : df['genres'].str.get_dummies(sep=",")
#   • Platform one-hot : pd.get_dummies(columns=['platforms'], drop_first=True)
# Returns X, y, and the column list needed to align new inputs later.
# -----------------------------------------------------------------------------

def engineer_features(df):
    """
    Transform text columns into binary numeric columns.

    Genres   → multi-label binarisation: one column per unique genre tag.
               A game tagged "Action,RPG" gets a 1 in both 'Action' and 'RPG'.
    Platforms→ standard one-hot encoding, drop_first=True avoids the dummy trap.
    releaseDate and the original genres string are dropped because they are
    either unused (releaseDate) or replaced by their expanded columns (genres).

    Returns
    -------
    X            : feature DataFrame ready for the model
    y            : target Series (metascore)
    feature_cols : ordered list of X column names.  This list is SAVED to disk
                   so the agent knows the exact columns it must build when
                   encoding a new game input — column order must be identical.
    """
    df_enc = df.copy()

    # Multi-hot encode genres (comma-separated → individual binary columns)
    genre_dummies = df_enc['genres'].str.get_dummies(sep=",")
    df_enc = df_enc.join(genre_dummies)

    # One-hot encode platforms (drop first category to avoid multicollinearity)
    df_enc = pd.get_dummies(df_enc, columns=['platforms'], drop_first=True)

    # Split into features (X) and target (y)
    y = df_enc['metascore']
    X = df_enc.drop(['metascore', 'genres', 'releaseDate'], axis=1)

    feature_cols = list(X.columns)
    print(f"[Features] Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns.")
    return X, y, feature_cols


# -----------------------------------------------------------------------------
# SECTION 4 – VISUALISATIONS (Activity 1 – Task 1 requirements)
# Re-creates both histograms and the actual-vs-predicted scatter plot.
# -----------------------------------------------------------------------------

def plot_distributions(df):
    """
    Two histograms required by Activity 1 Task 1:
      1. Distribution of metascore (the outcome variable)
      2. Distribution of userscore (the primary input variable)
    Saved as PNG files so they can be included in the report.
    """
    # Histogram 1 – Metascore distribution
    fig1, ax1 = plt.subplots()
    ax1.hist(df['metascore'], bins=20, color='skyblue', edgecolor='black')
    ax1.set_title("Distribution of Meta Scores")
    ax1.set_xlabel("Meta Score")
    ax1.set_ylabel("Number of Games")
    plt.tight_layout()
    plt.savefig("plot_metascore_dist.png", dpi=100)
    plt.close(fig1)
    print("[Plot] Saved: plot_metascore_dist.png")

    # Histogram 2 – Userscore distribution
    fig2, ax2 = plt.subplots()
    ax2.hist(df['userscore'], bins=20, color='salmon', edgecolor='black')
    ax2.set_title("Distribution of User Scores")
    ax2.set_xlabel("User Score")
    ax2.set_ylabel("Number of Games")
    plt.tight_layout()
    plt.savefig("plot_userscore_dist.png", dpi=100)
    plt.close(fig2)
    print("[Plot] Saved: plot_userscore_dist.png")


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Actual vs. predicted scatter plot from Activity 1 Task 2.
    Purple dots show each prediction; the red dashed line is the
    perfect-prediction diagonal.  Saved as PNG for the report.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, color='purple')
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title("Linear Regression: Actual vs. Predicted Metascores")
    ax.set_xlabel("Actual Metascore")
    ax.set_ylabel("Predicted Metascore")
    plt.tight_layout()
    plt.savefig("plot_actual_vs_predicted.png", dpi=100)
    plt.close(fig)
    print("[Plot] Saved: plot_actual_vs_predicted.png")


# -----------------------------------------------------------------------------
# SECTION 5 – TRAIN MODEL (Activity 1 – Task 2)
# LinearRegression, 80/20 split, random_state=42 — identical to Activity 1.
# -----------------------------------------------------------------------------

def train_model(X, y):
    """
    Split 80 % training / 20 % testing (random_state=42, same as Activity 1),
    then fit a LinearRegression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[Model] LinearRegression trained successfully.")
    return model, X_test, y_test


# -----------------------------------------------------------------------------
# SECTION 6 – EVALUATE MODEL
# Prints MSE and R² exactly as in Activity 1.
# -----------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate with Mean Squared Error and R-squared.
    These are the same two metrics reported in Activity 1.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print("\n[Evaluation] Linear Regression Results:")
    print(f"  Mean Squared Error : {round(mse, 2)}")
    print(f"  R-squared          : {round(r2, 2)}")
    return y_pred


# -----------------------------------------------------------------------------
# SECTION 7 – SAVE ARTIFACTS
# Two files are persisted for the Activity 2 agent:
#   model.pkl        – the fitted LinearRegression object
#   feature_cols.pkl – ordered list of column names the model was trained on
#
# The feature column list is critical: the agent must build an input vector
# with exactly the same columns in exactly the same order.  Without this list,
# the model would receive a misaligned or incorrectly sized array and crash.
# -----------------------------------------------------------------------------

def save_artifacts(model, feature_cols,
                   model_path="model.pkl",
                   cols_path="feature_cols.pkl"):
    """Save the trained model and feature column list."""
    joblib.dump(model,        model_path)
    joblib.dump(feature_cols, cols_path)
    print(f"[Saved] model        → {model_path}")
    print(f"[Saved] feature_cols → {cols_path}")


# -----------------------------------------------------------------------------
# MAIN – run the full Activity 1 pipeline when this file is executed directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_dataset("metacritic_games.csv")
    df = clean_data(df)
    plot_distributions(df)
    X, y, feature_cols = engineer_features(df)
    model, X_test, y_test = train_model(X, y)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_actual_vs_predicted(y_test, y_pred)
    save_artifacts(model, feature_cols)
