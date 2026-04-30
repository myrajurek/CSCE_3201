# =============================================================================
# evaluation_reflection.py
# Task 5 – Evaluation and Reflection
#
# PURPOSE:
#   Prints the complete Task 5 deliverable:
#     • One identified strength of the agent
#     • One identified failure case
#     • Two reflection paragraphs covering:
#         – what was learned from extending the ML project with an agent
#         – the main challenges in connecting the agent to Phase 1
#         – how the system could be improved in future
#     • A summary table for the written report
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 1 – STRENGTH
# Describes one thing the agent did well.
# -----------------------------------------------------------------------------

STRENGTH = {
    "title": "Consistent input validation and guided data collection",
    "detail": (
        "When users omitted one or more of the three required inputs "
        "(userscore, platform, or genre), the agent reliably identified "
        "the missing field and asked only for that specific value — it never "
        "re-asked for information already provided in the conversation. "
        "This made the interaction feel natural and efficient, which is "
        "important because the model requires all three inputs to produce a "
        "meaningful prediction. Without this guidance, a non-technical user "
        "would likely send incomplete data and receive an error."
    )
}


# -----------------------------------------------------------------------------
# SECTION 2 – FAILURE CASE
# Describes one situation where the agent performed poorly.
# -----------------------------------------------------------------------------

FAILURE_CASE = {
    "title": "Platform strings not in training data produce silent misses",
    "detail": (
        "When a user supplied a platform name that did not appear in the "
        "training dataset (for example 'Steam Deck' or 'PlayStation 5'), "
        "no matching one-hot column existed in feature_cols, so the platform "
        "value was silently ignored and the prediction was based only on "
        "userscore and genre. "
        "The agent did not warn the user that their platform was unrecognised, "
        "meaning the returned metascore appeared more confident than it should "
        "have been. This is a real failure because platform can be a meaningful "
        "signal — a game released exclusively on a niche platform often receives "
        "different critical treatment than one on a major console."
    )
}


# -----------------------------------------------------------------------------
# SECTION 3 – REFLECTION PARAGRAPHS
# Two paragraphs covering: what was learned, main challenges, future improvements.
# -----------------------------------------------------------------------------

REFLECTION = """
REFLECTION
----------

Building an AI agent on top of the Phase 1 Linear Regression model revealed
how different the challenges of deployment are from the challenges of modelling.
In Activity 1, success meant minimising MSE and maximising R²; in Activity 2,
success meant making the model accessible and trustworthy to a user who may
never have heard of R-squared. The most important thing learned was that a model
is only as useful as the interface that wraps it — a perfectly trained model
hidden behind a confusing input process serves nobody. Designing the agent's
system prompt to ask exactly for userscore, platform, and genre (and nothing
else) forced a clear understanding of what features the Phase 1 model actually
depends on, which in turn deepened understanding of the original pipeline.

The main technical challenge was aligning new user input with the encoding
produced by Activity 1's pipeline. Because Activity 1 used pd.get_dummies on
the entire training dataset, platform column names like "platforms_PlayStation 4"
and genre column names like "Action" were generated at training time and saved
as a list (feature_cols.pkl). Any input that used a slightly different string —
"PS4" instead of "PlayStation 4", or "Action-Adventure" instead of "Action" —
would silently produce a zero in that column rather than raising an error. A
future version should add a normalisation step that maps common synonyms to the
canonical training-set values before encoding. Additionally, the Linear Regression
model produces a single point estimate with no uncertainty measure; a future
improvement would be to add a prediction interval (e.g. from a quantile
regression or bootstrap ensemble) so the agent can tell users "the predicted
score is 74, but the typical range for games like this is 65–83", giving a much
more honest and actionable answer.
"""


# -----------------------------------------------------------------------------
# SECTION 4 – PRINT THE EVALUATION REPORT
# -----------------------------------------------------------------------------

def wrap(text, width=67, indent="  "):
    """Simple word-wrapper for console output."""
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(indent + line)
    return "\n".join(out)


def print_summary_table():
    """Quick-reference table for the written report."""
    rows = [
        ("Project Title",         "Metacritic Score Prediction Agent"),
        ("Dataset",               "metacritic_games.csv"),
        ("Prediction Task",       "Predict metascore (0–100) for a video game"),
        ("Target Variable",       "metascore  (continuous, regression)"),
        ("Phase 1 Model",         "LinearRegression (scikit-learn)"),
        ("Preprocessing",         "tbd→NaN coerce, genre multi-hot, platform one-hot"),
        ("Train / Test Split",    "80 % / 20 %  (random_state=42)"),
        ("Evaluation Metrics",    "MSE and R-squared"),
        ("Agent Task",            "Collect inputs → encode → predict → explain"),
        ("Deployed Chat Model",   "gpt-4o (Azure AI Foundry)"),
        ("Test 1",                "High-rated RPG on PS4 → Generally Favorable"),
        ("Test 2",                "Low-rated iOS Puzzle  → Mixed or Average"),
        ("Test 3",                "High user score Strat/Sim on PC (borderline)"),
        ("Strength",              STRENGTH["title"]),
        ("Failure Case",          FAILURE_CASE["title"]),
    ]
    sep = "-" * 70
    print(f"\n{sep}")
    print("  AGENT SUMMARY TABLE")
    print(sep)
    for label, value in rows:
        print(f"  {label:<28} : {value}")
    print(sep)


def print_evaluation_report():
    """Full Task 5 evaluation and reflection output."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  TASK 5 – EVALUATION AND REFLECTION")
    print(sep)

    print(f"\n  ★  STRENGTH: {STRENGTH['title']}\n")
    print(wrap(STRENGTH['detail']))

    print(f"\n  ✗  FAILURE CASE: {FAILURE_CASE['title']}\n")
    print(wrap(FAILURE_CASE['detail']))

    print()
    for line in REFLECTION.strip().split("\n"):
        if line.strip() == "":
            print()
        elif line.startswith("REFLECTION") or line.startswith("---"):
            print(f"  {line}")
        else:
            print(wrap(line))

    print(f"\n{sep}\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print_summary_table()
    print_evaluation_report()
