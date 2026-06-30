import json
import pandas as pd
import re

# Load JSON file
data = pd.read_excel("result/manual analysis/manual analysis.xlsx").to_dict(orient="records")

with open("pure_misjudgement_ids.txt", 'r') as f:
    target_ids = {int(line.strip()) for line in f}


data = [item for item in data if item.get("data_id") in target_ids]
print(f"Loaded {len(data)} items")

updated_data=data

# Convert to DataFrame
df = pd.DataFrame(data)

print(f"DataFrame shape: {df.shape}")
print(f"Head of DataFrame:\n{df.head()}")



# Extract actual verdict from evaluated_array
def extract_actual_verdict(text):
    if pd.isna(text):
        return None

    matches = re.findall(r'(\d+)\s+(AC|WA|CE|RE|TLE)', str(text))
    # we can have multiple matches as well 201 AC, 69 WA, 0 CE, 8 RE, 0 TLE


    # Filter out matches with zero counts
    matches = [match for match in matches if int(match[0]) > 0]

    if not matches:
        return None

    # Choose verdict with highest count
    verdict_priority = {'CE': 5, 'RE': 4, 'TLE': 3, 'WA': 2, 'AC': 1}
    highest_priority_verdict = max(matches, key=lambda x: verdict_priority[x[1]])

    return highest_priority_verdict[1]

# df["actual_verdict"] = df["evaluated_array"].apply(extract_actual_verdict)

#save this in the excel file
df.to_excel("result/manual analysis/pure misjudgement.xlsx", index=False)

# ============================================================
# 1. Predicted Verdict Counts
# ============================================================

print("\n=== Predicted Verdict Counts ===")
print(df["llm_verdict"].value_counts())

# ============================================================
# 2. Actual Verdict Counts
# ============================================================

print("\n=== Actual Verdict Counts ===")
print(df["actual_verdict"].value_counts())

# ============================================================
# 3. Confusion Matrix
# ============================================================

confusion = pd.crosstab(
    df["actual_verdict"],
    df["llm_verdict"],
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False
)

print("\n=== Actual vs Predicted Verdicts ===")
print(confusion)

# ============================================================
# 4. Pair Counts (AC->WA, CE->WA, etc.)
# ============================================================

pairs = (
    df.groupby(["actual_verdict", "llm_verdict"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

print("\n=== Actual -> Predicted Counts ===")

for _, row in pairs.iterrows():
    print(
        f"{row['actual_verdict']} -> "
        f"{row['llm_verdict']} = "
        f"{row['count']}"
    )

# ============================================================
# 5. Misclassifications Only
# ============================================================

print("\n=== Misclassifications Only ===")

errors = pairs[
    pairs["actual_verdict"] != pairs["llm_verdict"]
]

for _, row in errors.iterrows():
    print(
        f"{row['actual_verdict']} -> "
        f"{row['llm_verdict']} = "
        f"{row['count']}"
    )