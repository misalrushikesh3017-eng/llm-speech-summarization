import pandas as pd
from rouge_score import rouge_scorer

# CSV columns required:
# reference,prompt1,prompt2
df = pd.read_csv("results.csv")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rows = []
for _, row in df.iterrows():
    s1 = scorer.score(str(row["reference"]), str(row["prompt1"]))
    s2 = scorer.score(str(row["reference"]), str(row["prompt2"]))
    rows.append({
        "Prompt 1 ROUGE-1": s1["rouge1"].fmeasure,
        "Prompt 1 ROUGE-2": s1["rouge2"].fmeasure,
        "Prompt 1 ROUGE-L": s1["rougeL"].fmeasure,
        "Prompt 2 ROUGE-1": s2["rouge1"].fmeasure,
        "Prompt 2 ROUGE-2": s2["rouge2"].fmeasure,
        "Prompt 2 ROUGE-L": s2["rougeL"].fmeasure,
    })

out = pd.DataFrame(rows)
print(out.round(4))
print("\nMean scores:")
print(out.mean(numeric_only=True).round(4))
out.to_csv("rouge_results.csv", index=False)
