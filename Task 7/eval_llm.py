import csv, json, os
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "Tasks 2 & 4", "data", "analysis")
LABELS = ("entailment", "neutral", "contradiction")

SYSTEM = (
    "Decide the relationship between the premise and hypothesis. For entailment, the hypothesis must be true given the premise. For contradiction, the hypothesis must be false given the premise. For neutral, the hypothesis is neither guaranteed true nor false. Answer with exactly one word and NOTHING else: entailment, neutral, or contradiction."
)

def parse(text):
    t = text.strip().lower()
    for label in LABELS:
        if label in t:
            return label
    if "entail" in t: return "entailment"
    if "contradict" in t: return "contradiction"
    return None

# * premise & hypothesis pairs
lines = [l.rstrip("\n") for l in open(os.path.join(SRC, "snli_1.0_dev.tok"))]

# * groundtruth values
gold = [r["gt"] for r in csv.DictReader(open(os.path.join(SRC, "preds", "6_snli_1.0_dev.csv")))]

model, tok = load("mlx-community/Qwen3-4B-4bit")
correct = 0
with open(os.path.join(HERE, "llm_preds.jsonl"), "w") as out:
    for i, g in enumerate(gold):
        premise, hyp = lines[2 * i], lines[2 * i + 1]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hyp}\nWhat is the relationship between the two? Remember to only answer ONE word."}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        pred = parse(generate(model, tok, prompt=prompt, max_tokens=8, verbose=False))
        out.write(json.dumps({"premise": premise, "hypothesis": hyp, "gold": g, "pred": pred}) + "\n")
        out.flush()
        if pred == g: 
            correct += 1

n = len(gold)
labeled = sum(1 for g in gold if g in LABELS)
print(f"Labeled-only Qwen Accuray: {correct}/{labeled} = {correct/labeled:.4f}")
print("Bowman/LSTM reference (labeled-only): 0.7877")
