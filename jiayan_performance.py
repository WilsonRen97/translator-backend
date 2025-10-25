import time
import pandas as pd
from jiayan import load_lm, CRFPunctuator
import difflib
from sklearn.model_selection import train_test_split


def punctuate(text):
    lm = load_lm('./jiayan_models/jiayan.klm')
    punctuator = CRFPunctuator(lm, './jiayan_models/cut_model')
    punctuator.load('./jiayan_models/punc_model')
    return punctuator.punctuate(text)


def compute_accuracy(pred, gold):
    matcher = difflib.SequenceMatcher(None, pred, gold)
    return matcher.ratio()


def extract_punct_positions(text, punct_set="，。！？；：、,.!?;:"):
    return [(i, c) for i, c in enumerate(text) if c in punct_set]


def compute_punctuation_accuracy(pred, gold):
    pred_puncts = extract_punct_positions(pred)
    gold_puncts = extract_punct_positions(gold)

    pred_set = set(pred_puncts)
    gold_set = set(gold_puncts)

    true_pos = len(pred_set & gold_set)
    false_pos = len(pred_set - gold_set)
    false_neg = len(gold_set - pred_set)

    precision = true_pos / \
        (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) else 0

    return precision, recall, f1


def process_csv(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip')

    if not {'original', 'no_punctuation'}.issubset(df.columns):
        raise ValueError(
            "CSV must have columns: 'original' and 'no_punctuation'")

    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # take 1% subset for quick eval
    test_df = test_df.sample(frac=0.01, random_state=42)

    total_char_acc = 0
    total_p, total_r, total_f1 = 0, 0, 0
    n = 0

    for _, row in test_df.iterrows():
        original = str(row['original'])
        no_punc = str(row['no_punctuation'])

        generated = punctuate(no_punc)

        # Character-level
        acc = compute_accuracy(generated, original)
        # Punctuation-level
        p, r, f1 = compute_punctuation_accuracy(generated, original)

        total_char_acc += acc
        total_p += p
        total_r += r
        total_f1 += f1
        n += 1
        print('Processed sample:', n)

        # print(f"Original:  {original}")
        # print(f"Generated: {generated}")
        # print(f"Char Accuracy: {acc:.3f}")
        # print(f"Punctuation Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        # print("-" * 60)

    # Summary
    print("\n===== Overall Evaluation =====")
    print(f"Total Samples    : {n}")
    print(f"Avg Char Accuracy : {total_char_acc / n:.3f}")
    print(f"Avg Precision     : {total_p / n:.3f}")
    print(f"Avg Recall        : {total_r / n:.3f}")
    print(f"Avg F1 Score      : {total_f1 / n:.3f}")


if __name__ == "__main__":
    csv_file_path = './output_no_punctuation.csv'
    # time the code below
    start_time = time.time()
    process_csv(csv_file_path)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
