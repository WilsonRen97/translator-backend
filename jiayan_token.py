from jiayan import load_lm
from jiayan import CharHMMTokenizer


def tokenize_text(text):
    lm = load_lm('jiayan_models/jiayan.klm')
    tokenizer = CharHMMTokenizer(lm)
    return list(tokenizer.tokenize(text))


if __name__ == "__main__":
    sample_text = "是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉以自为方。"
    tokens = tokenize_text(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Tokenized text: {tokens}")
