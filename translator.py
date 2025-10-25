from transformers import BartForConditionalGeneration, AutoTokenizer
import re
import torch


def beautify_text(string):
    cleaned_str = re.sub(r'\[unused2\]|\s+', '', string)
    return cleaned_str


model = BartForConditionalGeneration.from_pretrained(
    "bart_from_scratch_model_8")
tokenizer = AutoTokenizer.from_pretrained("bart_fine_tune_tokenizer")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

model.to(device)

print("Model and tokenizer loaded successfully.")


def translate_classical_to_modern(input_text):
    inputs = tokenizer(input_text, return_token_type_ids=False,
                       return_tensors="pt", max_length=128, truncation=True).to(device)
    # outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    max_length = len(inputs["input_ids"][0]) + 10
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=7,
        early_stopping=True,
        length_penalty=2.0,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=40,
        top_p=0.8
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return beautify_text(translated_text)


if __name__ == "__main__":
    classical_text = "水陆草木之花，可爱者甚蕃。晋陶渊明独爱菊。自李唐来，世人甚爱牡丹。"
    modern_translation = translate_classical_to_modern(classical_text)

    print(f"Classical Chinese: {classical_text}")
    print(f"Modern Chinese Translation: {modern_translation}")
