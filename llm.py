import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from rag_db import retrieve_information
from jiayan_token import tokenize_text

login('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')

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# print the max length of input tokens
print(f"Max input length: {tokenizer.model_max_length} tokens")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

model.to(device)
print("Model and tokenizer loaded successfully.")


def rag_response(classical_text, modern_text):
    keywords = tokenize_text(classical_text)
    retrieved_info = retrieve_information(keywords)
    retrieved_text = "\n".join(
        [f"{i+1}. {info}" for i, info in enumerate(retrieved_info)])

    print(f"Retrieved Information:\n{retrieved_text}")

    prompt = f"""你是一個專業的文言文翻譯助手。
        請根據下列三個部分的資訊，產出準確、流暢的現代中文翻譯：
        1. 古文原文：需要被翻譯的文本。
        2. 翻譯參考資料：可能包含相關背景或字詞解釋。
        3. 別人的翻譯結果：其他人嘗試的翻譯，可作為參考但絕對不可直接照抄。

        請在理解以上資料後，寫出你自己的現代中文翻譯。
        ---
        古文原文：
        {classical_text}

        別人的翻譯結果：
        {modern_text}

        翻譯參考資料：
        {retrieved_text}
        ---
        你的翻譯結果（不可照抄）：
        """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=40,
                             pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove the prompt part from the response
    response = response.replace(prompt, "").strip()
    return response


if __name__ == "__main__":
    ancient_text = "陆草木之花，可爱者甚蕃。晋陶渊明独爱菊。自李唐来，世人甚爱牡丹。"
    modern_text = "现代人对花卉的喜好各有不同，陶渊明独爱菊花，而牡丹则是历代文人所推崇的花卉。"
    response = rag_response(ancient_text, modern_text)
    print(f"RAG Response: {response}")
