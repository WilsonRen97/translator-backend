import chinese_converter
from flask import Flask, request, jsonify
from flask_cors import CORS
from translator import translate_classical_to_modern
from jiayan_punc import punctuate
from bert_punctuator_files.bert_punc_testing import predict_punctuation
from rag_db import retrieve_information
from jiayan_token import tokenize_text
from dbscan_util import cluster_data

app = Flask(__name__)
CORS(app)


def convert_to_traditional_chinese(text):
    return chinese_converter.to_traditional(text)


def convert_to_simplified_chinese(text):
    return chinese_converter.to_simplified(text)

# add a get route for health check


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/punctuate", methods=["POST"])
def punctuate_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    print(f"Received text for punctuation: {text}")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # punctuated_text = punctuate(text)
        punctuated_text = predict_punctuation(text)
        print(f"Punctuated text: {punctuated_text}")
        return jsonify({"punctuated": punctuated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    ancient_text = data.get("text", "").strip()
    print(f"Received text for translation: {ancient_text}")
    if not ancient_text:
        return jsonify({"error": "No text provided"}), 400

    # tokenize ancient_text
    keywords = tokenize_text(ancient_text)
    print(f"Tokenized keywords: {keywords}")
    info = retrieve_information(keywords)
    print(f"Retrieved information: {info}")

    try:
        simplified_text = convert_to_simplified_chinese(ancient_text)
        modern_text = translate_classical_to_modern(simplified_text)
        traditional_text = convert_to_traditional_chinese(modern_text)
        print(f"Translated text: {traditional_text}")
        return jsonify({"translation": traditional_text, "retrieved_info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jsonData", methods=["POST"])
def json_data():
    data = request.get_json()
    listOfData = data['textObjects']
    # example item in listOfData:
    # {'id': 82, 'x': 1687, 'y': 1621, 'width': 44, 'height': 259, 'text': '子字孔桂', 'confidence': 0.766}
    clustered_data = cluster_data(listOfData)
    return jsonify({"status": "received", "data": clustered_data}), 200

# @app.route("/rag", methods=["POST"])
# def rag():
#     data = request.get_json()
#     classical_text = data.get("classical_text", "").strip()
#     modern_text = data.get("modern_text", "").strip()
#     print(f"Received classical text: {classical_text}")
#     print(f"Received modern text: {modern_text}")

#     if not classical_text or not modern_text:
#         return jsonify({"error": "Both classical and modern texts must be provided"}), 400

#     try:
#         response = rag_response(classical_text, modern_text)
#         print(f"RAG response: {response}")
#         return jsonify({"response": response})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
