from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch


# Global model and embeddings
model_embedding = None
passages = None
doc_embeddings = None
tokenizer = None
model = None

device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)


def compute_embeddings():
    model_embedding = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

    file_path = "processed_text.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        passages = [line.strip() for line in file if line.strip()]

    doc_embeddings = model_embedding.encode(passages, task="retrieval.passage")

    return model_embedding, passages, doc_embeddings

def retrieve_relevant_passages(query):
    query_embedding = model_embedding.encode(query, task="retrieval.query")
    similarities = model_embedding.similarity(query_embedding, doc_embeddings)[0]
    top_k = 3
    top_k_indices = torch.topk(similarities, k=top_k, dim=0).indices.tolist()
    relevant_passages = []
    for rank, idx in enumerate(top_k_indices, start=1):
        relevant_passages.append(passages[idx])
    return relevant_passages

def get_answer(question):
    context = retrieve_relevant_passages(question, )
    response = generate_answer(question, context)

    def keep_after_keyword(text, keyword):
        parts = text.split(keyword, 1)
        return parts[1] if len(parts) > 1 else ""

    print("Question:", question)
    print("Réponse:", keep_after_keyword(response, "Réponse: "))

    return keep_after_keyword(response, "Réponse: ")

def generate_answer(question, context):
    prompt = f"""
    Tu es un assistant IA français. Réponds toujours en français.

    Voici la question de l'utilisateur: "{question}"

    Contexte pertinent pour répondre: "{" ".join(context)}"

    Réponds manière courte et concise avec seulement le contexte donné. Si le contexte ne permet pas de répondre, dis "Le document ne permet pas de répondre".

    Évite de générer du texte inutile.

    Réponse:"""

    # Tokenize & Generate Response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    print("Received messages:", messages)  # Check the format in the terminal

    question = messages[-1]["content"] # Get the latest user message
    print("Received question:", question)
    response_text = get_answer(question)

    return jsonify({"content": response_text})


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print("Start embedding...")
    model_embedding, passages, doc_embeddings = compute_embeddings()
    print("Embedding done.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Optimized computation precision
        bnb_4bit_use_double_quant=True,  # Further memory reduction
        bnb_4bit_quant_type="nf4"  # Normalized float 4-bit
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        # quantization_config=quantization_config
    )

    print("LLM loaded successfully with 4-bit quantization!")

    app.run(host="0.0.0.0", port=8080, debug=False)
