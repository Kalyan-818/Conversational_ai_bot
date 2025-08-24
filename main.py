import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

# Initialize local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings:
    def embed_documents(self, texts):
        return embedding_model.encode(texts).tolist()

    def embed_query(self, text):
        return embedding_model.encode([text])[0].tolist()

    def __call__(self, text):
        return self.embed_query(text)

# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding token issue
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def remove_repeated_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
        else:
            break  # Stop adding sentences after first repetition
    return ' '.join(unique_sentences)

def clean_response(response_text, prompt):
    # Remove the prompt from response
    stripped_response = response_text.replace(prompt, '').strip()
    # Remove repeated sentences caused by repetition in generation
    cleaned = remove_repeated_sentences(stripped_response)
    return cleaned

def generate_response(prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,   # Use max_new_tokens instead of max_length
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_response(response, prompt)
    return cleaned if cleaned else "Sorry, I don't have an answer for that."

def load_docs(path='docs'):
    texts = []
    folder_path = os.path.abspath(path)
    print(f"Looking for files in folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return texts

    files = os.listdir(folder_path)
    print(f"Files found: {files}")
    filepath = os.path.join(folder_path, 'sample.txt')
    if os.path.exists(filepath):
        print(f"Loading file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    else:
        print(f"File {filepath} not found.")
    return texts

def main():
    print("Loading documents...")
    documents = load_docs()

    if len(documents) == 0:
        print("No documents loaded. Exiting.")
        return

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks locally...")
    embeddings = LocalEmbeddings()

    print("Creating FAISS vector store...")
    db = FAISS.from_texts(chunks, embeddings)

    print("Bot ready! Type 'exit' to quit.")
    chat_history = []

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break

        results = db.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in results])

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = generate_response(prompt)
        print("Bot:", answer)
        chat_history.append((query, answer))


if __name__ == '__main__':
    main()
