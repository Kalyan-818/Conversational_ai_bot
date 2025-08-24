import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import re

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings:
    def embed_documents(self, texts):
        return embedding_model.encode(texts).tolist()

    def embed_query(self, text):
        return embedding_model.encode([text])[0].tolist()

    def __call__(self, text):
        return self.embed_query(text)

# GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def remove_repeated_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
        else:
            break
    return ' '.join(unique_sentences)

def clean_response(response_text, prompt):
    stripped_response = response_text.replace(prompt, '').strip()
    cleaned = remove_repeated_sentences(stripped_response)
    return cleaned

def generate_response(prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
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
    if not os.path.exists(folder_path):
        st.error(f"Folder {folder_path} does not exist.")
        return texts
    filepath = os.path.join(folder_path, 'sample.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    else:
        st.error(f"File {filepath} not found.")
    return texts

@st.cache_data(show_spinner=True)
def setup_faiss_index():
    documents = load_docs()
    if len(documents) == 0:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    embeddings = LocalEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    return db

# Streamlit UI
def main():
    st.set_page_config(page_title="Conversational AI Chat Bot", page_icon="🤖", layout="centered")
    st.title("🤖 Conversational AI Search Bot")

    db = setup_faiss_index()
    if db is None:
        st.warning("No documents loaded. Please add text file in docs/sample.txt")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat message history using Streamlit chat API
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about your document..."):
        # Display user message immediately
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve relevant docs and generate response
        with st.spinner("Bot is thinking..."):
            results = db.similarity_search(prompt, k=3)
            context = " ".join([doc.page_content for doc in results])
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
            answer = generate_response(full_prompt)

        # Display bot response and update chat history
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
