import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import os
from urllib.parse import urlparse
import re

st.set_page_config(page_title="FAQ Retrieval System", layout="wide")


# Load model once for efficiency
@st.cache_resource
def load_model():
    # This model is pre-trained to understand the semantic meaning of sentences.
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Load FAQ data and create embeddings
@st.cache_data(show_spinner=False)
def load_faqs(website_name: str):

    path = f"QnA/{website_name}.jsonl"
    if not os.path.exists(path):
        st.error(f"Error: No data found for {website_name}.jsonl in the QnA folder.")
        st.info("Please make sure you have extracted FAQs using app.py and they are in the correct folder.")
        return [], [], None

    faqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                faqs.append(json.loads(line))
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON on line: {e}")
                continue

    questions = [faq.get("question", "") for faq in faqs]
    answers = [faq.get("answer", "") for faq in faqs]
    
    # Generate embeddings for all questions
    embeddings = model.encode(questions, convert_to_tensor=True)

    return questions, answers, embeddings

# Search for the best answer
def retrieve_answer(query, questions, answers, question_embeddings, top_k=2):
    
    if question_embeddings is None or not question_embeddings.numel():
        return []

    # Encode the user query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Perform a semantic search to find the most similar questions
    hits = util.semantic_search(query_embedding, question_embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        idx = hit["corpus_id"]
        score = hit["score"]
        # Retrieve the corresponding question and answer
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(score)
        })
    return results

def get_site_name(url):
    netloc = urlparse(url).netloc
    return re.sub(r"[^\w]+", "_", netloc)


# ---------------- Streamlit UI ----------------

st.title("FAQ Retrieval System")
st.markdown("Select a website domain and ask your question. The system will retrieve the most relevant answers.")

# Auto-detect all available FAQ files in QnA folder
qna_folder = "QnA"
available_domains = []
if os.path.exists(qna_folder):
    for f in os.listdir(qna_folder):
        if f.endswith(".jsonl"):
            available_domains.append(f.replace(".jsonl", ""))

if not available_domains:
    st.error("No FAQ data found in the QnA folder. Please run the crawler first.")
else:
    # Selectbox for domain choice
    website_name = st.selectbox("Select Website Domain:", available_domains)

    questions, answers, question_embeddings = load_faqs(website_name)

    if questions:
        query = st.text_input("Ask your question:")
        if query:
            with st.spinner("Searching for an answer..."):
                results = retrieve_answer(query, questions, answers, question_embeddings, top_k=5)

            if results:
                st.subheader("Most Relevant Answers")
                for r in results:
                    st.markdown(f"**Question:** {r['question']}")
                    st.markdown(f"**Answer:** {r['answer']}")
                    st.markdown(f"_(Similarity Score: {r['score']:.2f})_")
                    st.markdown("---")
            else:
                st.warning("Sorry, I could not find a relevant answer.")
