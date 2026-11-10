import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

PREFIX = "Represent this sentence for searching relevant passages: "

st.set_page_config(page_title="HR Retriever Demo", layout="wide")
st.title("HR Policy Retriever: Base vs Fine-tuned")
st.caption("BGE-small fine-tuned on a synthetic HR-policy corpus")

@st.cache_resource
def load():
    chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
    texts = [c["text"] for c in chunks]
    out = {}
    for name, path in [("Base", "BAAI/bge-small-en-v1.5"), ("Fine-tuned", "hr-bge-small-ft")]:
        m = SentenceTransformer(path)
        emb = m.encode(texts, normalize_embeddings=True)
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(np.array(emb, dtype="float32"))
        out[name] = (m, idx)
    return chunks, out

chunks, models = load()

query = st.text_input("Ask an HR question", "How many days of casual leave in a bank?")
k = st.slider("Top-k", 1, 5, 3)

if query:
    cols = st.columns(2)
    for col, name in zip(cols, ["Base", "Fine-tuned"]):
        m, idx = models[name]
        q = m.encode([PREFIX + query], normalize_embeddings=True)
        scores, ids = idx.search(np.array(q, dtype="float32"), k)
        with col:
            st.subheader(name)
            for s, i in zip(scores[0], ids[0]):
                c = chunks[i]
                st.markdown(f"**{c['source']}**  (score {s:.3f})")
                st.write(c["text"][:400] + "...")
                st.divider()