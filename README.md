# Domain-Tuned Retriever for HR Policy RAG

**Live demo:** https://domain-rag-retriever-qwxz2q7nxyjvzq9zlau3dq.streamlit.app/

**Model:** https://huggingface.co/Pranay0514/hr-bge-small-ft

Fine-tuned a BGE-small bi-encoder on an HR-policy corpus (contrastive loss) to improve retrieval for RAG.

## Results (held-out test chunks)
| Model | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
|---|---|---|---|---|
| Base BGE-small | 0.375 | 0.798 | 0.904 | 0.631 |
| Fine-tuned | 0.510 | 0.875 | 0.981 | 0.735 |
| + Hard negatives (ablation) | 0.500 | 0.885 | 0.981 | 0.731 |

Hard negatives gave no gain: near-duplicate chunks across companies create false negatives.

## Method
1. Generated 179 HR policy chunks (Groq LLM) across 30 topics and company types
2. Generated 3 specific questions per chunk (LLM)
3. Chunk-level train/test split (no leakage)
4. Fine-tuned BAAI/bge-small-en-v1.5 with MultipleNegativesRankingLoss (3 epochs, Colab T4)
5. Evaluated with a FAISS flat index
6. Deployed a side-by-side base vs fine-tuned demo (Streamlit), model hosted on Hugging Face Hub

## Limitations
Corpus and queries are LLM-generated, and the test set is small (about 100 queries). Real-world gains may differ.

## Run
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key   # PowerShell: $env:GROQ_API_KEY="your_key"
python src/make_chunks.py
python src/make_queries.py
python src/baseline_eval.py
streamlit run src/app.py
```