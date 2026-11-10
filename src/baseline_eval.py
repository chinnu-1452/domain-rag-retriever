import json, random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

random.seed(42)

chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
pairs = [json.loads(l) for l in open("data/train_pairs.jsonl", encoding="utf-8")]

# Split by CHUNK (not by query) so there is no leakage
ids = [c["id"] for c in chunks]
random.shuffle(ids)
cut = int(0.8 * len(ids))
train_ids, test_ids = set(ids[:cut]), set(ids[cut:])
json.dump({"train": sorted(train_ids), "test": sorted(test_ids)}, open("data/split.json", "w"))

test_pairs = [p for p in pairs if p["chunk_id"] in test_ids]
print("chunks:", len(chunks), "| test queries:", len(test_pairs))

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
texts = [c["text"] for c in chunks]
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(np.array(emb, dtype="float32"))

prefix = "Represent this sentence for searching relevant passages: "
q = model.encode([prefix + p["query"] for p in test_pairs], normalize_embeddings=True)
_, top = index.search(np.array(q, dtype="float32"), 10)

id_list = [c["id"] for c in chunks]
r1 = r5 = r10 = ndcg = 0
for p, row in zip(test_pairs, top):
    ranked = [id_list[i] for i in row]
    if p["chunk_id"] in ranked:
        rank = ranked.index(p["chunk_id"]) + 1
        r1 += rank <= 1; r5 += rank <= 5; r10 += rank <= 10
        ndcg += 1 / np.log2(rank + 1)

n = len(test_pairs)
print(f"Recall@1: {r1/n:.3f}  Recall@5: {r5/n:.3f}  Recall@10: {r10/n:.3f}  nDCG@10: {ndcg/n:.3f}")