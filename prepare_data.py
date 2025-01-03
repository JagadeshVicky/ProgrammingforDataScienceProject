import uuid
import chromadb
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks

# Instead of using Settings, directly instantiate a PersistentClient
client = chromadb.PersistentClient(path="chroma_db")

# If collection does not exist, create it
try:
    collection = client.get_collection("domain_knowledge")
except:
    collection = client.create_collection("domain_knowledge")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load domain-specific text file
with open("payment_request_spec.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunk the text
chunks = chunk_text(text, chunk_size=200, overlap=50)

# Embed each chunk and store in Chroma
for chunk in chunks:
    vector = embedding_model.encode([chunk])[0].tolist()
    doc_id = str(uuid.uuid4())
    collection.add(documents=[chunk], embeddings=[vector], ids=[doc_id])

print("Data preparation complete. Vector store populated.")
