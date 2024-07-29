import pandas as pd
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import SearchRequest
import requests

# Load the data
df = pd.read_csv('top_rated_wines.csv')
records = df.to_dict('records')

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
texts = [record['notes'] for record in records]
embeddings = model.encode(texts, show_progress_bar=True)

# Initialize Qdrant client
client = qdrant_client.QdrantClient(":memory:")

# Create a collection in Qdrant
client.recreate_collection(
    collection_name="wine_descriptions",
    vectors_config=qdrant_client.http.models.VectorParams(size=embeddings.shape[1], distance=qdrant_client.http.models.Distance.COSINE)
)

# Upload the embeddings to Qdrant
client.upload_collection(
    collection_name="wine_descriptions",
    vectors=embeddings,
    payload=records,
    batch_size=100
)

print("Embeddings created and uploaded to Qdrant successfully.")

def search_qdrant(query_text, collection_name="wine_descriptions", top_k=5):
    # Encode the query text to get the query vector
    query_vector = model.encode([query_text])[0]
    
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return search_result

def generate_completion(prompt, model="phi3"):
    base_url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_thread": 8,
            "num_ctx": 2024
        }
    }
    response = requests.post(base_url, json=payload)
    return response.json()

# Example usage
query_text = "Best wine with fruity flavor"
search_results = search_qdrant(query_text)
print("Search results from Qdrant:")
for result in search_results:
    print(result)

prompt = "Using the information from the search results, write a description of a perfect wine pairing."
completion = generate_completion(prompt)
print("Generated completion from LLM:")
print(completion)
