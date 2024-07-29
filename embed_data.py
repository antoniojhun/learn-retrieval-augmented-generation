import pandas as pd
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http import models

# Load the data
df = pd.read_csv('top_rated_wines.csv')

# Print the DataFrame's columns to inspect the data
print("DataFrame columns:", df.columns)

# Display the first few rows of the DataFrame to understand the structure
print(df.head())

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
    vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE)
)

# Upload the embeddings to Qdrant
client.upload_collection(
    collection_name="wine_descriptions",
    vectors=embeddings,
    payload=records,
    batch_size=100
)

print("Embeddings created and uploaded to Qdrant successfully.")
