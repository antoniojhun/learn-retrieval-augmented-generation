# Introduction to Retrieval Augmented Generation

This repository will introduce you to Retrieval Augmented Generation (RAG) with
easy-to-use examples that you can build upon. The examples use Python with
Jupyter Notebooks and CSV files. The vector database uses the Qdrant database
which can run in-memory.

## Setup your environment

This example can run in Codespaces but you can use the following if you are
cloning this repository:

**Install the dependencies**

Create the virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

**Upgrade `numpy` and `pandas`**

To avoid potential binary incompatibility issues, upgrade `numpy` and `pandas`:

```bash
python3 -m pip install --upgrade numpy pandas
```

Here is a summary of what this repository will use:

1. [Qdrant](https://github.com/qdrant/qdrant) for the vector database. We will use an in-memory database for the examples.
2. [Llamafile](https://github.com/Mozilla-Ocho/llamafile) for the LLM (alternatively you can use an OpenAI API compatible key and endpoint).
3. [OpenAI's Python API](https://pypi.org/project/openai/) to connect to the LLM after retrieving the vectors response from Qdrant.
4. Sentence Transformers to create the embeddings with minimal effort.

## Setup and Run Phi-3 Model with Ollama

To run the Phi-3 model with Ollama and make it accessible over the network, follow these steps:

### Step 1: Set Environment Variables

Open Terminal and set the `OLLAMA_HOST` environment variable to listen on all interfaces:

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0"
```

Restart the Ollama application if it's already running.

### Step 2: Start the Ollama Service

Run the Ollama service:

```bash
ollama run phi3
```

Note the port number from the terminal output. Let's assume it's `11434`.

### Step 3: Verify the Server is Running

Use `curl` to verify if the server is running:

```bash
curl http://localhost:11434/api/generate -d '
{
  "model": "phi3",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "num_thread": 8,
    "num_ctx": 2024
  }
}' | jq .
```

## Step 4: Run the Scripts

First, you will import and prepare your data. Then, you will embed the data and finally, you will run the main script to search and generate a response.

### Step 4.1: Import Your Data

#### `import_data.py`

```python
import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('top_rated_wines.csv')

# Step 2: Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Step 3: Display the descriptive statistics of the DataFrame
print("\nDescriptive statistics of the DataFrame:")
print(df.describe())

# Step 4: Convert the DataFrame to a list of dictionaries
records = df.to_dict('records')

# Step 5: Display the first few records to verify the conversion
print("\nFirst few records as dictionaries:")
print(records[:5])

# Save the list of dictionaries to a file for further processing if needed
import json
with open('top_rated_wines_records.json', 'w') as f:
    json.dump(records, f, indent=4)
```

### Step 4.2: Create Embeddings

#### `embed_data.py`

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http import models

# Load the data
df = pd.read_csv('/top_rated_wines.csv')
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
```

### Step 4.3: Run the Main Script

The combined script will create embeddings, upload them to Qdrant, and perform a search and generate a completion using the local LLM.

#### `main.py`

```python
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
```

### How to Run the Scripts

1. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run the `import_data.py` script** to import your data:
   ```bash
   python import_data.py
   ```

3. **Run the `embed_data.py` script** to create embeddings:
   ```bash
   python embed_data.py
   ```

4. **Run the `main.py` script** to search and generate responses:
   ```bash
   python main.py
   ```
