import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from langchain_openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load scraped data
with open("scraped_data.json", "r") as file:
    data = json.load(file)

# Step 1: Generate embeddings dynamically for unknown document structure
model = SentenceTransformer('all-MiniLM-L6-v2')

def json_to_string(json_obj):
    """Convert JSON object to a string representation."""
    return json.dumps(json_obj, indent=2)

# Convert each JSON object into a string for embedding
documents = [json_to_string(item) for item in data]

# Generate embeddings for the stringified documents
embeddings = model.encode(documents, convert_to_tensor=False)

# Step 2: Store embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 3: Querying function
def query_agent(user_query):
    query_embedding = model.encode([user_query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), k=5)
    
    # Retrieve the most relevant data
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n".join(retrieved_docs)
    
    # Use OpenAI to generate the answer
    llm = OpenAI(temperature=0)  # Ensure OPENAI_API_KEY is set in your environment
    prompt = f"You are an expert. Answer the following question using this data:\n\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    # Use the generate method
    response = llm.generate([prompt])  # Pass a list of prompts to generate()
    
    # Extract the response text
    return response.generations[0][0].text


# Step 4: Interactive terminal input
if __name__ == "__main__":
    print("Welcome to the AI Query Agent!")
    print("Type your questions below. Type 'exit' to quit.")
    
    while True:
        # Get user input
        user_query = input("Enter your query: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        
        # Process query and get response
        try:
            response = query_agent(user_query)
            print("\nAI Response:")
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")
