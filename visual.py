import chromadb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_chromadb_collection(collection_name="conversations"):
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="testing\chroma_db")
    
    # Get the collection
    collection = client.get_collection(collection_name)
    
    # Get all items from collection
    results = collection.get()
    
    if not results['embeddings']:
        print("No embeddings found in collection")
        return
    
    # Convert embeddings to numpy array
    embeddings = np.array(results['embeddings'])
    
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create the visualization
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=range(len(embeddings_2d)), 
                         cmap='viridis')
    
    # Add labels for each point
    for i, text in enumerate(results['metadatas']):
        if text and 'timestamp' in text:
            label = text['timestamp'][:10]  # Show only date part
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.colorbar(scatter, label='Conversation Order')
    plt.title('ChromaDB Embeddings Visualization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.show()

if __name__ == "__main__":
    visualize_chromadb_collection()