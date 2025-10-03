"""
RAG-based Chatbot with Vector Database
Stores conversations and retrieves relevant context for LLM prompting
"""
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import datetime
import uuid

class RAGChatbot:
    def __init__(self, db_path="./chroma_db", collection_name="conversations"):
        """Initialize the RAG chatbot with ChromaDB and embedding model"""
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Load embedding model (lightweight and effective)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Initialized RAG Chatbot. Current vectors: {self.collection.count()}")
    
    def store_conversation(self, user_message, bot_response, metadata=None):
        """Store a conversation turn in the vector database"""
        
        # Create a unique ID
        conv_id = str(uuid.uuid4())
        
        # Combine user message and bot response for context
        full_context = f"User: {user_message}\nAssistant: {bot_response}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(full_context).tolist()
        
        # Prepare metadata
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
        }
        if metadata:
            meta.update(metadata)
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[full_context],
            metadatas=[meta],
            ids=[conv_id]
        )
        
        return conv_id
    
    def retrieve_relevant_context(self, query, top_k=3):
        """Retrieve relevant conversation history for the current query"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search for similar conversations
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format retrieved contexts
        contexts = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if 'distances' in results else None
                contexts.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': distance
                })
        
        return contexts
    
    def build_prompt_with_context(self, user_query, system_prompt="You are a helpful assistant.", top_k=3):
        """Build a complete prompt with retrieved context for the LLM"""
        
        # Retrieve relevant contexts
        contexts = self.retrieve_relevant_context(user_query, top_k)
        
        # Build context section
        context_section = ""
        if contexts:
            context_section = "\n\nRelevant conversation history:\n"
            for i, ctx in enumerate(contexts, 1):
                context_section += f"\n{i}. {ctx['content']}\n"
        
        # Build complete prompt
        prompt = f"""{system_prompt}
{context_section}

Current user query: {user_query}

Please provide a helpful response based on the conversation history and the current query."""
        
        return prompt, contexts