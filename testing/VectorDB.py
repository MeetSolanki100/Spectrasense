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
    
    def chat(self, user_message, llm_generate_func, system_prompt="You are a helpful assistant.", store=True):
        """
        Complete chat workflow: retrieve context, generate response, store conversation
        
        Args:
            user_message: The user's input
            llm_generate_func: Your local LLM's generation function (takes prompt string, returns response)
            system_prompt: System instruction for the LLM
            store: Whether to store this conversation turn
        """
        
        # Build prompt with retrieved context
        prompt, contexts = self.build_prompt_with_context(user_message, system_prompt)
        
        # Generate response using your local LLM
        bot_response = llm_generate_func(prompt)
        
        # Store the conversation
        if store:
            conv_id = self.store_conversation(user_message, bot_response)
            print(f"Stored conversation: {conv_id}")
        
        return {
            'response': bot_response,
            'retrieved_contexts': contexts,
            'prompt_used': prompt
        }


# Example usage with different LLM backends
def example_with_ollama():
    """Example using Ollama for local LLM"""
    import requests
    
    def ollama_generate(prompt):
        """Generate response using Ollama"""
        url = 'http://localhost:11434/api/generate'
        payload = {
            "model": 'mistral:7b',
            "prompt": prompt,
            
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            output = []
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        output.append(data["response"])
            return ''.join(output).strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Chat with context retrieval
    result = chatbot.chat(
        "What did we discuss about Python?",
        llm_generate_func=ollama_generate
    )
    
    print("Response:", result['response'])


def example_with_transformers():
    """Example using HuggingFace Transformers"""
    from transformers import pipeline
    
    # Load model (use a smaller model for faster inference)
    generator = pipeline('text-generation', model='gpt2')
    
    def hf_generate(prompt):
        """Generate response using HuggingFace"""
        result = generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text'][len(prompt):]
    
    # Initialize and use chatbot
    chatbot = RAGChatbot()
    result = chatbot.chat(
        "Tell me about machine learning",
        llm_generate_func=hf_generate
    )
    
    print("Response:", result['response'])


def example_manual_workflow():
    """Example showing manual control over each step"""
    
    chatbot = RAGChatbot()
    
    # 1. Store some example conversations first
    chatbot.store_conversation(
        "What is Python?",
        "Python is a high-level programming language known for its simplicity."
    )
    
    chatbot.store_conversation(
        "How do I use lists in Python?",
        "Lists in Python are created using square brackets, like: my_list = [1, 2, 3]"
    )
    
    # 2. Query with new message
    user_query = "Can you explain Python data structures?"
    
    # 3. Retrieve relevant context
    contexts = chatbot.retrieve_relevant_context(user_query, top_k=2)
    print("\nRetrieved contexts:")
    for ctx in contexts:
        print(f"- {ctx['content'][:100]}... (distance: {ctx['distance']:.3f})")
    
    # 4. Build prompt
    prompt, _ = chatbot.build_prompt_with_context(user_query)
    print("\n\nPrompt to send to LLM:")
    print(prompt)
    
    # 5. Send to your LLM and get response
    # bot_response = example_with_ollama(prompt)
    
    # 6. Store the new conversation
    # chatbot.store_conversation(user_query, bot_response)


if __name__ == "__main__":
    # Run manual workflow example
    example_manual_workflow()
    example_with_ollama()
    # Uncomment to try with your LLM:
    # example_with_ollama()
    # example_with_transformers()