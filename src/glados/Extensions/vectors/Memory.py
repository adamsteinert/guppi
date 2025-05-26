import ollama
import chromadb


class Memory:
    EMBEDDING_MODEL = "nomic-embed-text"
    COLLECTION_NAME = "memory"

    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)

    def store_command_memories(self, dir):
        # Example documents with metadata
        documents = [
            {
                "text": "Guppy, shut down",
                "metadata": {
                    "title": "shutdown",
                    "category": "command"
                }
            },
            {
                "text": "Guppy, clear your memory",
                "metadata": {
                    "title": "clear_memory",
                    "category": "command",
                }
            },
            {
                "text": "What do I need for salinity with a current value of thirteen",
                "metadata": {
                    "title": "get_salinity",
                    "category": "tool_call",
                }
            },
            {
                "text": "My current salinity value is thirteen, how much salt do I need to add?",
                "metadata": {
                    "title": "get_salinity",
                    "category": "tool_call",
                }
            },
            {
                "text": "My current salinity is twenty one, what do I need to add",
                "metadata": {
                    "title": "get_salinity",
                    "category": "tool_call",
                }
            }
        ]

        # Generate embeddings and store with metadata
        for i, doc in enumerate(documents):
            # Generate embedding with Ollama
            response = ollama.embed(model=self.EMBEDDING_MODEL, input=doc["text"])
            embedding = response["embedding"]

            # Store in ChromaDB with metadata
            self.collection.upsert(
                ids=[f"doc_{i}"],
                embeddings=[embedding],
                documents=[doc["text"]],
                metadatas=[doc["metadata"]]  # Pass the metadata dictionary
            )

    def query_memory(self, query):
        query_response = ollama.embed(model=self.EMBEDDING_MODEL, input=query)
        query_embedding = query_response["embedding"]

        # 2. Retrieve similar documents
        results = self.collection.query(
            query_texts=query,
            query_embeddings=[query_embedding],
            n_results=3
        )

        # 3. Use Claude to process the query with retrieved documents
        relevant_docs = results["documents"][0]
        context = "\n\n".join(relevant_docs)



    def test_simple(self):
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(name="my_collection")
        collection.add(
            documents=[
                "This is a document about pineapple",
                "This is a document about oranges"
            ],
            ids=["id1", "id2"]
        )
        results = collection.query(
            query_texts=["This is a query document about hawaii"],  # Chroma will embed this for you
            n_results=2  # how many results to return
        )
        print(results)



if __name__ == "__main__":
    m = Memory()
    m.test_simple()