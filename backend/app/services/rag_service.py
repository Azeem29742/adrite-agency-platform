from app.rag.vector_store import create_vector_store

# Lazy load vector store
vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = create_vector_store()
    return vectorstore


def get_rag_response(query: str) -> str:
    try:
        vs = get_vectorstore()

        # 🔍 Retrieve docs
        docs = vs.similarity_search(query, k=3)

        # ✅ ADD THIS LINE
        print("DEBUG DOCS:", docs)

        if not docs:
            return "No relevant information found."

        context = "\n\n".join([doc.page_content for doc in docs])

        return context

    except Exception as e:
        return f"Error retrieving context: {str(e)}"