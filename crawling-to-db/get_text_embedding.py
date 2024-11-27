import openai

def get_text_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""

    EMBEDDING_MODEL = "text-embedding-3-small"

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None