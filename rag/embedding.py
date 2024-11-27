from dotenv import load_dotenv
import openai
import pandas as pd


EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""

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

load_dotenv()
combined_csv = pd.read_csv('../../data/combined_notices.csv')

combined_csv["embedding"] = combined_csv['context'].apply(get_embedding)

combined_csv.head()
combined_csv.to_csv('../../data/combined_notices_embedding.csv', index=False)