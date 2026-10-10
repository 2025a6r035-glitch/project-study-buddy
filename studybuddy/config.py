"""Central configuration, read from environment variables / the .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBEDDINGS = os.getenv("EMBEDDINGS", "auto").lower()  # "auto" (Gemini if possible) or "local"

# A topic whose recent average quiz score is below this counts as "weak".
MASTERY_THRESHOLD = 0.6
