"""One-time script: wipe all indexed documents from ChromaDB."""
import os
import sys

import chromadb


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(base_dir, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection("medlearn_docs")
        count = collection.count()
        print(f"medlearn_docs: {count} documents found. Wiping...")
        # ChromaDB does not support $exists; fetch all IDs then delete by ID.
        result = collection.get(include=[])
        ids = result.get("ids") or []
        if ids:
            collection.delete(ids=ids)
            print(f"Deleted {len(ids)} documents.")
        else:
            print("Collection already empty.")
    except Exception as e:
        print(f"Nothing to wipe ({e})")
    print("Done.")


if __name__ == "__main__":
    main()
