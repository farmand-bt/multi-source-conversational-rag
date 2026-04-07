"""Reset the ChromaDB vector store by deleting the persistence directory."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CHROMA_PERSIST_DIR


def reset():
    path = Path(CHROMA_PERSIST_DIR)
    if path.exists():
        shutil.rmtree(path)
        print(f"Deleted {path}")
    else:
        print(f"Nothing to delete — {path} does not exist")


if __name__ == "__main__":
    reset()
