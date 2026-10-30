
import json
from pathlib import Path

STORAGE_DIR = Path(__file__).parent / "storage"
DATA_FILE = STORAGE_DIR / "notes.json"


def load_notes():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        return []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_notes(notes):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(notes, file, ensure_ascii=False, indent=2)


def add_note(title, topic, content, embedding):
    notes = load_notes()

    notes.append({
        "title": title,
        "topic": topic,
        "content": content,
        "embedding": embedding
    })

    save_notes(notes)


def search_notes(query_embedding, top_k=3):
    import numpy as np

    notes = load_notes()

    if not notes:
        return []

    query_vector = np.asarray(query_embedding, dtype=float)
    document_vectors = np.asarray(
        [note["embedding"] for note in notes],
        dtype=float
    )

    denominator = (
        np.linalg.norm(document_vectors, axis=1)
        * np.linalg.norm(query_vector)
    )

    scores = (
        document_vectors @ query_vector
    ) / (denominator + 1e-12)

    ranked = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked:
        note = notes[int(index)].copy()
        note["similarity"] = float(scores[index])
        results.append(note)

    return results
