
import os
import json
import numpy as np
import faiss

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq


class RAGEngine:

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.chunks = []
        self.sources = []
        self.index = None

    def load_documents(self):

        documents = []

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Folder not found: {self.data_dir}"
            )

        for filename in os.listdir(self.data_dir):

            path = os.path.join(
                self.data_dir, filename
            )

            text = ""

            if filename.lower().endswith(".pdf"):
                reader = PdfReader(path)

                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"

            elif filename.lower().endswith(".docx"):
                doc = Document(path)

                text = "\n".join(
                    p.text for p in doc.paragraphs
                )

            if text.strip():
                documents.append((filename, text))

        return documents

    def split_text(self, text, chunk_size=700,
                   overlap=100):

        words = text.split()
        chunks = []

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(words[start:end])

            if chunk.strip():
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def build_index(self):

        documents = self.load_documents()

        if not documents:
            raise ValueError(
                "No readable PDF or DOCX documents found."
            )

        self.chunks = []
        self.sources = []

        for filename, text in documents:

            text_chunks = self.split_text(text)

            for chunk in text_chunks:
                self.chunks.append(chunk)
                self.sources.append(filename)

        embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = embeddings.astype("float32")

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

    def retrieve(self, question, top_k=4):

        if self.index is None:
            self.build_index()

        query_embedding = self.embedder.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            query_embedding, k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append({
                "text": self.chunks[idx],
                "source": self.sources[idx],
                "score": float(score)
            })

        return results

    def ask(self, question):

    results = self.retrieve(question)

    context = "\n\n".join(
        f"Source: {r['source']}\n{r['text']}"
        for r in results
    )

    prompt = f"""
You are an ISRO information assistant.

Answer the question using ONLY the
provided document context.

If the answer is unavailable, say so.
Do not invent facts.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and easy-to-understand answer.
"""

    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an accurate ISRO RAG assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(
            r["source"] for r in results
        ))
    }
