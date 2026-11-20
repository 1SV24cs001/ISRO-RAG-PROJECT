
import os
from pathlib import Path

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer


class RAGEngine:

    def __init__(self, data_dir="data"):

        self.data_dir = Path(data_dir)

        self.documents = []
        self.vector_store = None

        # Groq API configuration
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Please configure your API key."
            )

        self.client = Groq(api_key=api_key)

        self.model_name = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

        # Embedding model
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    # ======================================
    # LOAD DOCUMENTS
    # ======================================

    def load_documents(self):

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.data_dir}"
            )

        files = sorted(self.data_dir.rglob("*.txt"))

        if not files:
            raise FileNotFoundError(
                "No .txt files found inside the data folder."
            )

        documents = []

        for file_path in files:

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if text:
                documents.append({
                    "text": text,
                    "source": file_path.name
                })

        if not documents:
            raise ValueError(
                "The text files are empty."
            )

        return documents

    # ======================================
    # SPLIT DOCUMENTS
    # ======================================

    def split_documents(
        self,
        documents,
        chunk_size=1000,
        overlap=200
    ):

        chunks = []

        for document in documents:

            text = document["text"]

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunk_text = text[start:end].strip()

                if chunk_text:

                    chunks.append({
                        "text": chunk_text,
                        "source": document["source"]
                    })

                start += chunk_size - overlap

        return chunks

    # ======================================
    # BUILD FAISS INDEX
    # ======================================

    def build_index(self):

        print("Loading ISRO documents...")

        documents = self.load_documents()

        print("Splitting documents...")

        self.documents = self.split_documents(
            documents
        )

        if not self.documents:
            raise ValueError(
                "No document chunks were created."
            )

        texts = [
            document["text"]
            for document in self.documents
        ]

        print("Creating embeddings...")

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        self.vector_store = faiss.IndexFlatIP(
            dimension
        )

        self.vector_store.add(embeddings)

        print("FAISS index created successfully!")

    # ======================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ======================================

    def retrieve_documents(self, question, k=4):

        if self.vector_store is None:
            raise RuntimeError(
                "FAISS index is not built."
            )

        query_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        total_documents = len(self.documents)

        k = min(k, total_documents)

        scores, indices = self.vector_store.search(
            query_embedding,
            k
        )

        results = []

        for index in indices[0]:

            if index >= 0:
                results.append(
                    self.documents[index]
                )

        return results

    # ======================================
    # GENERATE ANSWER USING GROQ
    # ======================================

    def ask(self, question):

        relevant_documents = self.retrieve_documents(
            question,
            k=4
        )

        context = "\n\n".join(
            document["text"]
            for document in relevant_documents
        )

        system_prompt = """
You are an AI assistant specializing in ISRO
(Indian Space Research Organisation).

Answer questions using the provided context.

Rules:
1. Give accurate and clear answers.
2. Use the retrieved context as your primary source.
3. Do not invent facts or mission details.
4. If the answer is not available in the context,
   clearly say that the knowledge base does not
   contain enough information.
5. Explain technical concepts in simple language.
6. Use headings and bullet points when useful.
"""

        user_prompt = f"""
ISRO KNOWLEDGE BASE CONTEXT:

{context}

USER QUESTION:

{question}

Provide a clear and informative answer.
"""

        response = self.client.chat.completions.create(

            model=self.model_name,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.2,

            max_tokens=1200
        )

        answer = response.choices[0].message.content

        sources = list(dict.fromkeys(
            document["source"]
            for document in relevant_documents
        ))

        return {
            "answer": answer,
            "sources": sources
        }
