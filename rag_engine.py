import os
import numpy as np
import faiss

from groq import Groq
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer


class RAGEngine:

    def __init__(self, data_dir="data"):

        self.data_dir = data_dir

        # Load embedding model
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Initialize Groq API
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Please configure it in Streamlit Secrets."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.chunks = []
        self.sources = []
        self.index = None

    # ---------------------------------
    # LOAD PDF AND DOCX DOCUMENTS
    # ---------------------------------

    def load_documents(self):

        documents = []

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data folder not found: {self.data_dir}"
            )

        for filename in os.listdir(self.data_dir):

            path = os.path.join(
                self.data_dir,
                filename
            )

            text = ""

            # Read PDF
            if filename.lower().endswith(".pdf"):

                reader = PdfReader(path)

                for page in reader.pages:
                    text += (
                        page.extract_text() or ""
                    ) + "\n"

            # Read DOCX
            elif filename.lower().endswith(".docx"):

                doc = Document(path)

                text = "\n".join(
                    paragraph.text
                    for paragraph in doc.paragraphs
                )

            if text.strip():

                documents.append({
                    "filename": filename,
                    "text": text
                })

        if not documents:
            raise ValueError(
                "No readable PDF or DOCX files found in data folder."
            )

        return documents

    # ---------------------------------
    # SPLIT TEXT INTO CHUNKS
    # ---------------------------------

    def split_text(
        self,
        text,
        chunk_size=700,
        overlap=100
    ):

        words = text.split()

        chunks = []

        step = chunk_size - overlap

        if step <= 0:
            raise ValueError(
                "Chunk size must be greater than overlap."
            )

        for start in range(
            0,
            len(words),
            step
        ):

            chunk = " ".join(
                words[start:start + chunk_size]
            )

            if chunk.strip():
                chunks.append(chunk)

        return chunks

    # ---------------------------------
    # BUILD FAISS VECTOR INDEX
    # ---------------------------------

    def build_index(self):

        documents = self.load_documents()

        self.chunks = []
        self.sources = []

        for document in documents:

            text_chunks = self.split_text(
                document["text"]
            )

            for chunk in text_chunks:

                self.chunks.append(chunk)

                self.sources.append(
                    document["filename"]
                )

        if not self.chunks:
            raise ValueError(
                "No text chunks generated from documents."
            )

        # Generate embeddings
        embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = embeddings.astype(
            "float32"
        )

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        print(
            f"Successfully indexed {len(self.chunks)} chunks."
        )

    # ---------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # ---------------------------------

    def retrieve(
        self,
        question,
        top_k=4
    ):

        if self.index is None:
            self.build_index()

        # Generate question embedding
        query_embedding = self.embedder.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(
            query_embedding
        )

        k = min(
            top_k,
            len(self.chunks)
        )

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            results.append({

                "text": self.chunks[idx],

                "source": self.sources[idx],

                "score": float(score)

            })

        return results

    # ---------------------------------
    # GENERATE ANSWER USING GROQ
    # ---------------------------------

    def ask(self, question):

        if not question or not question.strip():
            raise ValueError(
                "Please enter a question."
            )

        # Retrieve relevant chunks
        results = self.retrieve(
            question,
            top_k=4
        )

        if not results:
            return {
                "answer": (
                    "No relevant information was found "
                    "in the ISRO documents."
                ),
                "sources": []
            }

        # Prepare context
        context = "\n\n".join(

            f"Source: {result['source']}\n"
            f"Content: {result['text']}"

            for result in results

        )

        # Prepare prompt
        prompt = f"""
You are an intelligent ISRO RAG Assistant.

Your task is to answer questions about the
Indian Space Research Organisation (ISRO).

IMPORTANT RULES:

1. Answer using only the provided document context.
2. Do not invent information.
3. If the answer is unavailable in the context,
   clearly mention that the uploaded documents
   do not contain sufficient information.
4. Give clear and simple explanations.
5. Use bullet points when appropriate.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

       # Call Groq API
response = self.client.chat.completions.create(

    model="openai/gpt-oss-120b",

    messages=[

        {
            "role": "system",
            "content": (
                "You are an accurate ISRO "
                "document-based question answering assistant."
            )
        },

        {
            "role": "user",
            "content": prompt
        }

    ],

    temperature=0.2,

    max_tokens=1500

)

        answer = response.choices[0].message.content

        # Return answer and sources
        return {

            "answer": answer,

            "sources": list(
                dict.fromkeys(
                    result["source"]
                    for result in results
                )
            )

        }
