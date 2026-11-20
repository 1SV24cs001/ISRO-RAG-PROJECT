import os
import streamlit as st

from rag_engine import RAGEngine


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="ISRO RAG Assistant",
    page_icon="🚀",
    layout="wide"
)


# ---------------------------------
# CUSTOM DESIGN
# ---------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: white;
}

.main-title {
    text-align: center;
    color: #00c8ff;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    font-size: 17px;
}

.stChatMessage {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# TITLE
# ---------------------------------

st.markdown(
    '<div class="main-title">🚀 ISRO RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about ISRO missions, satellites, '
    'technology and achievements.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------
# GET GROQ API KEY
# ---------------------------------

def get_api_key():

    try:
        key = st.secrets.get("GROQ_API_KEY")

        if key:
            return key

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


# ---------------------------------
# INITIALIZE RAG ENGINE
# ---------------------------------

@st.cache_resource(show_spinner="Loading ISRO documents and AI model...")
def load_rag_engine(api_key):

    engine = RAGEngine(
        data_dir="data",
        api_key=api_key
    )

    engine.build_index()

    return engine


api_key = get_api_key()

if not api_key:

    st.error(
        "Groq API key is missing. "
        "Please add GROQ_API_KEY in Streamlit Secrets."
    )

    st.stop()


# ---------------------------------
# LOAD ENGINE
# ---------------------------------

try:

    rag = load_rag_engine(api_key)

except Exception as error:

    st.error(
        f"Unable to initialize the RAG engine: {error}"
    )

    st.info(
        "Check your data folder, PDF files and requirements.txt."
    )

    st.stop()


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.title("🚀 ISRO Assistant")

    st.write(
        "An AI-powered Retrieval Augmented Generation "
        "application for ISRO documents."
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write(f"Documents indexed: {len(set(rag.sources))}")

    st.write(f"Text chunks: {len(rag.chunks)}")

    st.write("AI Model: GPT-OSS 120B")

    st.write("Vector Database: FAISS")

    st.write("Embedding: all-MiniLM-L6-v2")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------
# CHAT HISTORY
# ---------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View document sources"):

                for source in message["sources"]:
                    st.write(f"📄 {source}")


# ---------------------------------
# CHAT INPUT
# ---------------------------------

question = st.chat_input(
    "Ask anything about ISRO..."
)


if question:

    # Display user question

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner("Searching ISRO documents..."):

            try:

                result = rag.ask(question)

                answer = result["answer"]

                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("View document sources"):

                        for source in sources:
                            st.write(f"📄 {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(
                    f"Error generating response: {error}"
                )
