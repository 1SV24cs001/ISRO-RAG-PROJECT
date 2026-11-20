
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
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #60a5fa;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
}

.stChatMessage {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# HEADER
# ---------------------------------

st.markdown(
    '<div class="main-title">🚀 ISRO RAG ASSISTANT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore ISRO missions, satellites, technology and achievements'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------
# API CONFIGURATION
# ---------------------------------

api_key = st.secrets.get(
    "OPENAI_API_KEY",
    os.getenv("OPENAI_API_KEY", "")
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error(
        "OpenAI API key is missing. "
        "Configure OPENAI_API_KEY in Streamlit secrets."
    )
    st.stop()


# ---------------------------------
# LOAD RAG ENGINE
# ---------------------------------

@st.cache_resource
def load_rag_engine():

    engine = RAGEngine(data_dir="data")

    engine.build_index()

    return engine


try:
    with st.spinner("Loading ISRO knowledge base..."):
        rag = load_rag_engine()

except Exception as e:
    st.error(f"Unable to load documents: {e}")
    st.stop()


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.title("🚀 ISRO Assistant")

    st.write(
        "Ask questions about the Indian "
        "Space Research Organisation."
    )

    st.divider()

    st.subheader("Topics")

    st.markdown("""
    - ISRO History
    - Space Missions
    - Satellites
    - Launch Vehicles
    - Space Technology
    - Future Plans
    - Global Contributions
    """)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by RAG + FAISS + Streamlit")


# ---------------------------------
# CHAT HISTORY
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write("•", source)


# ---------------------------------
# CHAT INPUT
# ---------------------------------

question = st.chat_input(
    "Ask anything about ISRO..."
)


# ---------------------------------
# GENERATE ANSWER
# ---------------------------------

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching ISRO documents..."):

            try:

                result = rag.ask(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                with st.expander("📚 Source Documents"):
                    for source in sources:
                        st.write("📄", source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:

                st.error(f"Error generating response: {e}")
