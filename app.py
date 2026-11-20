
import os
import streamlit as st

from rag_engine import RAGEngine


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ISRO AI Assistant",
    page_icon="🚀",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #60a5fa;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 25px;
}

[data-testid="stChatMessage"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🚀 ISRO AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore ISRO missions, satellites, technology and achievements'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# API CONFIGURATION
# ==========================================

api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY", "")
)

if not api_key:
    st.error("Groq API key is missing.")
    st.info(
        "Add GROQ_API_KEY in Streamlit Cloud "
        "Settings > Secrets."
    )
    st.stop()

os.environ["GROQ_API_KEY"] = api_key


# ==========================================
# LOAD RAG ENGINE
# ==========================================

@st.cache_resource
def load_rag_engine():

    engine = RAGEngine(data_dir="data")

    engine.build_index()

    return engine


try:

    with st.spinner("🚀 Loading ISRO Knowledge Base..."):

        rag = load_rag_engine()

except Exception as e:

    st.error("Unable to load ISRO documents.")

    st.exception(e)

    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🚀 ISRO Assistant")

    st.write(
        "Your AI-powered assistant for "
        "Indian Space Research Organisation."
    )

    st.divider()

    st.subheader("Explore Topics")

    st.markdown("""
    - 🇮🇳 ISRO History
    - 🌙 Chandrayaan Missions
    - 🔴 Mangalyaan
    - 🛰️ Satellites
    - 🚀 Launch Vehicles
    - 🌌 Space Technology
    - 🔭 Future Missions
    """)

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.caption("Powered by Groq + FAISS + RAG")


# ==========================================
# CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:

                    st.write("📄", source)


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask anything about ISRO..."
)


# ==========================================
# GENERATE ANSWER
# ==========================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🔍 Searching ISRO knowledge base..."):

            try:

                result = rag.ask(question)

                answer = result["answer"]

                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("📚 Source Documents"):

                        for source in sources:

                            st.write("📄", source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:

                st.error("Unable to generate an answer.")

                st.exception(e)
