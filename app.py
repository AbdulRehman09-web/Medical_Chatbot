import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS

# Clipboard fallbacks
import json
import streamlit.components.v1 as components
try:
    import pyperclip
    _HAS_PYPERCLIP = True
except Exception:
    _HAS_PYPERCLIP = False

# -------------------- Load Environment Variables --------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# -------------------- FAISS Vectorstore Path ------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

# -------------------- Load Vectorstore ------------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# -------------------- Custom Prompt ------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful and accurate medical assistant.
Use only the context provided to answer the user's question.
If the answer is not in the context, respond with:
"I'm not sure based on the given information."

Context:
{context}

Question:
{question}

Answer directly and concisely:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# -------------------- Load LLM ------------------------
def load_LLM(model_choice):
    llm = ChatOpenAI(
        model=model_choice,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,
        max_tokens=512,
    )
    return llm

# -------------------- Helper Functions (robust clipboard) ------------------------
def copy_to_clipboard(text):
    """
    Try server-side copy (pyperclip) first (useful for local desktop).
    If that fails (headless server), render a tiny browser-side HTML/JS widget
    that performs the copy in the user's browser.
    Returns a dict describing the method used.
    """
    # Try pyperclip first (works on local desktops)
    if _HAS_PYPERCLIP:
        try:
            pyperclip.copy(text)
            # small confirmation to user
            try:
                st.toast("Copied to clipboard (server-side)!")
            except Exception:
                st.success("Copied to clipboard (server-side)!")
            return {"method": "pyperclip"}
        except Exception:
            # fall through to browser copy
            pass

    # Browser-side fallback using JS - will copy text in user's browser clipboard
    safe_text = json.dumps(text)  # escape text for JS
    html = f"""
    <div id="copy-container" style="display:flex; gap:8px; align-items:center;">
      <button id="copy-btn" style="
        padding:6px 10px;
        border-radius:8px;
        border:1px solid #ddd;
        cursor:pointer;
        background:#f7f7f7;
      ">
        Click to copy to clipboard (browser)
      </button>
      <span id="copy-msg" style="font-size:0.9em;color:#666;"></span>
    </div>
    <script>
      const btn = document.getElementById("copy-btn");
      const msg = document.getElementById("copy-msg");
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText({safe_text});
          msg.innerText = " — Copied!";
        }} catch (err) {{
          msg.innerText = " — Copy failed (please select and Ctrl/Cmd+C)";
        }}
      }});
    </script>
    """
    # Render the small HTML widget
    components.html(html, height=64)
    try:
        st.toast("Opened browser copy helper. Click the button shown to copy.")
    except Exception:
        st.info("Opened browser copy helper. Click the button shown to copy.")
    return {"method": "browser"}

def render_action_buttons(index, text, is_user=False):
    """Render neatly spaced action icons with hover tooltips (closer together)."""
    col1, col2, col3, col4, _ = st.columns([0.6, 0.6, 0.6, 0.6, 5])

    with col1:
        if st.button("📋", key=f"copy_{index}", help="Copy to clipboard"):
            res = copy_to_clipboard(text)
            # If pyperclip succeeded we already showed a toast; for browser fallback, JS widget is shown.
            if res.get("method") == "browser":
                # optional extra message (non-blocking)
                pass

    if not is_user:
        with col2:
            if st.button("🔄", key=f"regen_{index}", help="Regenerate response"):
                st.session_state["regenerate"] = True
                st.session_state["regen_index"] = index
                st.rerun()

        with col3:
            if st.button("👍", key=f"good_{index}", help="Good response"):
                try:
                    st.toast("✅ Thanks for your feedback!")
                except Exception:
                    st.success("✅ Thanks for your feedback!")

        with col4:
            if st.button("👎", key=f"bad_{index}", help="Bad response"):
                try:
                    st.toast("🙏 Feedback recorded — we'll improve.")
                except Exception:
                    st.info("🙏 Feedback recorded — we'll improve.")
    else:
        col2.markdown("")
        col3.markdown("")
        col4.markdown("")

# -------------------- Streamlit App ------------------------
def app():

    st.set_page_config(page_title="AI Meditbot", page_icon="💊", layout="centered")
    st.title("💊 AI Medical Chatbot")
    st.caption("Ask any medical question based on your uploaded medical knowledge base.")

    # -------------------- Sidebar ------------------------
    st.sidebar.header("⚙️ Settings")
    st.sidebar.subheader("🕘 Chat History")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.sidebar.button("➕ New Chat"):
        if "messages" in st.session_state and len(st.session_state.messages) > 0:
            st.session_state.history.append({
                "title": st.session_state.messages[0]["content"][:30] + "...",
                "messages": st.session_state.messages.copy()
            })
        st.session_state.messages = []
        st.rerun()

    if st.session_state.history:
        for i, chat in enumerate(st.session_state.history):
            if st.sidebar.button(chat["title"], key=f"chat_{i}"):
                st.session_state.messages = chat["messages"]
                st.rerun()

    model_choice = st.sidebar.selectbox(
        "Choose LLM Model",
        [
            "x-ai/grok-4-fast",
            "gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "mistralai/mistral-7b-instruct"
        ],
        index=0
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "regenerate" not in st.session_state:
        st.session_state.regenerate = False

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_action_buttons(i, message["content"], is_user=(message["role"] == "user"))

    user_query = st.chat_input("Enter your medical query...")

    if user_query or st.session_state.get("regenerate", False):
        if st.session_state.get("regenerate", False):
            last_query = st.session_state.messages[st.session_state["regen_index"] - 1]["content"]
            user_query = last_query
            st.session_state["regenerate"] = False

        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        render_action_buttons(len(st.session_state.messages) - 1, user_query, is_user=True)

        with st.spinner("🩺 Analyzing your question..."):
            try:
                vectorstore = get_vectorstore()
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                llm = load_LLM(model_choice)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
                )

                response = qa_chain.invoke({"query": user_query})
                result = response["result"].strip()
                source_docs = response["source_documents"]

                # ✅ Only show docs if model is confident
                if "I'm not sure based on the given information" in result:
                    response_text = "**Answer:** I'm not sure based on the given information."
                else:
                    response_text = f"**Answer:** {result}\n\n**Sources:**\n"
                    for i, doc in enumerate(source_docs, 1):
                        source = doc.metadata.get("source", "Unknown Source")
                        preview = doc.page_content[:200].strip().replace("\n", " ")
                        response_text += f"\n📄 **Document {i}:** {source}\n🩺 {preview}...\n"

                st.chat_message("assistant").markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                render_action_buttons(len(st.session_state.messages) - 1, response_text, is_user=False)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# -------------------- Run ------------------------
if __name__ == "__main__":
    app()
