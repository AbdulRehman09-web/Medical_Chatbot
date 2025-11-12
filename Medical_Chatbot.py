import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import pyperclip

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

# -------------------- Helper Functions ------------------------
def copy_to_clipboard(text):
    pyperclip.copy(text)

def render_action_buttons(index, text, is_user=False):
    """Render neatly spaced action icons with hover tooltips (closer together)."""
    # ✅ Adjusted spacing — icons will appear much closer horizontally
    col1, col2, col3, col4, _ = st.columns([0.6, 0.6, 0.6, 0.6, 5])

    with col1:
        if st.button("📋", key=f"copy_{index}", help="Copy to clipboard"):
            copy_to_clipboard(text)
            st.toast("Copied to clipboard!")

    if not is_user:
        with col2:
            if st.button("🔄", key=f"regen_{index}", help="Regenerate response"):
                st.session_state["regenerate"] = True
                st.session_state["regen_index"] = index
                st.rerun()

        with col3:
            if st.button("👍", key=f"good_{index}", help="Good response"):
                st.toast("✅ Thanks for your feedback!")

        with col4:
            if st.button("👎", key=f"bad_{index}", help="Bad response"):
                st.toast("🙏 Feedback recorded — we'll improve.")
    else:
        col2.markdown("")
        col3.markdown("")
        col4.markdown("")

# -------------------- Streamlit App ------------------------
def app():
    st.set_page_config(page_title="💊 AI Medical Chatbot", layout="centered")
    st.title("💊 AI Medical Chatbot")
    st.caption("Ask any medical question based on your uploaded medical knowledge base.")

    # -------------------- Sidebar ------------------------
    st.sidebar.header("⚙️ Settings")

    # --- Chat History Section ---
    st.sidebar.subheader("🕘 Chat History")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.session_state.chat_history:
        for i, chat_data in enumerate(st.session_state.chat_history[::-1], 1):
            chat_name = chat_data.get("name", f"Chat {len(st.session_state.chat_history) - i + 1}")
            chat = chat_data.get("messages", [])
            with st.sidebar.expander(f"💬 {chat_name}"):
                for msg in chat:
                    role = "🧑‍⚕️" if msg["role"] == "assistant" else "🙋‍♂️"
                    st.markdown(f"**{role} {msg['role'].capitalize()}:** {msg['content'][:100]}...")

    # ✅ Added your snippet here (for clickable previous chats)
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        st.sidebar.markdown("### 🕘 Previous Chats")
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

    if st.sidebar.button("➕ New Chat"):
        if st.session_state.get("messages"):
            first_msg = st.session_state.messages[0]["content"] if st.session_state.messages else "Untitled"
            chat_name = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
            st.session_state.chat_history.append({
                "name": chat_name if chat_name.strip() else f"Chat {len(st.session_state.chat_history) + 1}",
                "messages": st.session_state.messages.copy()
            })
        st.session_state.messages = []
        st.rerun()

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

                # ✅ Only show documents if the model gave a confident answer
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
