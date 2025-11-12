from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def load_LLM():
    llm = ChatOpenAI(
        model="x-ai/grok-4-fast",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,
        max_tokens=512,
    )
    return llm


CUSTOM_PROMPT_TEMPLATE = """
You are a helpful and accurate medical assistant.
Use only the context provided to answer the user's question.
If the answer is not in the context, respond with "I'm not sure based on the given information."

Context:
{context}

Question:
{question}

Answer directly and concisely:
"""

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt


DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


llm = load_LLM()

# Test a simple query
# print("Model Test Output:")
# response = llm.invoke("How to cure cancer?")
# print(response.content)
# print("--------------------------------------------------")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

while True:
    user_query = input("\nAsk your medical question (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("Goodbye 👋")
        break

    response = qa_chain.invoke({"query": user_query})
    print("\n RESULT:\n", response["result"])
    print("\n SOURCE DOCUMENTS:\n", [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]])
