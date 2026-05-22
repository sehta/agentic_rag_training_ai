import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

# ==============================
# LANGCHAIN / LANGGRAPH IMPORTS
# ==============================
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import tool, create_retriever_tool

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Agentic RAG AI",
    page_icon="🤖",
    layout="wide"
)

# ==============================
# ENV VARIABLES
# ==============================
load_dotenv()

AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_CHAT_DEPLOYMENT = st.secrets["AZURE_OPENAI_CHAT_DEPLOYMENT"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

# ==============================
# SESSION STATE
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "users" not in st.session_state:
    st.session_state.users = {
        "admin": "admin123"
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "graph" not in st.session_state:
    st.session_state.graph = None


# ==============================
# LOGIN FUNCTIONS
# ==============================
def login(username, password):

    users = st.session_state.users

    if username in users and users[username] == password:
        st.session_state.logged_in = True
        return True

    return False


def signup(username, password):

    users = st.session_state.users

    if username in users:
        return False

    users[username] = password
    return True


def logout():
    st.session_state.logged_in = False
    st.session_state.chat_history = []
    st.rerun()


# ==============================
# AZURE OPENAI MODELS
# ==============================
llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
    temperature=0
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
)


# ==============================
# DOCUMENT PROCESSING
# ==============================
def process_documents(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

            tmp_file.write(uploaded_file.getbuffer())

            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="search_pdf_knowledge_base",
        description="Search uploaded PDF documents"
    )

    tools = [retriever_tool]

    # ==============================
    # AGENT STATE
    # ==============================
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # ==============================
    # GENERATE NODE
    # ==============================
    def generate(state: AgentState):

        messages = state["messages"]

        question = messages[0].content

        docs = messages[-1].content

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an Agentic RAG AI Assistant.

Answer ONLY from retrieved documents.

If answer is not available in documents,
say:
'I could not find the answer in uploaded documents.'
"""
            ),
            (
                "human",
                """
Question:
{question}

Context:
{context}
"""
            )
        ])

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke({
            "question": question,
            "context": docs
        })

        return {
            "messages": [HumanMessage(content=response)]
        }

    # ==============================
    # AGENT NODE
    # ==============================
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState):

        messages = state["messages"]

        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    # ==============================
    # GRAPH
    # ==============================
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)

    workflow.add_node("retrieve", ToolNode(tools))

    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END
        }
    )

    workflow.add_edge("retrieve", "generate")

    workflow.add_edge("generate", END)

    graph = workflow.compile()

    st.session_state.graph = graph

    return "Documents processed successfully!"


# ==============================
# ASK AGENT
# ==============================
def ask_agent(question):

    graph = st.session_state.graph

    inputs = {
        "messages": [
            HumanMessage(content=question)
        ]
    }

    result = graph.invoke(inputs)

    final_answer = result["messages"][-1].content

    return final_answer


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:

    st.title("🤖 Agentic RAG")

    if st.session_state.logged_in:

        st.success("Logged In")

        st.markdown("---")

        st.subheader("📜 Chat History")

        if len(st.session_state.chat_history) == 0:
            st.info("No chat history")

        for chat in st.session_state.chat_history:

            with st.expander(chat["question"][:40]):

                st.write("Question:")
                st.write(chat["question"])

                st.write("Answer:")
                st.write(chat["answer"])

        st.markdown("---")

        if st.button("Logout"):
            logout()


# ==============================
# LOGIN SCREEN
# ==============================
if not st.session_state.logged_in:

    st.title("🔐 Agentic RAG Login")

    tab1, tab2 = st.tabs(["Login", "Create Account"])

    with tab1:

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if login(username, password):
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Credentials")

    with tab2:

        new_user = st.text_input("New Username")

        new_pass = st.text_input(
            "New Password",
            type="password"
        )

        if st.button("Create Account"):

            if signup(new_user, new_pass):
                st.success("Account Created")

            else:
                st.error("User already exists")


# ==============================
# MAIN APP
# ==============================
else:

    st.title("🤖 Agentic RAG AI")

    st.subheader("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):

        if uploaded_files:

            with st.spinner("Creating embeddings and graph..."):

                message = process_documents(uploaded_files)

            st.success(message)

        else:
            st.warning("Please upload at least one PDF")

    st.markdown("---")

    st.subheader("💬 Chat")

    question = st.chat_input("Ask your documents...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):

            answer = ask_agent(question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

    for item in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])
