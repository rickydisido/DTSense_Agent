import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal, Optional
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
st.set_page_config(page_title="Agentic RAG", layout="wide")

# Load environment variables
load_dotenv()

# --- PINECONE SETUP ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "dts-data-test"
index = pc.Index(index_name)

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/gtr-t5-base")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

search_tool = TavilySearchResults(k=3)
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b")

# --- Graph State ---
class GraphState(TypedDict):
    query: str
    docs: Optional[List[Document]]
    result: Optional[str]
    source: Literal["search", "vectorstore"]

# --- Nodes ---
def route(state: GraphState):
    if state["source"] == "auto":
        query = state["query"].lower()
        # Check for keywords indicating a search intent
        if any(word in query for word in ["latest", "current", "today", "news", "berita", "hari ini", "terbaru"]):
            # If keywords are found, route to search
            return {"source": "search"}
        return {"source": "vectorstore"}
    else:
        return {"source": state["source"]}


def tavily_node(state: GraphState):
    query = state["query"]
    results = search_tool.invoke(query)
    docs = [Document(page_content=res['content'], metadata={"source": res['url']}) for res in results]
    return {"docs": docs}

def vectorstore_node(state: GraphState):
    query = state["query"]
    docs = retriever.invoke(query)
    return {"docs": docs}

combine_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the provided context."),
    ("human", "Question: {query}\n\nContext:\n{context}")
])

def combine_node(state: GraphState):
    context = "\n\n".join([doc.page_content for doc in state["docs"]])
    chain = combine_prompt | llm
    result = chain.invoke({"query": state["query"], "context": context})
    return {"result": result.content}

# --- Build Graph ---
graph = StateGraph(GraphState)
graph.add_node("router", route)
graph.add_node("search", tavily_node)
graph.add_node("vectorstore", vectorstore_node)
graph.add_node("combine", combine_node)
graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda x: x["source"], {
    "search": "search",
    "vectorstore": "vectorstore"
})
graph.add_edge("search", "combine")
graph.add_edge("vectorstore", "combine")
graph.add_edge("combine", END)
agentic_rag = graph.compile()

# --- Memory and Runner ---
memory = MemorySaver()

# --- Streamlit UI ---
st.title("🤖 Agentic RAG with LangGraph")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 


with st.sidebar:
    st.image("data/dtsense.png", width=200, use_container_width=True)

    # Bio Information
    st.markdown("""
        <style>
        .center-text {
                text-align: center;
        }
        </style>
        <h2 class="center-text">DTSense AI Agent</h2>""", 
        unsafe_allow_html=True
    )
    with st.expander("🚀 About DTSense"):
        st.markdown(
            """
            <div style="text-align: justify;">
                DTSense is a platform that provides Generative AI and Agentic AI solutions for various industries, including finance, healthcare, and education. Our mission is to empower businesses with AI-driven insights and automation to enhance decision-making and operational efficiency
            </div>
        """, unsafe_allow_html=True
        )
        st.markdown(
            """
            [![Website](https://img.shields.io/badge/HTML-%23E34F26.svg?logo=html5&logoColor=white)](https://dtsense.id)

            [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/dtsense)
            """, unsafe_allow_html=True
        )

with st.sidebar:
    with st.expander("🔍 Source Options"):
        st.markdown("## Select Retrieval Source")
        retrieval_method = st.radio("Choose retrieval source:", ("Auto", "Pinecone", "Tavily"))
    with st.expander("Show Workflow Diagram"):
        st.markdown("## LangGraph Workflow")
        st.image(agentic_rag.get_graph().draw_mermaid_png())
    with st.expander("💬 Conversation History", expanded=True):
        search = st.text_input("Search history")
        filtered_history = [msg for msg in st.session_state.chat_history if search.lower() in msg["content"].lower()] if search else st.session_state.chat_history
        for i, msg in enumerate(filtered_history):
            role = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role}**: {msg['content']}")
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
        if st.download_button("⬇️ Export History", data="\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history]), file_name="chat_history.txt"):
            pass

# Input
query = st.chat_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        if retrieval_method == "Pinecone":
            source = "vectorstore"
        elif retrieval_method == "Auto":
            source = "auto"
        else:
            source = "search"
        result = agentic_rag.invoke({"query": query, "source": source}, config={"configurable": {"thread_id": "session-001"}, "checkpoint": memory})

        # Display
        # st.subheader("Answer")
        # st.write(result["result"])

        st.session_state.chat_history.append({"role": "assistant", "content": result["result"]})

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query:
    st.markdown("---")
    st.markdown("### Retrieved Documents")
    st.subheader("Context Sources")
    for i, doc in enumerate(result["docs"]):
        st.markdown(f"**Doc {i+1}:** - {doc.metadata.get('source', 'Unknown')}: content: {doc.page_content[:200]}...")


# Display memory history
if st.session_state.chat_history:
    st.sidebar.subheader("💬 History")
    for i, entry in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
        st.sidebar.markdown(f"**{i}.** {entry['role']}:")
        st.sidebar.markdown(f"→ {entry['content'][:100]}...")


