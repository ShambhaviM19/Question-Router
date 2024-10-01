import streamlit as st
import os
from dotenv import load_dotenv
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START

load_dotenv()

cassio.init(
    token=st.secrets["ASTRA_DB_APPLICATION_TOKEN"],
    database_id=st.secrets["ASTRA_DB_ID"]
)

os.environ["OPENAI_API_KEY"] =st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="AI-Powered Q&A System 🤖",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("About the Creator 👩‍🎓")
st.sidebar.markdown("# 👩‍🎓")  
st.sidebar.markdown(
    """
    **Shambhavi Mishra** is an engineering student passionate about artificial intelligence. 
    This application showcases her skills in building AI-powered systems using LangChain and Streamlit.
    
    Connect with Shambhavi:
    - [GitHub](https://github.com/ShambhaviM19)
    - [LinkedIn](https://www.linkedin.com/in/shambhavi-mishra-166163234/)
    """
)

st.title("AI-Powered Q&A System 🤖💬")
st.markdown(
    """
    This application uses a combination of vector stores and Wikipedia to answer your questions.
    It routes your query to the most appropriate source and provides informative responses.
    """
)

@st.cache_resource
def initialize_components():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="LangGraph_project",
        session=None,
        keyspace=None
    )
    
    if astra_vector_store.similarity_search("test", k=1) == []:
        st.warning("Vector store is empty. Populating with sample data...")
        populate_vector_store(astra_vector_store)
    
    retriever = astra_vector_store.as_retriever(search_kwargs={"k": 3})
    
    groq_api_key = st.secrets['GROQ_API_KEY']
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    
    return embeddings, astra_vector_store, retriever, llm

def populate_vector_store(vector_store):
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    vector_store.add_documents(doc_splits)
    st.success(f"Added {len(doc_splits)} documents to the vector store.")

embeddings, astra_vector_store, retriever, llm = initialize_components()

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    question = state["question"]
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    docs = wiki.run(question)
    wiki_results = Document(page_content=docs)
    return {"documents": wiki_results, "question": question}

def route_question(state):
    question = state["question"]
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system = """You are an expert at routing a user question to a vectorstore or wikipedia.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks on LLMs.
    Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    question_router = route_prompt | structured_llm_router
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        return "wiki_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"


workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
app = workflow.compile()

user_question = st.text_input("Ask your question here:", key="user_question")

if st.button("Get Answer", key="get_answer"):
    if user_question:
        with st.spinner("Thinking... 🤔"):
            inputs = {"question": user_question}
            result = None
            for output in app.stream(inputs):
                for key, value in output.items():
                    result = value
            
            if result:
                st.success("Here's what I found:")
                if isinstance(result['documents'], list):
                    for i, doc in enumerate(result['documents'], 1):
                        st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(doc.page_content)
                        st.markdown("---")
                else:
                    st.markdown(f"**Source:** Wikipedia")
                    st.markdown(result['documents'].page_content)
                
                response_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Use the following information to answer the user's question. If the information is not relevant, say so and provide a general answer."),
                    ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
                ])
                context = "\n\n".join([doc.page_content for doc in result['documents']] if isinstance(result['documents'], list) else [result['documents'].page_content])
                response = llm.invoke(response_prompt.format(question=user_question, context=context))
                st.markdown("**AI Assistant's Answer:**")
                st.markdown(response.content)
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.markdown("Created with ❤️ by Shambhavi Mishra")
