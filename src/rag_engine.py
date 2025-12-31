from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def setup_rag_chain(texts):
    """
    Takes split text, creates a Vector Store, and returns a queryable chain.
    
    Args:
        texts: List of document chunks from text splitting
        
    Returns:
        A retrieval chain that can answer questions about the documents
    """

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )


    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


    system_prompt = (
        "You are an expert legal assistant analyzing a contract. "
        "Use the following pieces of context from the contract to answer the user's question. "
        "\n\n"
        "Guidelines:\n"
        "- Answer based ONLY on the provided context\n"
        "- If the answer is not in the context, say 'I cannot find this information in the contract'\n"
        "- Cite specific sections, clauses, or page numbers when possible\n"
        "- Explain legal terms in plain English\n"
        "- Be precise and professional\n"
        "- If there are multiple relevant clauses, mention all of them\n"
        "\n\n"
        "Context from the contract:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])


    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain