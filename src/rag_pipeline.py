from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from .config import GROQ_API_KEY
from .retriever import RerankRetriever
def build_rag_chain(retriever: RerankRetriever):
    retriever_runnable = RunnableLambda(lambda question: retriever.get_relevant_documents(question))
    format_docs_runnable = RunnableLambda(lambda docs: "\n\n".join([d.page_content for d in docs]))

    prompt_template = """Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer: """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.7,
        max_tokens=512,
        groq_api_key=GROQ_API_KEY
    )

    return {
        "context": retriever_runnable | format_docs_runnable,
        "question": RunnablePassthrough()
    } | prompt | llm
