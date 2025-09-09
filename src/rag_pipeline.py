from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from .config import GROQ_API_KEY

def build_rag_chain(retriever):
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

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

    return {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm
