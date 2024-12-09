from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from models import llm, retriever 

# Define the system-level prompt to guide the behavior of the language model
system_prompt = (
    "You are an intelligent assistant for answering questions. "
    "Use the provided context to answer the question accurately. "
    "If the answer is not clear from the context, respond with 'I don't know.' "
    "Keep your answers concise, maximum 4-5 points."
    "\n\n"
    "{context}"
)

# Create a chat prompt template that combines the system prompt with user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a chain to combine retrieved documents into a coherent response
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create a retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

