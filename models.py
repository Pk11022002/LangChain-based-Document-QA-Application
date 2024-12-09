import os 
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import docs

# Load environment variables from a .env file
load_dotenv()

# Initialize the Azure OpenAI embeddings model
embed_model = AzureOpenAIEmbeddings(
    openai_api_version=os.getenv("AOAI_TE3S_VERSION"),
    azure_endpoint=os.getenv("AOAI_TE3S_BASE_URL"),
    openai_api_key=os.getenv("AOAI_TE3S_KEY"),
    deployment = os.getenv("AOAI_TE3S_DEPLOYMENT"))

# Initialize the Azure OpenAI Chat model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_GPT4O_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_GPT4O_API_KEY"),
    openai_api_type="azure",
    model=os.getenv("AZURE_OPENAI_GPT4O_MODEL"),
    temperature=0)

#Initialize a text splitter for dividing large documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the input documents into chunks
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the split documents
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)

# Configure a retriever from the vector store
retriever = vectorstore.as_retriever()

