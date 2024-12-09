from langchain_community.document_loaders import PyPDFLoader

# List of file paths
file_paths = [
    "Reports/Annual Report 2020-21 - FINAL.pdf",
    # "Reports/Annual Report 2021-22 - FINAL.pdf",
    # "Reports/Annual Report 2022-23 - FINAL.pdf"
]

# Load documents from all files
docs = []
for path in file_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())  # Combine documents into a single list