from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def load_documents(file_type, path):
    if file_type == 1:
        loader = PyPDFLoader(path)
    elif file_type == 2:
        loader = TextLoader(path, encoding="utf-8")
    else:
        loader = WebBaseLoader(path)
    return loader.load()

def build_qa_system(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        openai_api_key="YOUR API KEY",
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0,
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

file_type = int(input("Enter Number\n1. PDF FILE\n2. TEXT FILE\n3. Webpage\n"))
path = input("Enter path or URL: ")
documents = load_documents(file_type, path)
qa = build_qa_system(documents)

print("Summary:")
print(qa.run("Give me the Summary of the whole content."))
print("Feel free to ask me any query!")

while True:
    query = input("Query (press enter to exit): ")
    if not query:
        print("Have a nice day.")
        break
    print("Answer:", qa.run(query))
