from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

loader=PyPDFLoader("one.pdf")

docs=loader.load()

text = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

db.save_local("faiss_local")

reteriver = db.as_retriever(
    search_kwargs={"k":5}
)



llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:

"I could not find the answer in the document."

Context:
{content}

Question:
{query}

Answer:
""",
input_variables=["content","query"]
)

chain= prompt | llm | parser


while True:

    query = input("\nYou : ")

    if query.lower() == "exit":
        break

    docs = reteriver.invoke(query)

    content = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    result = chain.invoke({
        "content": content,
        "query": query
    })

    print("\nAI :", result)


print(result)