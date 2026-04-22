import os
import sys
import json
from pathlib import Path
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from nougat.utils.dataset import LazyDataset
from nougat import NougatModel
from nougat.utils.checkpoint import get_checkpoint


def nougat_extract_text(pdf_path, nougat_model):
    try:
        dataset = LazyDataset(str(pdf_path), partial=False)
        output = nougat_model.inference(dataset[0])
        return output
    except Exception as e:
        print(f"❌ Nougat failed on {pdf_path.name}: {e}")
        return ""


def parse_experimental_section(text):
    # Simple semantic chunker for "Experimental"/"Methods" section
    lowered = text.lower()
    start_idx = lowered.find("experimental")
    if start_idx == -1:
        start_idx = lowered.find("methods")
    if start_idx == -1:
        return ""
    end_keywords = ["results", "discussion", "conclusion"]
    end_idx = len(text)
    for key in end_keywords:
        idx = lowered.find(key, start_idx + 100)
        if idx != -1:
            end_idx = min(end_idx, idx)
    return text[start_idx:end_idx].strip()


def extract_documents(pdf_folder, use_nougat=True):
    print(f"📁 Parsing PDFs from: {pdf_folder}")
    docs = []
    nougat_model = None
    if use_nougat:
        checkpoint = get_checkpoint("facebook/nougat-base")
        nougat_model = NougatModel.from_pretrained(checkpoint)
        nougat_model.eval()

    for file in Path(pdf_folder).rglob("*.pdf"):
        try:
            text = ""
            if use_nougat:
                print(f"🔍 Extracting with Nougat: {file.name}")
                text = nougat_extract_text(file, nougat_model)
            else:
                print(f"📖 Extracting with PyPDF: {file.name}")
                reader = PdfReader(str(file))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)

            exp_section = parse_experimental_section(text)
            content = exp_section if exp_section else text
            docs.append(Document(page_content=content, metadata={"source": str(file)}))

        except Exception as e:
            print(f"⚠️ Failed to process {file.name}: {e}")

    print(f"✅ Extracted from {len(docs)} PDFs")
    return docs


def build_vectorstore(documents, save_path="faiss_hybrid"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(documents)

    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embed_model)
    vectordb.save_local(save_path)
    return vectordb


def load_vectorstore(save_path="faiss_hybrid"):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(save_path, embed_model)


def setup_rag_chain(vectordb):
    llm = ChatOllama(model="gemma:2b", temperature=0.2)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever(), memory=memory)


def main(pdf_folder):
    index_path = "faiss_hybrid"
    if os.path.exists(index_path):
        print("📦 Found FAISS index — loading.")
        vectordb = load_vectorstore(index_path)
    else:
        print("🔧 Building new FAISS index with Nougat+Llamaparse logic...")
        docs = extract_documents(pdf_folder, use_nougat=True)
        vectordb = build_vectorstore(docs, save_path=index_path)

    qa_chain = setup_rag_chain(vectordb)

    print("\n💬 Ask anything from your papers (type 'exit' to quit)\n")
    while True:
        question = input("🧠 You: ")
        if question.strip().lower() in ["exit", "quit"]:
            break
        response = qa_chain.run({"question": question})
        print(f"\n🤖 Gemma: {response}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hybrid_rag_pipeline.py /path/to/pdf/folder")
        sys.exit(1)
    main(sys.argv[1])
