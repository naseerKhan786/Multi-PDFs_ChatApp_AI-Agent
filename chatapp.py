import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------
# PDF Text Extraction
# -------------------------
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# -------------------------
# Text Chunking
# -------------------------
def get_text_chunks(text):
    """Splits text into overlapping chunks for better embedding coverage."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# -------------------------
# Vector Store Creation
# -------------------------
def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# -------------------------
# Conversational Chain
# -------------------------
def get_conversational_chain():
    """Sets up the QA chain using Gemini Pro."""
    prompt_template = """
    Answer the question as thoroughly as possible using only the provided context.
    If the answer is not contained in the context, say "Answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# -------------------------
# Handle User Query
# -------------------------
def user_input(user_question):
    """Retrieves similar chunks from FAISS and queries the model."""
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process PDFs first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.markdown(f"### 🤖 Answer:\n{response['output_text']}")


# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config("Multi-PDF Chatbot", page_icon="📚")
    st.title("📚 Multi-PDF Chatbot 🤖")
    st.write("Upload multiple PDF files and ask questions about their content!")

    # User Question
    user_question = st.text_input("💬 Ask a question about your uploaded PDFs:")
    if user_question:
        user_input(user_question)

    # Sidebar for PDF Uploads
    with st.sidebar:
        st.header("📁 Upload & Process PDFs")

        # Optional images (handled gracefully)
        try:
            st.image("img/Robot.jpg", use_container_width=True)
        except Exception:
            st.warning("🤖 Robot image not found.")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files:",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("⚙️ Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs... This may take a moment ⏳"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ PDFs processed and indexed successfully!")

        st.write("---")
        try:
            st.image("img/gkj.jpg", use_container_width=True)
        except Exception:
            pass
        st.caption("AI App created by @ Gurpreet Kaur")

    # Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; 
                    background-color: #0E1117; padding: 12px; text-align: center; font-size: 14px;">
            © <a href="https://github.com/gurpreetkaurjethra" target="_blank">Gurpreet Kaur Jethra</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
