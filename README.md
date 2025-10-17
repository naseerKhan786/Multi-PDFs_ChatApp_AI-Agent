# Multi-PDFs_ChatApp_AI-Agent
📚 Multi-PDF Chatbot 🤖

Chat with multiple PDF files using Google Gemini + LangChain + Streamlit

🧠 Overview

This app allows users to upload multiple PDF files and ask natural language questions about their contents.
It uses Google’s Gemini API (via LangChain) for question answering and FAISS for fast semantic search.

💡 Perfect for research papers, legal documents, reports, or any multi-PDF analysis.

⚙️ Tech Stack

Python 3.10+

Streamlit – for interactive web UI

LangChain – for building the QA chain

Google Gemini (ChatGoogleGenerativeAI) – for LLM responses

FAISS – for semantic vector search

PyPDF2 – for text extraction from PDFs

dotenv – for environment variable management

🚀 Features

✅ Upload and process multiple PDF files
✅ Extract and embed text chunks using Google embeddings
✅ Ask natural-language questions about your documents
✅ Get detailed, context-aware answers
✅ Persistent FAISS index (no need to reprocess every time)
✅ Polished Streamlit UI with sidebar and footer

🧩 Project Structure
multi-pdf-chatbot/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── .env                   # Contains your Google API Key
├── img/
│   ├── Robot.jpg
│   └── gkj.jpg
└── README.md              # This file

🔑 Environment Setup

Clone this repository:

git clone https://github.com/<your-username>/multi-pdf-chatbot.git
cd multi-pdf-chatbot


Create a virtual environment:

python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows


Install dependencies:

pip install -r requirements.txt


Add your Google API key:

Create a .env file in the project root and add:

GOOGLE_API_KEY=your_google_api_key_here


You can get an API key from Google AI Studio
.

🧾 requirements.txt

Here’s a sample requirements.txt you can include:

streamlit
PyPDF2
langchain
langchain-google-genai
faiss-cpu
python-dotenv
google-generativeai

▶️ Run the App

Once everything is set up, start the Streamlit app:

streamlit run app.py


Then open your browser at:

http://localhost:8501

💬 How It Works

Upload one or more PDFs via the sidebar.

Click “Submit & Process” to extract text and build the FAISS index.

Type a question in the input box — the app finds the most relevant text chunks and generates a detailed answer using Gemini.

Enjoy an AI-powered reading assistant for your documents!

📸 Example UI
📚 Multi-PDF Chatbot 🤖
--------------------------------
💬 Ask a question about your uploaded PDFs:
> What is the summary of the first chapter?

🤖 Answer:
The first chapter discusses...

⚠️ Troubleshooting

FAISS Load Error:
Add this fix when loading the index:

FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


Scanned PDFs not working:
PyPDF2 only extracts digital text, not OCR.
Use pytesseract if you need OCR support.

API Key issues:
Make sure your .env file is correctly configured and not shared publicly.

❤️ Credits

Developed by Gurpreet Kaur Jethra

Built with 💡 LangChain + Gemini + Streamlit
