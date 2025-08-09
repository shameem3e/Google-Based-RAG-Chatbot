# 📚 Google-Based RAG Chatbot (Terminal)

A lightweight Retrieval-Augmented Generation (RAG) system that uses **Google search results** as the knowledge source, stores them in a **FAISS vector database**, and allows you to chat with the system via the terminal — **without using ChatGPT or paid APIs** (except optional Google API).

---

## 📌 Features
- 🔍 **Google Search Integration** (via API or scraping)
- 🧠 **Local Embeddings** using Hugging Face model (`all-MiniLM-L6-v2`)
- 🗄 **FAISS Vector Store** for fast retrieval
- ✂ **Chunking & Cleaning** for optimal embedding quality
- 💬 **Terminal Chat Interface**
- 📂 **Offline Mode** after first data fetch
- 🚀 **No dependency on ChatGPT/OpenAI API**

---

## 📂 Folder Structure

Google-RAG-Chatbot/

│── src/

│ ├── google_search.py # Search and scrape results

│ ├── text_utils.py # Chunking & cleaning helpers

│ ├── embed_store.py # Embedding and FAISS storage

│ ├── retrieve_answer.py # Retrieve and generate answer

│ ├── chat.py # Terminal chat interface

│── data/

│ ├── faiss_index.bin # Vector DB file

│ ├── metadata.pkl # Mapping of chunks to sources

│── requirements.txt # Python dependencies

│── README.md # Documentation


---

## ⚙️ Setup Guide

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/shameem3e/Google-Based-RAG-Chatbot.git
cd Google-RAG-Chatbot

```
### **2️⃣ Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

```
### **3️⃣ Install Requirements**
```bash
pip install -r requirements.txt

```

### **4️⃣ (Optional) Set Google API Credentials**
If you use the Google Custom Search API:

* Create API key & Search Engine ID from [Google Custom Search](https://developers.google.com/custom-search/v1/overview)
* Replace them in `src/google_search.py`

If you want free scraping mode (less reliable), no API key needed.

### **5️⃣ Run Embedding Step**
This will:

* Search Google for your topic
* Scrape results**
* Clean & chunk text**
* Embed & store in FAISS**

```bash
python src/embed_store.py

```
### **6️⃣ Start Chat**
```bash
python src/chat.py

```
Type your question and chat with your RAG system.

## 📜 Code Overview
#### 1. google_search.py
Handles Google search (API or scraping) and returns cleaned text from results.

#### 2. text_utils.py
* `chunk_text()` → Splits long text into overlapping chunks.
* `clean_text()` → Removes extra spaces, newlines, and unwanted symbols.

#### 3. embed_store.py
* Fetches search results
* Cleans and chunks text
* Creates embeddings with `all-MiniLM-L6-v2`
* Stores in FAISS

#### 4. retrieve_answer.py
* Loads FAISS index
* Finds most relevant chunks for a given query
* Returns top matches

#### 5. chat.py
* Simple terminal interface
* Lets you query the RAG system interactively

## ❓ FAQ
Q1: Do I need an API key?

No — scraping mode works without an API key, but API mode is more reliable.

Q2: Can I use this without an internet connection?

Yes — once embeddings are stored in FAISS, you can run the chat locally.

Q3: Which embedding model is used?

[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from Hugging Face.

Q4: Can I change the chunk size?

Yes — edit the `chunk_size` parameter in `text_utils.py`.

Q5: Will Google block my requests?

Possibly, if you make too many requests in scraping mode. Use delays or the official API for stability.

## 🛠 Tech Stack
* Python 3.8+
* Hugging Face Transformers
* FAISS (Facebook AI Similarity Search)
* BeautifulSoup4 for HTML parsing
* Google API / googlesearch-python for search

## 🚀 Future Improvements
* Add summarization of retrieved results
* Integrate speech-to-text & text-to-speech
* Add web interface (Flask/Streamlit)

## 👨‍💻 Author
[MD. Shameem Ahammed](https://sites.google.com/view/shameem3e)
Graduate Student, AI & ML Enthusiast

---

If you want, I can also add a **"Preview" section** with a screenshot of your terminal chatbot in action so the GitHub page looks more engaging. Would you like me to make that?

