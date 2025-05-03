# Multimodal-RAG

---

## 🚀 Multimodal RAG System – PDF, Audio & Video Querying

This project implements a powerful **Multimodal Retrieval-Augmented Generation (RAG)** system capable of answering user queries from **PDFs, audio files, and videos** using advanced embedding and retrieval techniques. It features an **interactive Streamlit interface** for smooth user interaction.

---

### 📌 Key Features

* 🔍 **Multimodal Input Support**
  Ingest and query across **PDF documents**, **audio files**, and **video content**.

* ⚡ **RAG Pipeline with FAISS**
  Efficient document chunking and retrieval using **FAISS vector database** for semantic search.

* 🤖 **LLM-Powered QA**
  Leverages Large Language Models to generate accurate, context-aware answers from retrieved content.

* 🛠️ **Modular Codebase**

  * `main.py` handles backend processing: file ingestion, vectorization, and retrieval logic.
  * `app.py` provides an interactive **Streamlit-based UI** for seamless user experience.

---

### 🧠 How It Works

1. **PDFs**: Extracts text and splits it into meaningful chunks for embedding and retrieval.
2. **Audio Files**: Converts speech to text using a speech recognition model, then embeds the transcribed text.
3. **Video Files**:

   * Extracts frames from the video and stores them in a temporary directory.
   * Passes the frame directory to a **vision-language model** to generate descriptions of the visual content.
   * Stores these frame descriptions in the FAISS vector database.
   * Answers user queries using the most relevant visual-text descriptions.
4. **Retrieval & Answer Generation**:

   * Embeds the user query.
   * Retrieves top-k relevant context chunks from the FAISS index.
   * Feeds the context and query into a generative model to produce the final answer.

---

### 🧪 Supported Use Cases

* AI document assistants
* Audio/video lecture Q\&A
* Legal or research document analysis
* Customer support knowledge systems

---
