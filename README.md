# My Personal AI Assistant 🚀

A high-performance, visually stunning RAG (Retrieval-Augmented Generation) assistant that empowers you to chat with your documents and the web simultaneously. 

![Logo](frontend/public/logo.png)

## ✨ Features

- 📑 **Smart Document RAG**: Upload PDFs and text files to index them into a private knowledge base.
- 🌐 **Web Scraping**: Instantly ingest web pages by URL to expand your assistant's knowledge.
- 🔍 **Hybrid Search**: Intelligently prioritizes your uploaded documents, falling back to **Google Search** when needed via Gemini's latest grounding tools.
- 💎 **Premium Glassmorphic UI**: A futuristic, responsive interface featuring dynamic animated background orbs and a sleek Turquoise/Cyan/Teal aesthetic.
- 🔥 **Firebase Powered**: Secure user authentication and persistent Firestore database for chat history and document management.
- ☁️ **Free-Tier Optimized**: Designed to run 100% on the **Firebase Spark Plan** (No Cloud Storage or Functions cost).
- 📱 **Fully Responsive**: Optimized for desktop, tablet, and mobile browsers.

## 🛠️ Tech Stack

- **Frontend**: React (Vite), Lucide Icons, Vanilla CSS (Glassmorphism Utilities).
- **Backend**: FastAPI (Python), Uvicorn.
- **Database & Auth**: Firebase Firestore, Firebase Authentication.
- **AI Models**: Google Gemini 1.5 Flash (LLM), Text-Embedding-001 (Embeddings).
- **Tools**: BeautifulSoup4 (Scraping), PDFPlumber (Data Extraction).

## 🚀 Getting Started

### Prerequisites
- Node.js & npm
- Python 3.9+
- A Google Gemini API Key
- A Firebase Project (Spark Plan)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Debashish4357/My-Personal-AI-Assistant.git
   cd My-Personal-AI-Assistant
   ```

2. **Backend Configuration**:
   - Navigate to `backend/`
   - Create a `.env` file with your `GEMINI_API_KEY`.
   - Place your `serviceAccountKey.json` from Firebase inside the `backend/` folder.
   ```bash
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. **Frontend Configuration**:
   - Navigate to `frontend/`
   - Create a `.env` file with your Firebase config (`VITE_FIREBASE_API_KEY`, etc.).
   ```bash
   npm install
   npm run dev
   ```

## 📜 License
Internal use / Personal Project.

---
*Created with ❤️ by your Personal AI Assistant.*
