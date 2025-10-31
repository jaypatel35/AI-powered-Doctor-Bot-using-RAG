# 🩺 AI DoctorBot with RAG

An intelligent medical screening assistant powered by Retrieval-Augmented Generation (RAG) using evidence-based medical knowledge.

##  Overview

DoctorBot helps patients better communicate their symptoms before doctor visits by:
- Asking intelligent follow-up questions
- Detecting medical emergencies
- Providing evidence-based assessments
- Citing medical sources for transparency

##  Features

-  **Dual Knowledge Base**: 600-page medical textbook + MedlinePlus (2,333 chunks)
-  **Smart Follow-ups**: 3 contextual questions for better diagnosis
-  **Emergency Detection**: Immediate alerts for critical symptoms
-  **RAG Pipeline**: FAISS + BGE embeddings + GPT-3.5
-  **Source Transparency**: Shows which medical texts informed the response
-  **Clean UI**: Accessible Streamlit interface

##  Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation
```bash
cd AI-Doctor-bot-with-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY= Api_key" > .env
```

### Run
```bash
streamlit run ui/app.py
```

Open browser: http://localhost:8501

##  Architecture
```
User Input → Emergency Check → Medical Check → Follow-Ups (3x) → 
RAG Retrieval → GPT-3.5 → Structured Diagnosis
```

**Tech Stack:**
- Embeddings: BGE-small-en-v1.5
- Vector DB: FAISS
- LLM: GPT-3.5-turbo
- Framework: LangChain
- UI: Streamlit

##  Evaluation

- **Faithfulness**: 0.71
- **Answer Relevancy**: 0.87
- **Context Precision**: 0.94

##  Key Learnings

- RAG implementation from scratch
- Medical AI safety considerations
- Prompt engineering for structured outputs
- Balancing accuracy vs. cost

## Disclaimer

This is a screening tool, NOT medical advice. Always consult healthcare professionals.


---

*Built as part of MSDS-692 Data Science Practicum*
