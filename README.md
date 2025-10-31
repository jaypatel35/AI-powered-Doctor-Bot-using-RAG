# 🩺 Doctor Bot- AI powered clinical screening assistant with (RAG) 

Project Details: I built a Doctor-Bot. It is AI powered conversational chatbot and a pre-visit clinical screening assistant with RAG focused on healthcare sector problem to make the first minutes of care faster and clearer using evidence-based medical knowledge base i created. Many patients struggle to explain their symptoms, and clinicians spend valuable time doing basic screening. So, my idea is to take a person’s symptoms as input and connect them to possible conditions using the medical textbook “Symptom to Diagnosis: An Evidence-Based Guide” and MedlinePlus's Health Topics data. A patient can type their symptoms, and my doctor Bot model then asks a few short follow-up questions, looks up trusted guidance using RAG pipeline I created (Retrieval-Augmented Generation) system from the sources I used the medical textbook and MedlinePlus's Health Topics data and will produce a structured triage note in form of response. This note will list possible conditions to consider not a diagnosis, urgency level, key red-flag warnings, and tests a clinician might want to discuss. My goal is not to replace doctors but to give a structured first pass guidance that lists possible diseases, severity levels, tests to consider, and red-flag warnings. This way, patients and doctors can save time during the initial consultation. So, the first few minutes of care would be faster and clearer for both patients and clinicians and that will help before a patient meets a doctor.

### Steps I took and Approaches used:

In my first 3 weeks I worked on data cleaning steps and prepared data converting the textbook raw data and MedlinePlus XML data into right text format and splitted them into smaller chunks. Then, stored the chunks in a vector database- FAISS. After, data cleaning steps, moved into chunking and retrieval, I created a complete end-to-end RAG pipeline with multiple transformation steps. Also, added prompts and safety checks, connected my retrieval results with LLM and finished with a simple Streamlit UI and did evaluation checks. And this is how my Bot works. A patient first types symptoms → My Bot then check some runs first → and asks 2-3 follow-ups → the system retrieves trusted passages from my knowledge base I created from Symptom to Diagnosis and MedlinePlus →connected those retrieval results with large langauge model GPT-3.5 which generates a structured note with citations.

I started my work by first collecting data sources I wanted to use for creating knowledge base for my RAG pipeline and started turning two messy not in right format text data sources the Symptom to Diagnosis textbook PDF and the MedlinePlus Health Topics data into clean, readable correct text format as I told. I then, splitted the text into small, meaningful chunks so that each text captured a complete idea. In my extract_textbook.py you can see my approches used to extract and clean text from my raw textbook data. I removed noisy text like page numbers, headers, and OCR errors, and normalized spacing and encoding.In my prepare_data.py it has approaches I used to parse MedlinePlus XML into a structured CSV format with consistent field names and validated entries. For this step, I included checks for missing values, duplicates, and incorrect text types before merging both sources into  (unified_medical_data.csv).

Next, I created embeddings of those chunks using BGE-small-en-v1.5 and saved them in a simple, fast index (FAISS) that is Facebook AI Similarity Search strategy. It is a library created by Meta that helps system quickly find similar items based on their meaning, not just exact words. So, I could quickly find the most relevant passages for any symptom. My RAG folder "chunker.py" has Feature Engineering steps, how I splitted the unified data into overlapping text chunks. Each chunk I created has metadata such as title, source type, and URL, which allowed my system to accurately provide source citation and filtering during retrieval. In my build_index.py, you can see my work how I created dense vector embeddings and built a FAISS similarity index. These transformations, I did for converting clean medical text into searchable numerical representations and it is the backbone of my RAG pipeline I created.

So, when a user describes symptoms, the system searches this index, pulls a few trusted passages, and as I conntected my retrieval results to LLM GPT-3.5, my system gives to it with a clear prompt that tells it exactly how to format the result like a short triage note with possible conditions (not a diagnosis), urgency level, red flags, and tests a clinician might discuss. So, it give response as per it. Every answer shows citations back to the textbook or MedlinePlus so the reader can see where the information came from and it is easy for them to trust.
 

## How my Bot follows Retrieval + Generation (RAG Pipeline) strategy:

When someone types their symptoms as input, it follows my end-to-end RAG pipeline I created and works as per that and first turn that text into an embedding and search my FAISS index for the top five most relevant chunks from Symptom to Diagnosis and MedlinePlus the sources from which I created my knowledge base.

For the final response, I had written prompts and set clear output format it should follow. So, following that my Bot asks users follow up questions just for quick checks and I then connected the retrieval results to LLM GPT-3.5 which returns the response to user’s symptoms in form of possible conditions to consider (not a diagnosis), urgency level, red-flag warnings, and tests a clinician might discuss with citations back to the exact passages that informed the answer. I had designed separate prompt for follow-up questions that forces one question at a time with 2–3 multiple-choice options, so the conversation stays simple and focused. 

##  Whom My Doctor Bot is useful?

My DoctorBot helps patients better communicate their symptoms before pre-visiting doctors by:
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
echo "OPENAI_API_KEY= Api_key" > .env # I did not include my env file here as it has my API key. You can add your own API key.
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

## Evaluation

I tested my Doctor Bot on real symptom cases like fever and cough, chest pain with shortness of breath, severe headache with light sensitivity, stomach issues. And calculated three metrics and observed how much accurate answer my system was giving. It gave me: 
- Faithfulness: 0.83
- Answer Relevancy: 0.90
- Context Precision: 0.99
So, from this results, I confirmed that  retrieval was very accurate and the answers are mostly relevant, but faithfulness was still less in some cases I tested and needs improvement.

##  Key Learnings

- RAG implementation from scratch
- Medical AI safety considerations
- Prompt engineering for structured outputs
- Balancing accuracy vs. cost

## References:

Stern, S. D. C., Cifu, A. S., & Altkorn, D. (2020). Symptoms to diagnosis: An evidence-based guide (4th ed.). McGraw-Hill Education. Retrieved from: https://www.mheducation.com/highered/mhp/product/symptom-diagnosis-evidence-based-guide-fourth-edition.html

National Library of Medicine (US). (n.d.). MedlinePlus XML file. MedlinePlus. Retrieved from https://medlineplus.gov/xml.html

Streamlit UI used to build my bot for testing purpose. Retrieved from: https://streamlit.io/

LangChain Documentation. Available at https://python.langchain.com

Kumar, S., & contributors. (2025). python-dotenv. Retrieved from: https://github.com/theskumar/python-dotenv

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proceedings of EMNLP-IJCNLP 2019, 3980–3990. Retrieved from: https://aclanthology.org/D19-1410.pdf

Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. Proceedings of EMNLP 2020, 4512–4525. https://aclanthology.org/2020.emnlp-main.365

UKP Lab. (2025). Sentence-Transformers. Retrieved from: https://sbert.net/

Johnson, J., Douze, M., & Jégou, H. (2017). Billion-scale similarity search with GPUs. arXiv:1702.08734. https://arxiv.org/abs/1702.08734

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32. https://arxiv.org/pdf/1912.01703

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAs: Automated evaluation of retrieval augmented generation. Proceedings of the 18th Conference of the European Chapter of the ACL: System Demonstrations, 150–158. https://aclanthology.org/2021.emnlp-demo.21

Pandas, Document- Used for data processing. Retrieved from: https://pandas.pydata.org/docs/

NumPy Documentation-Used for data processing. https://numpy.org/doc/

Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E., Lomeli, M., Hosseini, L., & Jégou, H. (2024). The Faiss library. Retrieved from: https://arxiv.org/abs/2401.08281

OpenAI API Documentation. https://platform.openai.com/docs

Meta AI. (2025). FAISS. Retrieved from: https://faiss.ai/

LangChain AI. (2025). langchain-community. Retrieved from: https://pypi.org/project/langchain-community/

LangChain AI. (2025). langchain-openai documentation: https://pypi.org/project/langchain-openai

OpenAI. (2025). ChatGPT, Large language model. Retrieved from https://chatgpt.com. Used to troubleshoot several technical errors and to better understand certain topics in depth.

Anthropic. (2025). Claude. Retrieved from https://claude.ai Used for understanding errors and to get idea what went wrong.

## Disclaimer

This is a screening tool, NOT medical advice. Always consult healthcare professionals.

---

*Built as part of MSDS-692 Data Science Practicum*
