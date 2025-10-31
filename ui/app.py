import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.rag_pipeline import RAGPipeline, ConversationManagerWithFollowUps


# Page config
st.set_page_config(
    page_title="DoctorBot | Medical Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CSS with visible buttons
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background-color: #ffffff !important;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 900px;
        background-color: #ffffff !important;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 2px solid #000000;
        margin-bottom: 2rem;
        background-color: #ffffff;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000 !important;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #666666 !important;
        margin-top: 0.5rem;
    }
    
    /* Buttons - highly visible blue */
    .stButton>button {
        background-color: #0066cc !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.2s;
        width: 100%;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.3) !important;
    }
    
    .stButton>button:hover {
        background-color: #0052a3 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.4) !important;
        transform: translateY(-1px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Text inputs */
    .stTextArea textarea, .stTextInput input {
        border: 2px solid #cccccc !important;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1rem;
        color: #000000 !important;
        background: #ffffff !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #0066cc !important;
        outline: none;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f7ff !important;
        border: 1px solid #b3d9ff !important;
        border-left: 4px solid #0066cc !important;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #000000 !important;
    }
    
    .info-box strong {
        color: #000000 !important;
        display: block;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Emergency box */
    .emergency-box {
        background: #fff5f5 !important;
        border: 2px solid #dc3545 !important;
        border-left: 6px solid #dc3545 !important;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        color: #000000 !important;
    }
    
    .emergency-title {
        color: #dc3545 !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background: #f0f9f4 !important;
        border: 2px solid #28a745 !important;
        border-left: 6px solid #28a745 !important;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #000000 !important;
    }
    
    /* Warning box */
    .warning-box {
        background: #fffbf0 !important;
        border: 2px solid #ffc107 !important;
        border-left: 6px solid #ffc107 !important;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #000000 !important;
    }
    
    /* Progress */
    .progress-box {
        background: #e7f3ff !important;
        border: 2px solid #0066cc !important;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        color: #000000 !important;
    }
    
    .progress-box strong {
        color: #0066cc !important;
        font-size: 1.1rem;
    }
    
    /* RAG container */
    .rag-container {
        background: #ffffff !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .rag-status {
        font-weight: 600;
        color: #000000 !important;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .rag-detail {
        color: #666666 !important;
        font-size: 0.95rem;
        margin: 0.25rem 0;
    }
    
    /* Source items */
    .source-section {
        margin-top: 1.5rem;
    }
    
    .source-section h4 {
        color: #000000 !important;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .source-item {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    
    .source-item a {
        color: #0066cc !important;
        text-decoration: none;
        font-weight: 500;
    }
    
    .source-item a:hover {
        text-decoration: underline;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #dee2e6;
    }
    
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stChatMessage * {
        color: #000000 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px;
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* Remove branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Text colors */
    p, span, div, label, li {
        color: #000000 !important;
    }
    
    a {
        color: #0066cc !important;
    }
    
    hr {
        border: none;
        border-top: 1px solid #dee2e6;
        margin: 2rem 0;
    }
    
    /* Feature list */
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-list li {
        padding: 0.5rem 0;
        color: #000000 !important;
    }
    
    .feature-list li:before {
        content: "✓ ";
        color: #28a745;
        font-weight: bold;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    project_root = Path(__file__).parent.parent
    store_dir = project_root / 'store'
    return RAGPipeline(store_dir)


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_manager' not in st.session_state:
        pipeline = load_pipeline()
        st.session_state.conversation_manager = ConversationManagerWithFollowUps(pipeline, num_followups=3)
    if 'stage' not in st.session_state:
        st.session_state.stage = 'initial'
    if 'current_question_num' not in st.session_state:
        st.session_state.current_question_num = 0


def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🩺 DoctorBot</div>
        <div class="main-subtitle">AI-Powered Medical Screening Assistant</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Clean sidebar without emergency box."""
    with st.sidebar:
        st.markdown("## About")
        st.write("DoctorBot analyzes symptoms using evidence-based medical knowledge from textbooks and MedlinePlus.")
        
        st.markdown("---")
        
        st.markdown("## Features")
        st.markdown("""
        <ul class="feature-list">
            <li>Emergency symptom detection</li>
            <li>Follow-up questions</li>
            <li>Evidence-based diagnosis</li>
            <li>Dual knowledge sources</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("## Privacy")
        st.markdown("""
        <ul class="feature-list">
            <li>Secure & Private</li>
            <li>Not stored</li>
            <li>For screening only</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.caption("🕐 Always consult a healthcare provider for medical advice")


def display_emergency_alert(response):
    categories_text = ', '.join(response['categories'])
    
    st.markdown(f"""
    <div class="emergency-box">
        <div class="emergency-title">🚨 EMERGENCY DETECTED</div>
        <p><strong>Severity:</strong> {response['severity']}</p>
        <p><strong>Type:</strong> {categories_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.error(response['content'])


def display_progress(current, total):
    st.markdown(f"""
    <div class="progress-box">
        <strong>Follow-Up Questions: {current} of {total}</strong><br>
        <small>Gathering information for accurate assessment</small>
    </div>
    """, unsafe_allow_html=True)


def display_rag_info(metadata):
    if not metadata:
        return
    
    st.markdown("---")
    
    used_rag = metadata.get('used_rag', False)
    reason = metadata.get('reason', 'Unknown')
    sources = metadata.get('sources', [])
    
    status_icon = "✅" if used_rag else "ℹ️"
    status_text = "Using Medical Knowledge Base" if used_rag else "Using General AI Knowledge"
    
    st.markdown(f"""
    <div class="rag-container">
        <div class="rag-status">{status_icon} {status_text}</div>
        <div class="rag-detail">Source: {reason}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if used_rag and sources:
        st.markdown("### 📚 Referenced Sources")
        
        textbook_sources = [s for s in sources if s['source_type'] == 'textbook']
        medline_sources = [s for s in sources if s['source_type'] == 'medlineplus']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if textbook_sources:
                st.markdown("""
                <div class="source-section">
                    <h4>📖 Clinical Textbook</h4>
                """, unsafe_allow_html=True)
                for i, source in enumerate(textbook_sources, 1):
                    st.markdown(f"""
                    <div class="source-item">
                        {i}. Symptom to Diagnosis<br>
                        <small>Relevance: {source['relevance_score']:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if medline_sources:
                st.markdown("""
                <div class="source-section">
                    <h4>📚 MedlinePlus</h4>
                """, unsafe_allow_html=True)
                for i, source in enumerate(medline_sources, 1):
                    st.markdown(f"""
                    <div class="source-item">
                        {i}. <a href="{source['url']}" target="_blank">{source['title']}</a><br>
                        <small>Relevance: {source['relevance_score']:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


def main():
    initialize_session_state()
    render_header()
    render_sidebar()
    
    # Disclaimer
    with st.expander("⚠️ Medical Disclaimer", expanded=False):
        st.markdown("""
        <div class="warning-box">
            <strong>This tool is for informational purposes only</strong><br><br>
            • Not a substitute for professional medical advice<br>
            • Always consult qualified healthcare providers<br>
            • Call 911 for medical emergencies<br>
            • AI may make mistakes - verify with your doctor
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role, avatar="👤" if role == "user" else "🩺"):
            if message.get("is_emergency"):
                display_emergency_alert({
                    'content': content,
                    'severity': message.get('severity', 'HIGH'),
                    'categories': message.get('categories', [])
                })
            else:
                st.markdown(content)
    
    # Complete
    if st.session_state.stage == 'complete':
        st.markdown("""
        <div class="success-box">
            <strong>✓ Assessment Complete</strong>
        </div>
        """, unsafe_allow_html=True)
        
        if 'metadata' in st.session_state:
            display_rag_info(st.session_state.metadata)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Start New Assessment", use_container_width=True):
                st.session_state.messages = []
                pipeline = load_pipeline()
                st.session_state.conversation_manager = ConversationManagerWithFollowUps(pipeline, num_followups=3)
                st.session_state.stage = 'initial'
                st.session_state.current_question_num = 0
                if 'metadata' in st.session_state:
                    del st.session_state.metadata
                st.rerun()
        return
    
    # Progress
    if st.session_state.stage == 'followup' and st.session_state.current_question_num > 0:
        display_progress(st.session_state.current_question_num, 3)
    
    # Input
    if st.session_state.stage == 'initial':
        st.markdown("### Describe Your Symptoms")
        
        st.markdown("""
        <div class="info-box">
            <strong>For Best Results:</strong>
            • Be specific about symptoms and duration<br>
            • Mention severity (mild/moderate/severe)<br>
            • Include relevant medical history
        </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Your symptoms:",
            placeholder="Example: I've had a persistent fever of 101°F for 3 days, with severe headaches...",
            height=150,
            key="symptom_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.button("Analyze Symptoms", type="primary", use_container_width=True)
        
        if submit and user_input.strip():
            process_input(user_input)
            st.rerun()
    
    elif st.session_state.stage == 'followup':
        user_input = st.text_input(
            "Your answer:",
            key=f"followup_input_{st.session_state.current_question_num}"
        )
        
        if st.button("Submit Answer", type="primary"):
            if user_input.strip():
                process_input(user_input)
                st.rerun()


def process_input(user_input):
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.spinner("Processing..."):
        response = st.session_state.conversation_manager.process_message(user_input)
    
    response_type = response['type']
    
    if response_type == 'emergency':
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['content'],
            "is_emergency": True,
            "severity": response['severity'],
            "categories": response['categories']
        })
        st.session_state.stage = 'emergency'
    
    elif response_type == 'rejection':
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['content']
        })
        st.session_state.stage = 'initial'
    
    elif response_type == 'followup_question':
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['content']
        })
        st.session_state.stage = 'followup'
        st.session_state.current_question_num = response['question_num']
    
    elif response_type == 'diagnosis':
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['content']
        })
        st.session_state.stage = 'complete'
        st.session_state.metadata = {
            'sources': response.get('sources', []),
            'used_rag': response.get('used_rag', False),
            'reason': response.get('reason', 'Unknown'),
            'best_score': response.get('best_score')
        }


if __name__ == "__main__":
    main()