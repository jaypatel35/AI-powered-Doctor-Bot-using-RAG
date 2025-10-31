import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv
from openai import OpenAI
from rag.emergency_detector import EmergencyDetector


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import MedlineRetriever
from rag.prompts import create_diagnosis_prompt

# Load environment variables
load_dotenv()


class RAGPipeline:
    """Manages the complete RAG workflow for medical symptom checking."""
    
    def __init__(self, store_dir: Path):
        """
        Initialize RAG pipeline.
        
        Args:
            store_dir: Directory containing FAISS index and metadata
        """
        print("🚀 Initializing RAG Pipeline...")
        
        # Initialize retriever
        data_dir = Path(__file__).parent.parent / 'data'
        self.retriever = MedlineRetriever(store_dir, data_dir)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        
        print("✅ RAG Pipeline ready!")
    
    def check_medical_relevance(self, user_input: str) -> bool:
        """
        Check if the user input is related to medical/health topics.
        
        Returns:
            bool: True if medical-related, False otherwise
        """
        print(f"🔍 Checking if query is medical-related...")
        
        prompt = f"""You are a medical query classifier. Determine if the following user input is related to health, medical symptoms, diseases, or healthcare.

User input: "{user_input}"

Respond with ONLY "YES" if it's medical/health-related, or "NO" if it's not.

Examples:
- "I have a headache" -> YES
- "What causes diabetes?" -> YES
- "How to treat a cold?" -> YES
- "What is the use of a pen?" -> NO
- "Tell me a joke" -> NO
- "What's the weather?" -> NO

Answer (YES or NO):"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            is_medical = "YES" in answer
            
            print(f"   {'✓' if is_medical else '✗'} Query classified as: {'Medical' if is_medical else 'Non-medical'}")
            return is_medical
            
        except Exception as e:
            print(f"   ⚠️ Error in classification: {e}")
            # Default to True to avoid blocking legitimate queries
            return True
    
    def generate_diagnosis(
        self, 
        user_symptoms: str,
        use_rag: bool = True
    ) -> Dict[str, any]:
        """
        Generate diagnosis from symptoms.
        
        Args:
            user_symptoms: User's symptom description
            use_rag: Whether to use RAG retrieval (default True)
        
        Returns:
            Dict with diagnosis, sources, and metadata
        """
        # First check if query is medical-related
        if not self.check_medical_relevance(user_symptoms):
            return {
                'diagnosis': """I am a medical symptom checker AI assistant, designed specifically to help with health-related questions and symptoms.

I can only assist with:
- Medical symptoms and health concerns
- Disease information and conditions
- Health guidance and recommendations
- Medical questions and clarifications

For non-medical questions, please use a general-purpose AI assistant or search engine.

If you have any health-related symptoms or medical questions, I'm here to help! Please describe your symptoms.""",
                'sources': [],
                'used_rag': False,
                'reason': 'Non-medical query rejected'
            }
        
        print(f"🔍 Retrieving relevant medical information...")
        
        # Try RAG retrieval first
        results = self.retriever.retrieve(user_symptoms, top_k=5)
        
        # Check relevance of retrieved results
        # If best score is too high (indicating poor match), fall back to LLM
        best_score = results[0]['score'] if results else float('inf')
        relevance_threshold = 0.7  # Lower score = better match in L2 distance
        
        if best_score > relevance_threshold:
            use_rag = False
            print(f"⚠️ Low relevance score ({best_score:.3f} > {relevance_threshold})")
            print(f"   Falling back to LLM general medical knowledge")
        
        if use_rag and results:
            context = self.retriever.format_context(results)
            print(f"✓ Retrieved {len(results)} relevant sources")
            print(f"   - Best match score: {best_score:.3f}")
            
            # Count sources
            textbook_count = sum(1 for r in results if r['source_type'] == 'textbook')
            medline_count = sum(1 for r in results if r['source_type'] == 'medlineplus')
            print(f"   - Textbook sources: {textbook_count}")
            print(f"   - MedlinePlus sources: {medline_count}")
        else:
            context = "No specific medical reference information retrieved. Using general medical knowledge."
            print(f"⚠️ No relevant sources found - using LLM general knowledge")
        
        # Create conversation history
        conversation_history = [
            {"role": "user", "content": user_symptoms}
        ]
        
        # Generate diagnosis
        if use_rag and results:
            prompt = create_diagnosis_prompt(conversation_history, context)
            system_message = "You are a knowledgeable medical AI assistant with access to MedlinePlus and medical textbook information."
        else:
            # Fallback prompt without RAG
            prompt = f"""Based on the patient's symptoms: "{user_symptoms}"

Provide a medical assessment using your general medical knowledge. Include:

## Likely Condition
[Explain possible conditions based on the symptoms]

## Expected Progression (30/60/90 Days)
- **30 days**: [Short-term outlook]
- **60 days**: [Medium-term outlook]
- **90 days**: [Long-term outlook]

## Lifestyle Recommendations
[Self-care and lifestyle advice]

## Red Flags & Next Steps
[Warning signs requiring immediate medical attention]

## Important Note
⚠️ **Note**: This response is based on general medical knowledge as specific reference materials did not contain directly relevant information for your symptoms. For accurate diagnosis and treatment, please consult a healthcare provider.

## Important Disclaimer
This AI assessment is for informational purposes only and is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment."""
            
            system_message = "You are a knowledgeable medical AI assistant."
        
        print(f"🤖 Generating diagnosis with GPT-3.5...")
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        diagnosis_text = response.choices[0].message.content.strip()
        
        # Extract sources for citations
        sources = []
        if use_rag and results:
            sources = [
                {
                    'title': result['title'],
                    'url': result['url'],
                    'source_type': result['source_type'],
                    'relevance_score': result['score']
                }
                for result in results
            ]
        
        return {
            'diagnosis': diagnosis_text,
            'sources': sources,
            'used_rag': use_rag and bool(results),
            'reason': 'RAG retrieval successful' if (use_rag and results) else 'No relevant sources - using LLM knowledge',
            'best_score': best_score if results else None
        }


class ConversationManager:
    """Manages conversation state for a symptom checking session."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.pipeline = rag_pipeline
        self.conversation_history: List[Dict[str, str]] = []
        self.stage = "initial"
    
    def add_user_message(self, message: str):
        """Add user message to conversation history."""
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
    
    def add_assistant_message(self, message: str):
        """Add assistant message to conversation history."""
        self.conversation_history.append({
            'role': 'assistant',
            'content': message
        })
    
    def process_message(self, user_input: str) -> Dict:
        """
        Process user message and return diagnosis.
        
        Returns:
            Dict with 'type', 'content', 'sources', and metadata
        """
        self.add_user_message(user_input)
        
        if self.stage == "initial":
            self.stage = "diagnosis"
            result = self.pipeline.generate_diagnosis(user_input)
            self.add_assistant_message(result['diagnosis'])
            self.stage = "complete"
            return {
                'type': 'diagnosis',
                'content': result['diagnosis'],
                'sources': result['sources'],
                'used_rag': result.get('used_rag', False),
                'reason': result.get('reason', ''),
                'best_score': result.get('best_score')
            }
        
        elif self.stage == "complete":
            return {
                'type': 'complete',
                'content': "Thank you for using the symptom checker. If you have new symptoms, please start a new session."
            }


class ConversationManagerWithFollowUps:
    """Enhanced conversation manager with follow-up questions."""
    
    def __init__(self, rag_pipeline: RAGPipeline, num_followups: int = 3):
        self.pipeline = rag_pipeline
        self.num_followups = num_followups
        self.conversation_history: List[Dict[str, str]] = []
        self.stage = "initial"  # initial, followup, diagnosis, complete
        self.followup_count = 0
        
        # Import here to avoid circular imports
        sys.path.append(str(Path(__file__).parent.parent))
        from rag.followup_manager import FollowUpManager
        self.followup_manager = FollowUpManager()
    
    def add_user_message(self, message: str):
        """Add user message to conversation history."""
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
    
    def add_assistant_message(self, message: str):
        """Add assistant message to conversation history."""
        self.conversation_history.append({
            'role': 'assistant',
            'content': message
        })


    def process_message(self, user_input: str) -> Dict:
        """
        Process user message through the conversation flow:
        0. Check for emergency symptoms (if initial)
        1. Initial symptoms
        2. Follow-up questions (3-4)
        3. Final diagnosis
        
        Returns:
            Dict with 'type', 'content', and metadata
        """
        # Add user input to history
        self.add_user_message(user_input)
        
        # Initial symptoms
        if self.stage == "initial":
            # FIRST: Check for emergency symptoms
            emergency_detector = EmergencyDetector()
            emergency_result = emergency_detector.detect(user_input)
            
            if emergency_result['is_emergency']:
                print(f"\n🚨 EMERGENCY DETECTED: {emergency_result['severity']}")
                print(f"   Categories: {emergency_result['categories']}")
                
                return {
                    'type': 'emergency',
                    'content': emergency_result['message'],
                    'severity': emergency_result['severity'],
                    'categories': emergency_result['categories'],
                    'keywords': emergency_result['matched_keywords'],
                    'stage': 'emergency'
                }
            
            # Check if medical-related
            if not self.pipeline.check_medical_relevance(user_input):
                return {
                    'type': 'rejection',
                    'content': """I am a medical symptom checker AI assistant, designed specifically to help with health-related questions and symptoms.

I can only assist with:
- Medical symptoms and health concerns
- Disease information and conditions
- Health guidance and recommendations
- Medical questions and clarifications

For non-medical questions, please use a general-purpose AI assistant or search engine.

If you have any health-related symptoms or medical questions, I'm here to help! Please describe your symptoms.""",
                    'stage': 'initial'
                }
            
            # Move to follow-up stage
            self.stage = "followup"
            self.followup_count = 1
            
            # Generate first follow-up question
            question = self.followup_manager.generate_followup_question(
                self.conversation_history, 
                self.followup_count
            )
            self.add_assistant_message(question)
            
            return {
                'type': 'followup_question',
                'content': question,
                'question_num': self.followup_count,
                'total_questions': self.num_followups,
                'stage': 'followup'
            }
        
        # Follow-up Q&A stage
        elif self.stage == "followup":
            self.followup_count += 1
            
            # Check if we've asked enough questions
            if self.followup_count > self.num_followups:
                # Move to diagnosis
                self.stage = "diagnosis"
                return self._generate_diagnosis()
            else:
                # Generate next follow-up question
                question = self.followup_manager.generate_followup_question(
                    self.conversation_history,
                    self.followup_count
                )
                self.add_assistant_message(question)
                
                return {
                    'type': 'followup_question',
                    'content': question,
                    'question_num': self.followup_count,
                    'total_questions': self.num_followups,
                    'stage': 'followup'
                }
        
        # After diagnosis
        elif self.stage == "complete":
            return {
                'type': 'complete',
                'content': "Thank you for using the symptom checker. If you have new symptoms, please start a new session.",
                'stage': 'complete'
            }
    
    
    def _generate_diagnosis(self) -> Dict:
        """Generate final diagnosis after all follow-ups."""
        print(f"\n{'='*70}")
        print("ALL FOLLOW-UPS COMPLETE - GENERATING DIAGNOSIS")
        print(f"{'='*70}")
        
        # Combine all user inputs (symptoms + answers)
        all_symptoms = " | ".join([
            msg['content'] 
            for msg in self.conversation_history 
            if msg['role'] == 'user'
        ])
        
        print(f"📝 Combined patient information:")
        print(f"   {all_symptoms[:200]}...")
        
        # Generate diagnosis using RAG
        result = self.pipeline.generate_diagnosis(all_symptoms, use_rag=True)
        
        self.add_assistant_message(result['diagnosis'])
        self.stage = "complete"
        
        return {
            'type': 'diagnosis',
            'content': result['diagnosis'],
            'sources': result['sources'],
            'used_rag': result.get('used_rag', False),
            'reason': result.get('reason', ''),
            'best_score': result.get('best_score'),
            'stage': 'complete'
        }

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    store_dir = project_root / 'store'
    
    print("="*70)
    print("TESTING RAG PIPELINE WITH ENHANCED FEATURES")
    print("="*70)
    
    # Initialize
    pipeline = RAGPipeline(store_dir)
    
    # Test cases
    test_cases = [
        "I have a fever, headache, and body aches for 3 days",
        "What is the use of a pen?",
        "Tell me a joke",
        "I have chest pain and shortness of breath"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test_input}")
        print(f"{'='*70}")
        
        manager = ConversationManager(pipeline)
        response = manager.process_message(test_input)
        
        print(f"\n🤖 RESPONSE:")
        print(f"{'-'*70}")
        print(f"{response['content'][:500]}...")
        
        if response.get('sources'):
            print(f"\n📚 Sources Used:")
            for source in response['sources'][:3]:
                print(f"   - [{source['source_type']}] {source['title']}")
        
        print(f"\n📊 Metadata:")
        print(f"   Used RAG: {response.get('used_rag', False)}")
        print(f"   Reason: {response.get('reason', 'N/A')}")
        if response.get('best_score'):
            print(f"   Best match score: {response['best_score']:.3f}")