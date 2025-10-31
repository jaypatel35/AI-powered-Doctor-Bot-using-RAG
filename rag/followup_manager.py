import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.prompts import create_followup_prompt

load_dotenv()
class FollowUpManager:
    """Manages follow-up questions for symptom screening."""
    
    def __init__(self):
        """Initialize with OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = OpenAI(api_key=api_key)
    
    def generate_followup_question(
        self, 
        conversation_history: List[Dict[str, str]], 
        question_num: int
    ) -> str:
        """
        Generate a single follow-up question.
        
        Args:
            conversation_history: List of messages so far
            question_num: Which question number (1-4)
        
        Returns:
            Generated question text
        """
        print(f"🤔 Generating follow-up question #{question_num}...")
        
        prompt = create_followup_prompt(conversation_history, question_num)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant asking diagnostic questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        question = response.choices[0].message.content.strip()
        print(f"✓ Generated question #{question_num}")
        
        return question
    
    def parse_question_options(self, question_text: str) -> Dict:
        """
        Parse the question to extract options if present.
        
        Returns:
            Dict with 'question' and 'options' (if multiple choice)
        """
        lines = question_text.strip().split('\n')
        
        # Find the question line
        question = ""
        options = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line and (line[0] in ['A', 'B', 'C', 'D'] and line[1] in [')', '.', ']']):
                # This is an option
                options.append(line)
        
        return {
            'question': question if question else question_text,
            'options': options,
            'has_options': len(options) > 0
        }


if __name__ == "__main__":
    # Test the follow-up manager
    print("="*70)
    print("TESTING FOLLOW-UP QUESTION GENERATOR")
    print("="*70)
    
    manager = FollowUpManager()
    
    # Test conversation
    test_history = [
        {"role": "user", "content": "I have chest pain and shortness of breath"}
    ]
    
    for i in range(1, 4):
        print(f"\n{'='*70}")
        print(f"GENERATING QUESTION {i}")
        print(f"{'='*70}")
        
        question = manager.generate_followup_question(test_history, i)
        parsed = manager.parse_question_options(question)
        
        print(f"\n📋 Question: {parsed['question']}")
        if parsed['has_options']:
            print(f"\n Options:")
            for opt in parsed['options']:
                print(f"   {opt}")
        
        # Simulate user answer
        if i < 3:
            test_answer = f"Test answer for question {i}"
            test_history.append({"role": "assistant", "content": question})
            test_history.append({"role": "user", "content": test_answer})
            print(f"\n💬 Simulated answer: {test_answer}")
    
    print(f"\n{'='*70}")
    print("✅ Follow-up question generation test complete!")
    print(f"{'='*70}")