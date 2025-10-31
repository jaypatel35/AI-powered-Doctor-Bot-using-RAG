import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from rag.rag_pipeline import RAGPipeline, ConversationManagerWithFollowUps

def test_followup_flow():
    """Test the complete follow-up question flow."""
    
    project_root = Path(__file__).parent
    store_dir = project_root / 'store'
    
    print("="*70)
    print("TESTING COMPLETE FOLLOW-UP FLOW")
    print("="*70)
    
    # Initialize
    print("\n🚀 Initializing system...")
    pipeline = RAGPipeline(store_dir)
    manager = ConversationManagerWithFollowUps(pipeline, num_followups=3)
    
    print("\n" + "="*70)
    print("SIMULATION: Patient Consultation with Follow-Ups")
    print("="*70)
    
    # Step 1: Initial symptoms
    print("\n👤 Patient: I have chest pain and shortness of breath")
    response1 = manager.process_message("I have chest pain and shortness of breath")
    
    print(f"\n📊 Response Type: {response1['type']}")
    print(f"📊 Stage: {response1['stage']}")
    
    if response1['type'] == 'followup_question':
        print(f"\n🩺 DoctorBot [Question {response1['question_num']}/{response1['total_questions']}]:")
        print(f"   {response1['content']}")
        
        # Step 2: Answer question 1
        answer1 = "It happens during physical activity"
        print(f"\n👤 Patient: {answer1}")
        response2 = manager.process_message(answer1)
        
        if response2['type'] == 'followup_question':
            print(f"\n🩺 DoctorBot [Question {response2['question_num']}/{response2['total_questions']}]:")
            print(f"   {response2['content']}")
            
            # Step 3: Answer question 2
            answer2 = "Yes, I have mild swelling in my ankles"
            print(f"\n👤 Patient: {answer2}")
            response3 = manager.process_message(answer2)
            
            if response3['type'] == 'followup_question':
                print(f"\n🩺 DoctorBot [Question {response3['question_num']}/{response3['total_questions']}]:")
                print(f"   {response3['content']}")
                
                # Step 4: Answer question 3
                answer3 = "No weight gain, but I've been feeling very tired"
                print(f"\n👤 Patient: {answer3}")
                response4 = manager.process_message(answer3)
                
                # Should now get diagnosis
                if response4['type'] == 'diagnosis':
                    print(f"\n{'='*70}")
                    print("🩺 FINAL ASSESSMENT")
                    print(f"{'='*70}")
                    print(f"\n{response4['content'][:800]}...")
                    
                    print(f"\n{'='*70}")
                    print("📊 METADATA")
                    print(f"{'='*70}")
                    print(f"✓ Used RAG: {response4.get('used_rag', False)}")
                    print(f"✓ Reason: {response4.get('reason', 'N/A')}")
                    print(f"✓ Sources: {len(response4.get('sources', []))} documents")
                    if response4.get('best_score'):
                        print(f"✓ Match Score: {response4['best_score']:.3f}")
                    
                    print(f"\n📚 Source Breakdown:")
                    for i, source in enumerate(response4.get('sources', [])[:3], 1):
                        print(f"   {i}. [{source['source_type']}] {source['title'][:50]}...")
                else:
                    print(f"\n❌ Expected diagnosis but got: {response4['type']}")
            else:
                print(f"\n❌ Expected question 3 but got: {response3['type']}")
        else:
            print(f"\n❌ Expected question 2 but got: {response2['type']}")
    else:
        print(f"\n❌ Expected follow-up question but got: {response1['type']}")
    
    print(f"\n{'='*70}")
    print("✅ TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_followup_flow()