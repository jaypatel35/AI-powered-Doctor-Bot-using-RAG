import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import MedlineRetriever

def test_retrieval():
    """Test retrieval with various symptom queries."""
    
    project_root = Path(__file__).parent.parent
    store_dir = project_root / 'store'
    
    print("="*70)
    print("TESTING UNIFIED RETRIEVAL SYSTEM")
    print("="*70)
    
    # Initialize retriever
    print("\n🔄 Loading retriever...")
    retriever = MedlineRetriever(store_dir)
    
    # Test queries
    test_queries = [
        "I have chest pain and shortness of breath",
        "fever and headache for 3 days",
        "abdominal pain and nausea",
        "dizziness and fatigue",
        "cough and sore throat"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST QUERY {i}: {query}")
        print(f"{'='*70}")
        
        # Retrieve top 5 results
        results = retriever.retrieve(query, top_k=5)
        
        # Show results
        for result in results:
            source_emoji = "📖" if result['source_type'] == 'textbook' else "📚"
            print(f"\n{source_emoji} Rank #{result['rank']} - {result['source_type'].upper()}")
            print(f"   Title: {result['title']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:200]}...")
        
        # Count sources
        textbook_count = sum(1 for r in results if r['source_type'] == 'textbook')
        medline_count = sum(1 for r in results if r['source_type'] == 'medlineplus')
        
        print(f"\n📊 Source distribution:")
        print(f"   Textbook: {textbook_count}/5")
        print(f"   MedlinePlus: {medline_count}/5")
    
    print(f"\n{'='*70}")
    print("✅ RETRIEVAL TEST COMPLETE!")
    print(f"{'='*70}")
    print("\nYour system is now retrieving from BOTH sources!")
    print("✨ Ready for the next step: Enhanced prompts and UI")


if __name__ == "__main__":
    test_retrieval()