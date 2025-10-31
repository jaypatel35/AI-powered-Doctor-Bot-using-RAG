import faiss
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class MedlineRetriever:
    """Retrieves relevant medical information from FAISS index."""
    
    def __init__(self, store_dir: Path):
        """Load FAISS index, embeddings, and metadata."""
        print("🔄 Loading retriever components...")
        
        # Load FAISS index
        index_path = store_dir / 'faiss_index.bin'
        self.index = faiss.read_index(str(index_path))
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_path = store_dir / 'chunks_metadata.pkl'
        self.chunks_df = pd.read_pickle(metadata_path)
        print(f"✓ Loaded metadata for {len(self.chunks_df)} chunks")
        
        # Load embedding model
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print(f"✓ Loaded embedding model")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: User's symptom description or question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of dicts with chunk info and relevance scores
        """
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            chunk_info = self.chunks_df.iloc[idx]
            results.append({
                'rank': i + 1,
                'score': float(dist),  # Lower is better (L2 distance)
                'title': chunk_info['title'],
                'text': chunk_info['chunk_text'],
                'url': chunk_info['url'],
                'chunk_id': chunk_info['chunk_id'],
                'source_type': chunk_info.get('source_type', 'unknown')  # NOW INCLUDED
            })
        
        return results
    
    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved chunks as context for LLM."""
        context_parts = []
        for result in results:
            source_label = "Textbook" if result['source_type'] == 'textbook' else "MedlinePlus"
            context_parts.append(
                f"[Source: {source_label} - {result['title']}]\n{result['text']}\n"
            )
        return "\n---\n".join(context_parts)


if __name__ == "__main__":
    # Test retriever
    project_root = Path(__file__).parent.parent
    store_dir = project_root / 'store'
    
    retriever = MedlineRetriever(store_dir)
    
    # Test query
    test_query = "I have chest pain and shortness of breath"
    print(f"\n🔍 Test Query: '{test_query}'")
    print("="*60)
    
    results = retriever.retrieve(test_query, top_k=5)
    
    for result in results:
        source_emoji = "📖" if result['source_type'] == 'textbook' else "📚"
        print(f"\n{source_emoji} #{result['rank']} - {result['source_type'].upper()}")
        print(f"   {result['title']} (score: {result['score']:.3f})")
        print(f"   {result['text'][:200]}...")
    
    print("\n" + "="*60)
    print("📄 Formatted Context:")
    print("="*60)
    print(retriever.format_context(results[:3]))