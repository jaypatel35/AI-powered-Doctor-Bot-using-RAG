import faiss
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import streamlit as st


class MedlineRetriever:
    """Retrieves relevant medical information from FAISS index."""
    
    def __init__(self, store_dir: Path, data_dir: Path = None):
        """
        Load or build FAISS index, embeddings, and metadata.
        
        Args:
            store_dir: Directory where FAISS index will be stored/loaded
            data_dir: Directory containing raw Medline data (for building index)
        """
        self.store_dir = Path(store_dir)
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        
        # Ensure store directory exists
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # Load embedding model first
        print("🔄 Loading embedding model...")
        self.model = self._load_embedding_model()
        print(f"✓ Loaded embedding model")
        
        # Load or build index
        self.index, self.chunks_df = self._load_or_build_index()
    
    @st.cache_resource
    def _load_embedding_model(_self):
        """Load embedding model (cached)"""
        return SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    def _index_exists(self) -> bool:
        """Check if FAISS index and metadata exist"""
        index_path = self.store_dir / 'faiss_index.bin'
        metadata_path = self.store_dir / 'chunks_metadata.pkl'
        embeddings_path = self.store_dir / 'embeddings.npy'
        config_path = self.store_dir / 'config.pkl'
        
        return all([
            index_path.exists(), 
            metadata_path.exists(),
            embeddings_path.exists(),
            config_path.exists()
        ])
    
    def _load_existing_index(self):
        """Load existing FAISS index and metadata"""
        print("🔄 Loading existing FAISS index...")
        
        index_path = self.store_dir / 'faiss_index.bin'
        metadata_path = self.store_dir / 'chunks_metadata.pkl'
        embeddings_path = self.store_dir / 'embeddings.npy'
        config_path = self.store_dir / 'config.pkl'
        
        # Check if all required files exist
        missing_files = []
        if not index_path.exists():
            missing_files.append('faiss_index.bin')
        if not metadata_path.exists():
            missing_files.append('chunks_metadata.pkl')
        if not embeddings_path.exists():
            missing_files.append('embeddings.npy')
        if not config_path.exists():
            missing_files.append('config.pkl')
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {self.store_dir}: {', '.join(missing_files)}\n"
                f"Please ensure the store/ directory contains all required files for deployment."
            )
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        print(f"✓ Loaded FAISS index with {index.ntotal} vectors")
        
        # Load metadata
        chunks_df = pd.read_pickle(metadata_path)
        print(f"✓ Loaded metadata for {len(chunks_df)} chunks")
        
        return index, chunks_df
    
    def _build_index_from_data(self):
        """Build FAISS index from raw Medline data"""
        print("🔨 Building FAISS index from raw data...")
        
        # Look for CSV file in data directory
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                "Please add your Medline dataset (CSV format) to the data/ folder."
            )
        
        # Use first CSV file found
        data_file = csv_files[0]
        print(f"📂 Loading data from: {data_file}")
        
        try:
            df = pd.read_csv(data_file)
            print(f"✓ Loaded {len(df)} rows from dataset")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
        # Prepare chunks (adjust column names based on your CSV structure)
        chunks_data = self._prepare_chunks(df)
        
        # Generate embeddings
        print(f"🔄 Generating embeddings for {len(chunks_data)} chunks...")
        texts = [chunk['chunk_text'] for chunk in chunks_data]
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        print(f"✓ Generated embeddings with shape {embeddings.shape}")
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        print(f"✓ Built FAISS index with {index.ntotal} vectors")
        
        # Create chunks dataframe
        chunks_df = pd.DataFrame(chunks_data)
        
        # Save index and metadata
        self._save_index(index, chunks_df)
        
        return index, chunks_df
    
    def _prepare_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare chunks from raw dataframe.
        Adjust this method based on your CSV structure.
        """
        chunks = []
        
        # Detect column structure
        # Common variations: 'disease', 'illness', 'title', 'condition'
        # Common variations: 'symptoms', 'description', 'text', 'content'
        
        title_col = None
        text_col = None
        
        # Try to detect title column
        for col in ['title', 'disease', 'illness', 'condition', 'topic']:
            if col in df.columns:
                title_col = col
                break
        
        # Try to detect text column
        for col in ['symptoms', 'description', 'text', 'content', 'summary']:
            if col in df.columns:
                text_col = col
                break
        
        if not title_col or not text_col:
            # Fallback: use first two columns
            title_col = df.columns[0]
            text_col = df.columns[1]
            print(f"⚠️ Using columns: '{title_col}' and '{text_col}'")
        
        print(f"📋 Using title column: '{title_col}', text column: '{text_col}'")
        
        for idx, row in df.iterrows():
            title = str(row[title_col])
            text = str(row[text_col])
            
            # Create chunk
            chunk = {
                'chunk_id': f"chunk_{idx}",
                'title': title,
                'chunk_text': text,
                'url': row.get('url', f"#item_{idx}"),
                'source_type': row.get('source_type', 'medlineplus')
            }
            chunks.append(chunk)
        
        return chunks
    
    def _save_index(self, index, chunks_df):
        """Save FAISS index and metadata"""
        index_path = self.store_dir / 'faiss_index.bin'
        metadata_path = self.store_dir / 'chunks_metadata.pkl'
        
        faiss.write_index(index, str(index_path))
        chunks_df.to_pickle(metadata_path)
        
        print(f"💾 Saved FAISS index to: {index_path}")
        print(f"💾 Saved metadata to: {metadata_path}")
    
    def _load_or_build_index(self):
        """Load existing index or build new one"""
        if self._index_exists():
            try:
                return self._load_existing_index()
            except Exception as e:
                print(f"⚠️ Error loading existing index: {e}")
                print("🔨 Building new index...")
                return self._build_index_from_data()
        else:
            return self._build_index_from_data()
    
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
                'source_type': chunk_info.get('source_type', 'unknown')
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
    data_dir = project_root / 'data'
    
    retriever = MedlineRetriever(store_dir, data_dir)
    
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