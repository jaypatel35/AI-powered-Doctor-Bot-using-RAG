import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm

def build_faiss_index(chunks_df: pd.DataFrame, model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Create embeddings and build FAISS index from unified chunks.
    
    Args:
        chunks_df: DataFrame with chunk_text column
        model_name: SentenceTransformer model to use
    
    Returns:
        tuple: (faiss_index, embeddings_array, model)
    """
    print("="*70)
    print("BUILDING FAISS INDEX FROM UNIFIED DATA")
    print("="*70)
    
    print(f"\n🤖 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Get embedding dimension
    sample_embedding = model.encode("test")
    embedding_dim = len(sample_embedding)
    print(f"   ✓ Embedding dimension: {embedding_dim}")
    
    # Encode all chunks with progress bar
    print(f"\n📊 Creating embeddings for {len(chunks_df)} chunks...")
    print(f"   - MedlinePlus chunks: {len(chunks_df[chunks_df['source_type']=='medlineplus'])}")
    print(f"   - Textbook chunks: {len(chunks_df[chunks_df['source_type']=='textbook'])}")
    
    texts = chunks_df['chunk_text'].tolist()
    
    # Batch encode for efficiency
    print(f"\n⚙️  Encoding (this may take a few minutes)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"\n✓ Created embeddings with shape: {embeddings.shape}")
    
    # Build FAISS index
    print(f"\n🔍 Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
    index.add(embeddings.astype('float32'))
    
    print(f"✓ FAISS index built with {index.ntotal} vectors")
    
    return index, embeddings, model


def save_index_and_metadata(index, embeddings, chunks_df, model):
    """Save FAISS index, embeddings, and metadata."""
    store_dir = Path(__file__).parent.parent / 'store'
    store_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"💾 SAVING INDEX AND METADATA")
    print(f"{'='*70}")
    
    # Save FAISS index
    index_path = store_dir / 'faiss_index.bin'
    faiss.write_index(index, str(index_path))
    print(f"✅ Saved FAISS index to: {index_path}")
    
    # Save embeddings
    embeddings_path = store_dir / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"✅ Saved embeddings to: {embeddings_path}")
    
    # Save metadata (chunks DataFrame)
    metadata_path = store_dir / 'chunks_metadata.pkl'
    chunks_df.to_pickle(metadata_path)
    print(f"✅ Saved metadata to: {metadata_path}")
    
    # Save config
    config_path = store_dir / 'config.pkl'
    config = {
        'model_name': 'BAAI/bge-small-en-v1.5',
        'total_chunks': len(chunks_df),
        'medlineplus_chunks': len(chunks_df[chunks_df['source_type']=='medlineplus']),
        'textbook_chunks': len(chunks_df[chunks_df['source_type']=='textbook']),
        'embedding_dim': embeddings.shape[1]
    }
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"✅ Saved config to: {config_path}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    
    print("="*70)
    print("STEP 1: Loading chunks")
    print("="*70)
    
    # Load chunks
    chunks_path = project_root / 'data' / 'chunks_unified.pkl'
    chunks_df = pd.read_pickle(chunks_path)
    print(f"✓ Loaded {len(chunks_df)} chunks")
    
    # Step 2: Build index
    print(f"\n{'='*70}")
    print("STEP 2: Building FAISS index")
    print(f"{'='*70}")
    index, embeddings, model = build_faiss_index(chunks_df)
    
    # Step 3: Save everything
    print(f"\n{'='*70}")
    print("STEP 3: Saving index and metadata")
    print(f"{'='*70}")
    save_index_and_metadata(index, embeddings, chunks_df, model)
    
    print(f"\n{'='*70}")
    print("✨ INDEX BUILDING COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Summary:")
    print(f"   Total chunks indexed: {len(chunks_df)}")
    print(f"   MedlinePlus chunks: {len(chunks_df[chunks_df['source_type']=='medlineplus'])}")
    print(f"   Textbook chunks: {len(chunks_df[chunks_df['source_type']=='textbook'])}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Index size: {index.ntotal} vectors")
    print(f"\n✅ Your RAG system is now ready with both data sources!")
    print(f"{'='*70}")