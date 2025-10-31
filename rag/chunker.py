import pandas as pd
from typing import List
from pathlib import Path
import re

def chunk_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks by words, respecting sentence boundaries.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in words
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    # Split into sentences (basic sentence splitting)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If adding this sentence would exceed chunk_size
        if current_word_count + sentence_word_count > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            # Calculate how many words to overlap
            overlap_words = []
            overlap_count = 0
            for sent in reversed(current_chunk):
                sent_words = sent.split() if isinstance(sent, str) else []
                if overlap_count + len(sent_words) <= overlap:
                    overlap_words.insert(0, sent)
                    overlap_count += len(sent_words)
                else:
                    break
            
            current_chunk = overlap_words + [sentence]
            current_word_count = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def create_chunks_from_unified_data(csv_path: Path) -> pd.DataFrame:
    """
    Create chunks from unified dataset (MedlinePlus + textbook).
    Uses different chunk sizes based on source type.
    
    Returns:
        DataFrame with columns: chunk_id, title, chunk_text, source_id, url, source_type
    """
    print("="*70)
    print("CREATING CHUNKS FROM UNIFIED DATASET")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    all_chunks = []
    chunk_id = 0
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"   Processing document {idx + 1}/{len(df)}...")
        
        title = row['title']
        summary = row['summary']
        source_type = row['source_type']
        also_called = row['also_called'] if pd.notna(row['also_called']) else ''
        
        # Build full text with context
        full_text = f"{title}. "
        if also_called:
            full_text += f"Also known as: {also_called}. "
        full_text += summary
        
        # Different chunking strategy based on source
        if source_type == 'textbook':
            # Larger chunks for detailed clinical content
            chunk_size = 600
            overlap = 150
            print(f"\n📖 Chunking textbook content...")
        else:
            # Smaller chunks for concise MedlinePlus summaries
            chunk_size = 400
            overlap = 50
        
        # Create chunks
        chunks = chunk_text_with_overlap(full_text, chunk_size, overlap)
        
        for chunk_text in chunks:
            all_chunks.append({
                'chunk_id': chunk_id,
                'title': title,
                'chunk_text': chunk_text,
                'source_id': row['id'],
                'url': row['url'],
                'source_type': source_type
            })
            chunk_id += 1
    
    chunks_df = pd.DataFrame(all_chunks)
    
    print(f"\n{'='*70}")
    print(f"📊 CHUNKING SUMMARY")
    print(f"{'='*70}")
    print(f"   Total chunks created: {len(chunks_df)}")
    print(f"   - From MedlinePlus: {len(chunks_df[chunks_df['source_type']=='medlineplus'])}")
    print(f"   - From textbook: {len(chunks_df[chunks_df['source_type']=='textbook'])}")
    print(f"   Average chunk length: {chunks_df['chunk_text'].str.len().mean():.0f} characters")
    
    return chunks_df


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'data' / 'unified_medical_data.csv'
    
    chunks_df = create_chunks_from_unified_data(csv_path)
    
    # Save chunks
    output_path = project_root / 'data' / 'chunks_unified.pkl'
    chunks_df.to_pickle(output_path)
    print(f"\n✅ Saved chunks to: {output_path}")
    
    # Show samples
    print(f"\n{'='*70}")
    print(f"📋 SAMPLE CHUNKS")
    print(f"{'='*70}")
    
    print("\nMedlinePlus chunk example:")
    medline_chunk = chunks_df[chunks_df['source_type']=='medlineplus'].iloc[0]
    print(f"   Title: {medline_chunk['title']}")
    print(f"   Chunk: {medline_chunk['chunk_text'][:300]}...")
    
    print("\nTextbook chunk example:")
    textbook_chunk = chunks_df[chunks_df['source_type']=='textbook'].iloc[0]
    print(f"   Title: {textbook_chunk['title']}")
    print(f"   Chunk: {textbook_chunk['chunk_text'][:300]}...")
    
    print(f"\n{'='*70}")