import fitz  # PyMuPDF
from pathlib import Path
import json
import re

def extract_all_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract ALL text from the textbook as one continuous document.
    We'll chunk it intelligently later.
    """
    print(f"📖 Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    total_pages = len(doc)
    print(f"📄 Total pages: {total_pages}")
    
    all_text = []
    
    for page_num in range(total_pages):
        if page_num % 50 == 0:
            print(f"   Extracting page {page_num + 1}/{total_pages}...")
        
        page = doc[page_num]
        text = page.get_text()
        
        # Skip completely empty pages
        if text.strip():
            all_text.append(text)
    
    full_text = '\n\n'.join(all_text)
    doc.close()
    
    print(f"✓ Extracted {len(full_text):,} characters from {total_pages} pages")
    
    return full_text


def clean_textbook_text(text: str) -> str:
    """Clean extracted text."""
    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Fix common OCR errors
    text = text.replace('w ', '')  # Remove stray 'w' characters
    text = text.replace('I ', 'I ')
    text = text.replace('l ', 'l ')
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove headers/footers with book title
    text = re.sub(r'Symptom to Diagnosis.*?Edition', '', text, flags=re.IGNORECASE)
    
    # Fix bullet points
    text = re.sub(r'^\s*[•●○]\s*', '- ', text, flags=re.MULTILINE)
    
    # Remove non-ASCII but keep medical symbols
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()


def main():
    pdf_path = Path('data/symptom_to_diagnosis.pdf')
    output_path = Path('data/textbook_extracted.json')
    
    output_path.parent.mkdir(exist_ok=True)
    
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF not found at {pdf_path}")
        return
    
    print("="*70)
    print("EXTRACTING 'SYMPTOM TO DIAGNOSIS' TEXTBOOK")
    print("="*70)
    
    # Extract all text
    full_text = extract_all_text_from_pdf(pdf_path)
    
    # Clean text
    print(f"\n{'='*70}")
    print(f"🧹 CLEANING EXTRACTED TEXT")
    print(f"{'='*70}")
    cleaned_text = clean_textbook_text(full_text)
    print(f"✓ Cleaned text: {len(cleaned_text):,} characters")
    
    # Save as JSON with single entry
    print(f"\n{'='*70}")
    print(f"💾 SAVING TO JSON")
    print(f"{'='*70}")
    
    data = {
        "Symptom to Diagnosis - Full Textbook": cleaned_text
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"{'='*70}")
    print(f"   Total characters: {len(cleaned_text):,}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Saved to: {output_path}")
    
    # Show sample
    print(f"\n{'='*70}")
    print(f"📋 SAMPLE TEXT (First 1000 characters)")
    print(f"{'='*70}")
    print(f"\n{cleaned_text[:1000]}")
    print(f"\n... (total {len(cleaned_text):,} characters)")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()