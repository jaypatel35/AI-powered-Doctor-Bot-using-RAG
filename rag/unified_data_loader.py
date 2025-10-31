import json
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import html
import re

def parse_medlineplus_xml(xml_path: Path) -> pd.DataFrame:
    """Parse MedlinePlus XML data (your existing data)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    topics = []
    
    for health_topic in root.findall('health-topic'):
        language = health_topic.get('language', '')
        if language != 'English':
            continue
        
        title = health_topic.get('title', '')
        topic_id = health_topic.get('id', '')
        url = health_topic.get('url', '')
        
        also_called = [ac.text for ac in health_topic.findall('also-called') if ac.text]
        also_called_str = ', '.join(also_called) if also_called else ''
        
        full_summary = health_topic.find('full-summary')
        if full_summary is not None and full_summary.text:
            summary = html.unescape(full_summary.text)
            summary = re.sub(r'<[^>]+>', ' ', summary)
            summary = re.sub(r'\s+', ' ', summary).strip()
        else:
            summary = ''
        
        if not summary or len(summary) < 50:
            continue
        
        topics.append({
            'id': topic_id,
            'title': title,
            'also_called': also_called_str,
            'summary': summary,
            'url': url,
            'source_type': 'medlineplus'
        })
    
    df = pd.DataFrame(topics)
    print(f"✓ Loaded {len(df)} MedlinePlus topics")
    return df


def load_textbook_json(json_path: Path) -> pd.DataFrame:
    """Load the extracted textbook JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # The textbook is one large document
    # I'll create a single entry for now
    textbook_title = list(data.keys())[0]
    textbook_content = data[textbook_title]
    
    records = [{
        'id': 'textbook_full',
        'title': 'Symptom to Diagnosis - Evidence-Based Guide',
        'also_called': '',
        'summary': textbook_content,
        'url': 'textbook://symptom_to_diagnosis',
        'source_type': 'textbook'
    }]
    
    df = pd.DataFrame(records)
    print(f"✓ Loaded textbook ({len(textbook_content):,} characters)")
    return df


def create_unified_dataset():
    """Combine MedlinePlus and textbook into one dataset."""
    project_root = Path(__file__).parent.parent
    
    print("="*70)
    print("CREATING UNIFIED MEDICAL DATASET")
    print("="*70)
    
    # Load MedlinePlus
    print("\n📚 Loading MedlinePlus data...")
    medlineplus_path = project_root / 'data' / 'medplus.xml'
    medlineplus_df = parse_medlineplus_xml(medlineplus_path)
    
    # Load Textbook
    print("\n📖 Loading Symptom to Diagnosis textbook...")
    textbook_path = project_root / 'data' / 'textbook_extracted.json'
    textbook_df = load_textbook_json(textbook_path)
    
    # Combine
    print("\n🔗 Combining datasets...")
    unified_df = pd.concat([medlineplus_df, textbook_df], ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"📊 UNIFIED DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"   Total documents: {len(unified_df)}")
    print(f"   - MedlinePlus topics: {len(medlineplus_df)}")
    print(f"   - Textbook: {len(textbook_df)}")
    print(f"   Total characters: {unified_df['summary'].str.len().sum():,}")
    
    # Save
    output_path = project_root / 'data' / 'unified_medical_data.csv'
    unified_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved unified dataset to: {output_path}")
    print(f"{'='*70}")
    
    return unified_df


if __name__ == "__main__":
    df = create_unified_dataset()
    
    # Show samples
    print("\n📋 Sample entries:")
    print(f"\nMedlinePlus example:")
    medline_sample = df[df['source_type'] == 'medlineplus'].iloc[0]
    print(f"   Title: {medline_sample['title']}")
    print(f"   Content: {medline_sample['summary'][:200]}...")
    
    print(f"\nTextbook example:")
    textbook_sample = df[df['source_type'] == 'textbook'].iloc[0]
    print(f"   Title: {textbook_sample['title']}")
    print(f"   Content: {textbook_sample['summary'][:200]}...")