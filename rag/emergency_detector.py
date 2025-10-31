from typing import Dict, List
import re


class EmergencyDetector:
    """Detects emergency symptoms that require immediate medical attention."""
    
    # Critical emergency keywords
    EMERGENCY_KEYWORDS = {
        'cardiac': [
            'chest pain', 'chest pressure', 'crushing chest', 'heart attack',
            'chest tightness', 'pain radiating to arm', 'pain in left arm',
            'jaw pain with chest', 'severe chest discomfort'
        ],
        'respiratory': [
            'can\'t breathe', 'cannot breathe', 'difficulty breathing',
            'shortness of breath severe', 'gasping for air', 'choking',
            'turning blue', 'blue lips', 'severe breathing difficulty'
        ],
        'neurological': [
            'stroke', 'face drooping', 'arm weakness', 'speech difficulty',
            'sudden confusion', 'severe headache worst ever', 'thunderclap headache',
            'loss of consciousness', 'passed out', 'seizure', 'convulsion',
            'sudden weakness', 'sudden numbness', 'can\'t move arm', 'can\'t move leg'
        ],
        'bleeding': [
            'severe bleeding', 'heavy bleeding', 'uncontrolled bleeding',
            'bleeding won\'t stop', 'coughing up blood', 'vomiting blood',
            'blood in vomit', 'blood in stool black'
        ],
        'trauma': [
            'severe injury', 'head injury', 'major trauma', 'car accident',
            'fell from height', 'broken bone protruding', 'severe burn'
        ],
        'severe_pain': [
            'worst pain of my life', 'worst headache ever', '10/10 pain',
            'excruciating pain', 'unbearable pain', 'severe abdominal pain sudden'
        ],
        'altered_mental': [
            'very confused', 'disoriented', 'hallucinating', 'can\'t wake up',
            'unresponsive', 'not making sense'
        ],
        'allergic': [
            'throat swelling', 'tongue swelling', 'severe allergic reaction',
            'anaphylaxis', 'epipen needed', 'allergic reaction severe',
            'face swelling rapidly', 'hives with breathing difficulty'
        ]
    }
    
    def detect(self, text: str) -> Dict:
        """
        Detect if symptoms indicate an emergency.
        
        Args:
            text: Patient's symptom description
        
        Returns:
            Dict with 'is_emergency', 'category', 'matched_keywords', 'severity'
        """
        text_lower = text.lower()
        
        detected_categories = []
        matched_keywords = []
        
        # Check each category
        for category, keywords in self.EMERGENCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_categories.append(category)
                    matched_keywords.append(keyword)
                    break  # One match per category is enough
        
        is_emergency = len(detected_categories) > 0
        
        # Determine severity
        severity = 'CRITICAL' if len(detected_categories) >= 2 else 'HIGH' if is_emergency else 'NONE'
        
        return {
            'is_emergency': is_emergency,
            'categories': detected_categories,
            'matched_keywords': matched_keywords,
            'severity': severity,
            'message': self._get_emergency_message(detected_categories) if is_emergency else None
        }
    
    def _get_emergency_message(self, categories: List[str]) -> str:
        """Generate appropriate emergency message based on detected categories."""
        
        category_messages = {
            'cardiac': '**CARDIAC EMERGENCY** - Possible heart attack',
            'respiratory': '**RESPIRATORY EMERGENCY** - Severe breathing difficulty',
            'neurological': '**NEUROLOGICAL EMERGENCY** - Possible stroke or brain injury',
            'bleeding': '**SEVERE BLEEDING** - Immediate medical attention needed',
            'trauma': '**MAJOR TRAUMA** - Serious injury',
            'severe_pain': '**SEVERE PAIN** - Immediate evaluation needed',
            'altered_mental': '**ALTERED CONSCIOUSNESS** - Immediate medical attention',
            'allergic': '**SEVERE ALLERGIC REACTION** - Possible anaphylaxis'
        }
        
        messages = [category_messages.get(cat, cat.upper()) for cat in categories]
        
        emergency_response = f"""
🚨 **EMERGENCY DETECTED** 🚨

{' | '.join(messages)}

**IMMEDIATE ACTION REQUIRED:**

🆘 **Go to the nearest Emergency Room immediately

⚠️ **DO NOT WAIT** - These symptoms require immediate medical attention

**While waiting for emergency services:**
- Stay calm and try to remain seated or lying down
- Do not drive yourself - wait for ambulance
- Have someone stay with you
- If you have prescribed emergency medication (like nitroglycerin or EpiPen), use it as directed

**Critical symptoms detected:** {', '.join(categories)}

---

*This AI cannot replace emergency medical services. Your symptoms indicate a potentially life-threatening condition that requires immediate professional medical care.*
"""
        
        return emergency_response


if __name__ == "__main__":
    # Test emergency detection
    print("="*70)
    print("TESTING EMERGENCY SYMPTOM DETECTION")
    print("="*70)
    
    detector = EmergencyDetector()
    
    test_cases = [
        {
            "input": "I have chest pain radiating to my left arm",
            "expected": "Emergency (cardiac)"
        },
        {
            "input": "I have a mild headache",
            "expected": "Not emergency"
        },
        {
            "input": "I can't breathe and my chest hurts badly",
            "expected": "Critical emergency (cardiac + respiratory)"
        },
        {
            "input": "I think I'm having a stroke - face drooping",
            "expected": "Emergency (neurological)"
        },
        {
            "input": "Severe bleeding that won't stop",
            "expected": "Emergency (bleeding)"
        },
        {
            "input": "I have a fever of 100°F",
            "expected": "Not emergency"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test['expected']}")
        print(f"{'='*70}")
        print(f"Input: {test['input']}")
        
        result = detector.detect(test['input'])
        
        print(f"\n📊 Detection Results:")
        print(f"   Emergency: {result['is_emergency']}")
        print(f"   Severity: {result['severity']}")
        print(f"   Categories: {result['categories']}")
        print(f"   Keywords: {result['matched_keywords']}")
        
        if result['is_emergency']:
            print(f"\n{result['message']}")
    
    print(f"\n{'='*70}")
    print("✅ EMERGENCY DETECTION TEST COMPLETE")
    print(f"{'='*70}")