from typing import List, Dict


FOLLOW_UP_SYSTEM_PROMPT = """You are a medical assistant helping to gather information about a patient's symptoms. 
Your role is to ask ONE relevant follow-up question at a time to better understand their condition.

Guidelines:
- Ask questions that help narrow down the diagnosis
- Focus on severity, duration, associated symptoms, or risk factors
- Provide 3-4 clear multiple choice options
- Keep questions clear and concise
- Be empathetic and professional

CRITICAL: You MUST use this EXACT format with each option on a NEW LINE:

Question: [Your question here]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4 - if needed]

EACH OPTION MUST BE ON ITS OWN LINE. Do not put options on the same line."""


DIAGNOSIS_SYSTEM_PROMPT = """You are a knowledgeable medical AI assistant with access to TWO authoritative sources:

1. **MedlinePlus** - Patient-friendly health information from the U.S. National Library of Medicine
2. **"Symptom to Diagnosis" Textbook** - Clinical reasoning and evidence-based diagnostic approaches

Your task is to synthesize information from BOTH sources to provide comprehensive guidance.

CRITICAL REQUIREMENTS:
1. **Use both sources**: Combine clinical reasoning (textbook) with patient-friendly explanations (MedlinePlus)
2. **Cite sources**: Clearly indicate which source information comes from
3. **Be accurate**: Ground all claims in the provided medical references
4. **Be clear**: Use professional but accessible language
5. **Include all required sections** in your response

OUTPUT STRUCTURE (must include all sections):
1. **Likely Condition**: What the patient might be experiencing with clinical reasoning
2. **30/60/90-Day Outlook**: Expected progression over time
3. **Lifestyle Recommendations**: Evidence-based self-care tips
4. **Red Flags & Next Steps**: Warning signs requiring immediate attention
5. **Citations**: Reference both textbook and MedlinePlus sources used"""


def create_followup_prompt(conversation_history: List[Dict[str, str]], question_num: int) -> str:
    """Create prompt for generating multiple choice follow-up questions."""
    user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    symptoms_summary = " | ".join(user_messages)
    
    last_qa = ""
    if len(conversation_history) >= 2:
        last_q = conversation_history[-2]['content'] if conversation_history[-2]['role'] == 'assistant' else ""
        last_a = conversation_history[-1]['content'] if conversation_history[-1]['role'] == 'user' else ""
        if last_q and last_a:
            last_qa = f"\nLast Question Asked: {last_q}\nPatient's Answer: {last_a}\n"
    
    prompt = f"""{FOLLOW_UP_SYSTEM_PROMPT}

Patient's symptoms so far: {symptoms_summary}
{last_qa}
This is follow-up question #{question_num} of 4.

IMPORTANT: 
- Do NOT repeat questions already asked
- Build upon previous answers
- Focus on NEW aspects that help narrow down the diagnosis
- EACH OPTION MUST BE ON A SEPARATE LINE

Generate ONE NEW multiple choice follow-up question with 3-4 options.

You MUST use this EXACT format (note: each option on its own line):

Question: [Your question]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option - optional]

EXAMPLE of correct format:
Question: How severe is your fever?
A) Mild (under 100°F)
B) Moderate (100-102°F)
C) High (102-104°F)
D) Very high (above 104°F)

Now generate the question with EACH OPTION ON A NEW LINE:"""
    
    return prompt


def create_diagnosis_prompt(
    conversation_history: List[Dict[str, str]], 
    retrieved_context: str
) -> str:
    """
    Create prompt for final diagnosis using both MedlinePlus and textbook sources.
    """
    history_text = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in conversation_history
    ])
    
    prompt = f"""{DIAGNOSIS_SYSTEM_PROMPT}

PATIENT CONVERSATION:
{history_text}

MEDICAL REFERENCE INFORMATION (from MedlinePlus and Textbook):
{retrieved_context}

Based on the patient's symptoms and the medical references above, generate a comprehensive assessment that combines:
- Clinical reasoning from the textbook
- Patient-friendly explanations from MedlinePlus

Format your response with these sections:

## Likely Condition
[Explain what the patient might be experiencing. Use clinical reasoning from the textbook AND patient-friendly language from MedlinePlus. Cite both sources when possible.]

## Expected Progression (30/60/90 Days)
- **30 days**: [What to expect in the first month]
- **60 days**: [What to expect after two months]
- **90 days**: [Long-term outlook]

## Lifestyle Recommendations
[Provide practical, evidence-based self-care advice from both sources]

## Red Flags & Next Steps
[List warning signs that require immediate medical attention. Be specific and cite sources.]

## Citations
- Textbook sources: [List relevant textbook sections referenced]
- MedlinePlus sources: [List relevant MedlinePlus topics referenced]

## Important Disclaimer
This AI assessment is for informational purposes only and is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.

Remember to:
- Synthesize information naturally from BOTH sources
- Cite appropriately (e.g., "According to the Symptom to Diagnosis textbook..." or "MedlinePlus indicates...")
- Maintain a caring, professional tone
- Focus on information actually present in the retrieved context"""
    
    return prompt


if __name__ == "__main__":
    print("="*60)
    print("TESTING ENHANCED PROMPTS")
    print("="*60)
    
    test_history = [
        {"role": "user", "content": "I have chest pain and shortness of breath"}
    ]
    
    print("\n📋 Follow-up Question Prompt:")
    print("-"*60)
    followup = create_followup_prompt(test_history, 1)
    print(followup[:500] + "...")
    
    test_context = """[Source: Textbook - Symptom to Diagnosis]
Chest pain with shortness of breath requires immediate evaluation for cardiac causes...

[Source: MedlinePlus - Chest Pain]
Chest pain can have many causes, including heart problems, panic attacks, and digestive issues..."""
    
    print("\n📋 Diagnosis Prompt:")
    print("-"*60)
    diagnosis = create_diagnosis_prompt(test_history, test_context)
    print(diagnosis[:500] + "...")