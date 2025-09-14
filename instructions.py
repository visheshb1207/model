import re
from sentence_transformers import SentenceTransformer, util

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Canonical requirements with descriptions + regex
requirements = {
    "context_only": {
        "desc": "Answer using only the provided context without external knowledge.",
        "regex": r"(only use|refer only to|without outside|no external|solely|based only on|provided context)"
    },
    "formatting_bullets": {
        "desc": "Respond in bullet point format.",
        "regex": r"(bullet|list|use dashes|•)"
    },
    "formatting_paragraph": {
        "desc": "Respond in paragraph format without lists.",
        "regex": r"(paragraph|continuous text|avoid bullets|no list)"
    },
    "tone_professional": {
        "desc": "Use a professional and formal tone.",
        "regex": r"(professional|formal tone|business-like)"
    },
    "tone_casual": {
        "desc": "Use a casual, conversational tone.",
        "regex": r"(casual|friendly|informal tone)"
    },
    "security_medical": {
        "desc": "For medical questions, provide disclaimers and avoid direct medical advice.",
        "regex": r"(medical|doctor|health|treatment|diagnosis)"
    },
    "personification_teacher": {
        "desc": "Respond as if you are a teacher explaining concepts.",
        "regex": r"(you are a teacher|act as a teacher|explain like a teacher)"
    },
    "personification_pm": {
        "desc": "Respond as if you are the Prime Minister of the USA.",
        "regex": r"(prime minister|PM of USA|act as a leader)"
    },
    "summarization": {
        "desc": "Summarize the given content clearly.",
        "regex": r"(summarize|in summary|short version|brief)"
    }
}

# Hybrid extractor
def extract_requirements(system_instruction, threshold=0.7):
    asked = []
    instr_embedding = embedder.encode(system_instruction, convert_to_tensor=True)

    for req, data in requirements.items():
        desc_embedding = embedder.encode(data["desc"], convert_to_tensor=True)
        similarity = util.cos_sim(instr_embedding, desc_embedding).item()

        if similarity >= threshold or re.search(data["regex"], system_instruction, re.IGNORECASE):
            asked.append(req)

    return asked

# Evaluator
def evaluate_instruction_following(system_instruction, response):
    asked_requirements = extract_requirements(system_instruction)
    details = {}

    for req in asked_requirements:
        if req == "formatting_bullets":
            details[req] = bool(re.search(r"[-•]\s|\n\d+\.", response))
        elif req == "formatting_paragraph":
            details[req] = not bool(re.search(r"[-•]\s|\n\d+\.", response))
        elif req == "tone_professional":
            details[req] = not bool(re.search(r"\b(dude|hey|lol|gonna|wanna)\b", response.lower()))
        elif req == "tone_casual":
            details[req] = bool(re.search(r"\b(dude|hey|lol|gonna|wanna)\b", response.lower()))
        elif req == "context_only":
            details[req] = not bool(re.search(r"\b(according to|wikipedia|google|external|outside)\b", response.lower()))
        elif req == "security_medical":
            details[req] = "disclaimer" in response.lower()
        else:
            details[req] = True  # placeholder (improve with NLP rules later)

    fulfilled = sum(1 for v in details.values() if v)
    total_asked = len(asked_requirements)
    overall_score = fulfilled / total_asked if total_asked > 0 else 1.0

    return {
        "overall_score": overall_score,
        "details": details,
        "asked_requirements": asked_requirements,
        "fulfilled": fulfilled,
        "total_asked": total_asked
    }