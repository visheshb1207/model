import spacy
import textstat
import math
from sentence_transformers import SentenceTransformer, util

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class AssumptionDetector:
    def __init__(self, similarity_threshold=0.7, readability_threshold=30):
        self.similarity_threshold = similarity_threshold
        self.readability_threshold = readability_threshold

        self.assumption_cues = [
            "everyone knows", "obviously", "clearly", "of course",
            "without a doubt", "as we all know", "needless to say",
            "in general", "it goes without saying"
        ]
        self.vague_words = [
            "maybe", "might", "possibly", "could be", "seems",
            "likely", "some say", "unclear", "allegedly"
        ]

    def check_similarity(self, context, answer):
        embeddings = embedder.encode([context, answer], convert_to_tensor=True)
        return util.cos_sim(embeddings[0], embeddings[1]).item()

    def check_new_entities(self, context, answer):
        context_ents = {ent.text for ent in nlp(context).ents}
        answer_ents = {ent.text for ent in nlp(answer).ents}
        return answer_ents - context_ents

    def check_assumption_cues(self, answer):
        cues = []
        for phrase in self.assumption_cues:
            if phrase in answer.lower():
                cues.append(f"Assumption cue: '{phrase}'")
        return cues

    def check_vague_language(self, answer):
        return any(word in answer.lower() for word in self.vague_words)

    def check_readability(self, answer):
        try:
            return textstat.flesch_reading_ease(answer)
        except Exception:
            return 50

    def detect(self, context, answer):
        reasons = []

        similarity = self.check_similarity(context, answer)
        new_ents = self.check_new_entities(context, answer)

        # Unsupported factual claims
        if new_ents:
            reasons.append((f"Introduces new unsupported entities: {', '.join(new_ents)}", 2.0))

        # Low similarity
        if similarity < self.similarity_threshold:
            reasons.append(("Low grounding in context", 1.5))

        # Assumption cues
        for cue in self.check_assumption_cues(answer):
            reasons.append((cue, 1.0))

        # Vague language
        if self.check_vague_language(answer):
            reasons.append(("Contains vague/uncertain language", 1.0))

        readability = self.check_readability(answer)
        if readability < self.readability_threshold:
            reasons.append((f"Low readability: {readability:.2f}", 0.5))

        total_score = sum(weight for _, weight in reasons)

        # --- NEW CONFIDENCE CALCULATION (sigmoid mapping) ---
        confidence = 1 / (1 + math.exp(-1.0 * (total_score - 2.0)))
        # ---------------------------------------------------

        assumed = total_score >= 1.5

        return {
            "assumed": assumed,
            "confidence": round(confidence, 3),
            "total_score": total_score,
            "reasons": reasons,
            "similarity": similarity,
            "readability": readability,
            "new_entities": list(new_ents),
        }