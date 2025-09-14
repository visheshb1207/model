import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# nltk.download('punkt')


# Download the necessary NLTK data (only need to do this once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


# --- 1. Initialize the Model ---
# This model is excellent for semantic similarity tasks.
# It will be downloaded on the first run.
# print("Loading semantic similarity model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
# print("Model loaded.")


# --- 2. Define the Accuracy Checking Function ---

def check_accuracy(response, reference_document):
    """
    Calculates the accuracy score based on semantic similarity between a response
    and a reference document.
    """
    # Split the response and document into individual sentences
    response_sentences = nltk.word_tokenize(response)
    doc_sentences = nltk.word_tokenize(reference_document)

    if not response_sentences or not doc_sentences:
        return 0.0

    # Convert sentences into vector embeddings
    response_embeddings = model.encode(response_sentences)
    doc_embeddings = model.encode(doc_sentences)

    # Calculate the cosine similarity between each response sentence and all doc sentences
    similarity_matrix = cosine_similarity(response_embeddings, doc_embeddings)

    # For each response sentence, find the highest similarity score it has with any sentence in the document.
    # This finds the "best matching" evidence in the source document for each claim.
    max_similarity_scores = np.max(similarity_matrix, axis=1)

    # The final accuracy score is the average of these best-match scores.
    final_accuracy_score = np.mean(max_similarity_scores)

    return final_accuracy_score








doc = "high blood pressure \n People who have consistent high blood pressure (hypertension) in mid-life (ages 45 to 65) are more likely to develop dementia compared to those with normal blood pressure.\n \n\n High blood pressure can increase the risk of developing dementia, particularly vascular dementia, because of its effect on the heart, the arteries, and blood circulation.\n \n\n Smoking \n The evidence is strong and consistent that smokers are at a higher risk of developing dementia vs. non-smokers or ex-smokers.\n \n\n It’s never too late to quit! Smokers who quit can reduce their risk of developing dementia.\n \n\n diabetes \n People with type 2 diabetes in mid-life (ages 45 to 65) are at an increased risk of developing dementia, particularly Alzheimer’s disease and vascular dementia.\n \n\n Obesity\n Obesity in mid-life (ages 45 to 65) increases the risk of developing dementia. Obesity also increases the risk of developing other risk factors such as type 2 diabetes.\n \n\n lack of physical activity \n Physical inactivity in later life (ages 65 and up) increases the risk of developing dementia.\n \n\n poor diet\n An unhealthy diet, high in saturated fat, sugar, and salt, can increase the risk of developing many illnesses, including dementia and cardiovascular disease.\n \n\n high alcohol consumption \n Drinking excessively (more than 12 drinks per week), can increase your risk of developing dementia\n \n\n low cognitive engagement \n Cognitive engagement is thought to support the development of a\n “cognitive reserve”. This is the idea that people who actively use their brains throughout their lives may be more protected against brain cell damage caused by dementia.\n \n\n depression \n People who experience depression in mid- or later life have a higher risk of developing dementia. However, the relationship between depression and dementia is still unclear.\n \n\n Many researchers believe that depression is a risk factor for dementia, whereas others believe it may be an early symptom of the disease, or both.\n \n\n traumatic brain injury \n People who experience severe or repeated head injuries are at increased risk of developing dementia. Brain injuries may trigger a process that might eventually lead to dementia.\n \n\n This particularly affects athletes in boxing, soccer, hockey, and football, which often have repeated head injuries.\n \n\n Falls are the leading cause of traumatic brain injury. Falling is especially dangerous for older adults.\n \n\n hearing loss\n Mild levels of hearing loss increase the risk of cognitive decline and dementia. Though it is still unclear how exactly it affects cognitive decline, hearing loss can lead to social isolation, loss of independence, and problems with everyday activities.\n \n\n social isolation\n Social isolation can increase the risk of hypertension, coronary heart disease, depression, and dementia.\n \n\n Staying socially active may reduce the risk of dementia. Social interaction may also help slow down the progression of the disease.\n \n\n air pollution\n The relationship between air pollution and dementia is still unclear. However, it’s estimated that those living close to busy roads have a higher risk of dementia because they may be exposed to higher levels of air pollution from vehicle emissions.\n \n\n It’s never too soon, or too late, to make changes that will maintain or improve your brain health. Learn more about managing some of these risk factors."
ans = "Based on the provided information, your risk factors for dementia are:\n\n1. Depression: You mentioned experiencing regular episodes of depression, which increases your risk of developing dementia.\n2. Hearing loss: Your ears were blown out in the war, which may have caused hearing loss, increasing your risk of cognitive decline and dementia.\n3. Poor diet: Consuming a diet high in saturated fat, sugar, and salt increases your risk of developing many illnesses, including dementia and cardiovascular disease, which are risk factors for dementia.\n4. Lack of physical activity: There is no mention of you engaging in regular physical activity, which increases your risk of developing dementia and other health problems.\n5. Social isolation: Your frequent episodes of depression may lead to social isolation, which increases your risk of dementia.\n\nRegarding cognitive engagement, it refers to the idea that people who actively use their brains throughout their lives may be more protected against brain cell damage caused by dementia. Cognitive engagement can be achieved through various activities, such as learning new skills, reading, solving puzzles, or engaging in mentally stimulating tasks. The more you challenge your brain, the better it may perform and the lower your risk of developing dementia."

print(check_accuracy(ans,doc))