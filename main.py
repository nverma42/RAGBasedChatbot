from gettext import npgettext
from math import e

import numpy as np
from model import rag_model
import openai
import os
from dotenv import load_dotenv
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
def open_ai_generate(user_message, retrieved_response):
    messages = [
                {"role": "system", "content": "You are a compassionate, empathetic therapist. Respond warmly, supportively, and incorporate any professional suggestions provided. Do not offer strict medical advice."},
                {"role": "user", "content": f"User said: '{user_message}'\nTherapist suggests: '{retrieved_response}'\nPlease craft a warm, supportive reply."}
               ]

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

    return response["choices"][0]["message"]

def compute_self_bleu(generated_responses):
    """
    Computes the self-BLEU score between the retrieved response and the generated response.
    """
    # Tokenize the responses
    scores = []
    smoothing_fx = SmoothingFunction().method1  # Helps avoid zero BLEU for short sentences
    for sentence in generated_responses:
        hypothesis = sentence['content'].split()
        references = [other_sentence['content'].split() for other_sentence in generated_responses if other_sentence['content'] != sentence['content']]
        
        # Calculate BLEU score
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothing_fx)
        scores.append(score)
    # Calculate average BLEU score
    return sum(scores) / len(scores) if scores else 0

# Main Flow

# Load environment variables from .env file.
load_dotenv()

# Access key
openai.api_key = os.getenv("OPENAI_API_KEY")

model = rag_model()

# build the knowledge graph
model.build_knowledge_graph()

# retrieve the data from the knowledge graph
retrieved_data_list = model.retrieve_data()

# Filter the retrieved data by eliminating duplicates
filtered_data = []
for data in retrieved_data_list:
    if data not in filtered_data:
        filtered_data.append(data)

# Generate the response for filtered data
generated_responses = []
for data in filtered_data:
    response = open_ai_generate(data[0], data[1])
    generated_responses.append(response)

score = compute_self_bleu(generated_responses)
print(f"Self-BLEU score: {score}")

# Calculate the average similarity between the generated responses and rerieved documents
similarity = 0
for i, data in enumerate(filtered_data):
    encoded_retrieved_doc = model.encoder.encode(data[1])
    encoded_generated_doc = model.encoder.encode(generated_responses[i]['content'])
    similarity += np.dot(encoded_retrieved_doc, encoded_generated_doc) / (np.linalg.norm(encoded_retrieved_doc) * np.linalg.norm(encoded_generated_doc))

print(f"Average similarity between retrieved documents and generated responses : {similarity/len(filtered_data)}") 
