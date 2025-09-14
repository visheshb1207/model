import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model_save_path = "./hallucination_judge_model"
# model.save_pretrained(model_save_path)

# # Save the tokenizer
# tokenizer.save_pretrained(model_save_path)

def hallu(sys_req,user_req,context,response) :
    model = TFAutoModelForSequenceClassification.from_pretrained(model_save_path)
    input_text = sys_req + "\n" + user_req + "\n" + context + "\n\n" + response

    # predictions = []
    # probabilities = []
        
    encodings = tokenizer(
        input_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="tf"
    )
        
        # Get logits from model
    outputs = model(encodings)
    logits = outputs.logits  # shape: [batch_size, num_classes]
        
    # Predicted class
    predictions = tf.argmax(logits, axis=1).numpy()
    # predictions.extend(batch_preds)

    # Probability of class 1 (hallucination)
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
    # probabilities.extend(batch_probs)
    return probabilities

