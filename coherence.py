import textstat
import language_tool_python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import os
import numpy as np

import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)



def check_grammar(text):
    """Returns the number of grammatical errors."""
    return len(tool.check(text))

def check_readability(text):
    """Returns the Flesch Reading Ease score. Higher is better."""
    return textstat.flesch_reading_ease(text)
def check_perplexity(text):
    """Returns the perplexity score. Lower is better."""
    if not text.strip(): 
        return float('inf')
    
    tokenize_input = tokenizer.encode(text, return_tensors='pt')
    if tokenize_input.size(1) == 0: 
        return float('inf')

    with torch.no_grad():
        outputs = model(tokenize_input, labels=tokenize_input)
        loss = outputs.loss
    
    return torch.exp(loss).item()



def calculate_final_coherence_score(grammar_errors, readability, perplexity, 
                                    weights=(0.4, 0.3, 0.3)):
    # Grammar score: fewer errors → higher score
    grammar_score = np.exp(-grammar_errors/3)
    
    # Readability: Normalize 0-100 to 0-1
    readability_score = readability / 100
    
    # Perplexity: Lower is better, invert with log scale
    capped_perplexity = min(perplexity, 400)
    perplexity_score = 1 / (1 + np.log(1 + capped_perplexity / 10))
    
    # Apply weights
    w_grammar, w_readability, w_perplexity = weights
    final_score = (
        w_grammar * grammar_score +
        w_readability * readability_score +
        w_perplexity * perplexity_score
    )
    
    # Convert to 0-100 scale
    return final_score * 100

def coherence_check(text):
    
    
    grammar_errors = check_grammar(text)
    readability_score = check_readability(text)
    perplexity_score = check_perplexity(text)
    
    
    # Calculate final coherence score with weighted metrics
    final_score = calculate_final_coherence_score(
        grammar_errors, readability_score, perplexity_score
    )
    return final_score
