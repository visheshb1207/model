import streamlit as st
import pickle
import numpy as np
import pandas as pd
from accuracy import *
from assumptions import *
from instructions import *
from hallucination import *
from coherence import *
import time
import nltk
nltk.download('punkt')


# Page configuration
st.set_page_config(
    page_title="Agentic Evaluation Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🧩 Agentic Evaluation")
st.sidebar.write("Fill in the details to evaluate your AI response.")

sys_req = st.sidebar.text_input("System Requirements")
user_req = st.sidebar.text_input("User Request")
context = st.sidebar.text_input("Context")
response = st.sidebar.text_input("Response")

if st.sidebar.button("✅ Evaluate"):
    
    
    
    # Assumption Evaluation
    Detector = AssumptionDetector()
    results = Detector.detect(context, response)
    assumption_score = results['confidence']
    
    # Instruction Evaluation
    res = evaluate_instruction_following(sys_req, response)
    instruction_score = res['overall_score']
    
    # Coherence Evaluation
    coh_score = coherence_check(response)

    # Accuracy Evaluation
    acc_score = check_accuracy(response, context)

    # Hallucination Evaluation
    hallu_score = hallu(sys_req,user_req,context,response)


    
    
    # Display results in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="🧠 Assumption Confidence", value=f"{assumption_score:.2f}")
        
    with col2:
        st.metric(label="📋 Instruction Score", value=f"{instruction_score:.2f}")
        
    with col3:
        st.metric(label="🔗 Coherence Score", value=f"{coh_score:.2f}")
    
    with col4:
        st.metric(label="✅ Accuracy Score", value=f"{acc_score:.2f}")

    with col5:
        st.metric(label="👻 Hallucination Score", value=f"{hallu_score[0]:.2f}")
    

    st.success("✅ Evaluation Complete!")
    # Detailed Results Section
    with st.expander("📖 View Details"):
        st.markdown("### Inputs")
        st.write("**System Requirements:**", sys_req)
        st.write("**User Request:**", user_req)
        st.write("**Context:**", context)
        st.write("**Response:**", response)
        
        st.markdown("### Evaluation Results")
        st.write("**Assumption Confidence:**", assumption_score)
        st.write("**Instruction Following Score:**", instruction_score)
        st.write("**Coherence Score:**", coh_score)
        st.write("**Hallucination Score:**", hallu_score)
    
    st.balloons()
