import streamlit as st
import sys
import os

# Add the utils folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import pandas as pd
from utils.helper import call_llama

st.set_page_config(layout="wide")

# Custom Title Centering
st.markdown("<h1 style='text-align: center; font-size: 40px; color: #fff;'>Duel Agent Simulation 🦙🦙</h1>", unsafe_allow_html=True)

# Sidebar with instruction manual
with st.sidebar:
    with st.expander("Instruction Manual"):
        st.markdown("""
            # 🦙🦙 Duel Agent Simulation Streamlit App
            ## Overview
            Welcome to the **Duel Agent Simulation** app! This Streamlit application allows you to chat with Meta's Llama3 model in a unique interview simulation.
            ## Features
            - Input a topic and start the simulation.
            - View previous conversations and feedback.
        """)

    # Text input and buttons with styling
    user_topic = st.text_input("Enter a topic", "Data Science", key="topic_input")
    submit_button = st.button("Run Simulation!", key="submit_button")
    clear_button = st.button("Clear Session", key="clear_button")

# Custom styling for buttons
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 50px;
        background-color: #FF5B5B;
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #444;
        border-radius: 5px;
        font-size: 18px;
        padding: 10px;
    }
    .stExpanderHeader {
        font-weight: bold;
        color: #FF5B5B;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create agents
interviewer = call_llama
interviewee = call_llama
judge = call_llama

iter = 0
list_of_iters = []
list_of_questions = []
list_of_answers = []
list_of_judge_comments = []
list_of_passes = []

if submit_button:
    prompt = f"Ask a question about this topic: {user_topic}"

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    while True:
        question = interviewer(prompt)
        st.chat_message("assistant").markdown(question)
        st.session_state.messages.append({"role": "assistant", "content": question})

        if iter < 5:
            answer = interviewee(f"Answer the question: {question} in a mediocre way because you are an inexperienced interviewee.")
        else:
            answer = interviewee(f"Answer the question: {question} in a mediocre way because you want to improve, using the judge's comments: {judge_comments}")

        st.chat_message("user").markdown(answer)
        st.session_state.messages.append({"role": "user", "content": answer})

        # Get judge comments
        judge_comments = judge(f"The question is: {question}\nThe answer is: {answer}\nProvide feedback and rate the answer from 1 to 10.")

        # Ensure judge_comments is not None and is treated as a string
        judge_comments = str(judge_comments) if judge_comments is not None else ""

        # Check if '8' is in judge_comments (now safe to check)
        passed_or_not = 1 if '8' in judge_comments else 0

        list_of_iters.append(iter)
        list_of_questions.append(question)
        list_of_answers.append(answer)
        list_of_judge_comments.append(judge_comments)
        list_of_passes.append(passed_or_not)

        results_tab = pd.DataFrame({
            "Iter.": list_of_iters,
            "Questions": list_of_questions,
            "Answers": list_of_answers,
            "Judge Comments": list_of_judge_comments,
            "Passed": list_of_passes
        })

        with st.expander("See explanation"):
            st.table(results_tab)

        if '8' in judge_comments:
            break

        iter += 1

# Handle the clear session functionality
if clear_button:
    st.session_state.messages = []
    st.experimental_rerun()
