import asyncio
import streamlit as st
import os
import sys
import pandas as pd

# Add parent directory to sys.path so Python can find helper.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import call_llama  # ✅ Only import what you need

st.set_page_config(layout="wide")
st.title("Duel Agent Simulation 🦙🦙")

with st.sidebar:
    with st.expander("Instruction Manual"):
        st.markdown("""
            # 🦙🦙 Duel Agent Simulation Streamlit App

            ## Overview
            Welcome to the **Duel Agent Simulation** app! This Streamlit application allows you to chat with Meta's Llama3 model in a unique interview simulation. The app features two agents in an interview scenario, with a judge providing feedback. The best part? You simply provide a topic, and the simulation runs itself!

            ## Features

            ### 📝 Instruction Manual

            **Meta Llama3 🦙 Chatbot**

            This application lets you interact with Meta's Llama3 model through a fun interview-style chat.

            **How to Use:**
            1. **Input:** Type a topic into the input box labeled "Enter a topic".
            2. **Submit:** Press the "Submit" button to start the simulation.
            3. **Chat History:** View the previous conversations as the simulation unfolds.

            **Credits:**
            - **Developer:** Kethan Dosapati 
               - [LinkedIn](https://www.linkedin.com/in/kethan-dosapati/)  
        """)

    user_topic = st.text_input("Enter a topic", "Data Science")
    submit_button = st.button("Run Simulation!")

    if st.button("Clear Session"):
        st.session_state.messages = []
        st.experimental_rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create agents
interviewer = call_llama
interviewee = call_llama
judge = call_llama

# Initialize result lists
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
            answer = interviewee(
                f"Answer the question: {question} in a mediocre way because you are an inexperienced interviewee."
            )
        else:
            answer = interviewee(
                f"Answer the question: {question} in a mediocre way because you are an inexperienced interviewee but you really want to learn, so you learn from the judge comments: {judge_comments}"
            )

        st.chat_message("user").markdown(answer)
        st.session_state.messages.append({"role": "user", "content": answer})

        judge_comments = judge(
            f"The question is: {question}\nThe answer is: {answer}\nProvide feedback and rate the answer from 1 to 10 while 10 being the best and 1 is the worst."
        )

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
