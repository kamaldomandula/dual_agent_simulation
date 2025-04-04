import streamlit as st
import pandas as pd
import os
from huggingface_hub import InferenceClient

# Initialize HF Inference client
@st.cache_resource
def get_client():
    # Try to get token from environment variables (works in Hugging Face Spaces)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            hf_token = st.secrets["HF_TOKEN"]  # Fallback to Streamlit secrets
        except:
            st.error("HF_TOKEN not found. Please set it in your Space's settings or secrets.toml")
            st.stop()
    return InferenceClient(token=hf_token)

client = get_client()

# Define the LLM call function
def call_llama(prompt, model="mistralai/Mistral-7B-Instruct-v0.2", max_tokens=500):
    try:
        response = client.text_generation(
            prompt=prompt,
            model=model,
            max_new_tokens=max_tokens,
            temperature=0.7
        )
        return response
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return "Sorry, I encountered an error."

# Set page config
st.set_page_config(layout="wide")
st.title("Duel Agent Simulation 🦙🦙")

# Sidebar setup
with st.sidebar:
    with st.expander("Instruction Manual"):
        st.markdown("""
            # 🦙🦙 Duel Agent Simulation
            ## Overview
            This app simulates an interview with two AI agents:
            1. **Interviewer**: Asks questions about your topic
            2. **Interviewee**: Attempts to answer (poorly at first)
            3. **Judge**: Provides feedback after each answer
            
            ## How to Use
            1. Enter a topic below
            2. Click "Run Simulation"
            3. Watch the conversation unfold
            4. See the evaluation results
            
            The simulation stops when the interviewee gives a good answer (8/10 or higher).
        """)
    
    # User inputs
    user_topic = st.text_input("Enter a topic", "Artificial Intelligence")
    submit_button = st.button("Run Simulation!")
    
    if st.button("Clear Session"):
        st.session_state.clear()
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "simulation_data" not in st.session_state:
    st.session_state.simulation_data = {
        "iterations": [],
        "questions": [],
        "answers": [],
        "feedback": [],
        "scores": []
    }

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Run simulation when button is pressed
if submit_button:
    iter_count = 0
    current_prompt = f"Ask a technical interview question about: {user_topic}"
    
    # Display initial topic
    with st.chat_message("user"):
        st.markdown(f"**Topic:** {user_topic}")
    st.session_state.messages.append({"role": "user", "content": f"Topic: {user_topic}"})
    
    with st.spinner("Running simulation..."):
        while iter_count < 6:  # Max 6 iterations
            # Interviewer asks question
            question = call_llama(
                f"""You are a technical interviewer. Ask one specific question about {user_topic}.
                Make it challenging but answerable. Return only the question."""
            ).strip()
            
            with st.chat_message("assistant"):
                st.markdown(f"**Interviewer:** {question}")
            st.session_state.messages.append({"role": "assistant", "content": f"Interviewer: {question}"})
            
            # Interviewee answers
            if iter_count < 2:  # First attempts are poor
                answer_prompt = f"""You are a nervous interviewee. Give a mediocre answer to:
                                "{question}". Return only the answer."""
            else:  # Later attempts improve
                feedback = st.session_state.simulation_data["feedback"][-1] if iter_count > 0 else ""
                answer_prompt = f"""You're learning to answer better. Previous feedback was:
                                "{feedback}". Now answer: "{question}". Return only the improved answer."""
            
            answer = call_llama(answer_prompt).strip()
            
            with st.chat_message("user"):
                st.markdown(f"**Interviewee:** {answer}")
            st.session_state.messages.append({"role": "user", "content": f"Interviewee: {answer}"})
            
            # Judge evaluates
            feedback_prompt = f"""Evaluate this interview exchange:
                                Question: {question}
                                Answer: {answer}
                                Provide specific feedback and a score from 1-10 (10=best). 
                                Format: Feedback: [your feedback] Score: [1-10]"""
            
            judge_response = call_llama(feedback_prompt).strip()
            
            # Extract score
            score = 5  # default
            if "Score:" in judge_response:
                try:
                    score_part = judge_response.split("Score:")[1].strip()
                    score = int(score_part.split()[0])
                except (ValueError, IndexError):
                    pass
            
            # Store data
            st.session_state.simulation_data["iterations"].append(iter_count)
            st.session_state.simulation_data["questions"].append(question)
            st.session_state.simulation_data["answers"].append(answer)
            st.session_state.simulation_data["feedback"].append(judge_response)
            st.session_state.simulation_data["scores"].append(score)
            
            # Show feedback
            with st.chat_message("assistant"):
                st.markdown(f"**Judge:** {judge_response}")
            st.session_state.messages.append({"role": "assistant", "content": f"Judge: {judge_response}"})
            
            # Display results table
            with st.expander("Detailed Results"):
                results_df = pd.DataFrame({
                    "Round": st.session_state.simulation_data["iterations"],
                    "Question": st.session_state.simulation_data["questions"],
                    "Answer": st.session_state.simulation_data["answers"],
                    "Score": st.session_state.simulation_data["scores"],
                    "Feedback": st.session_state.simulation_data["feedback"]
                })
                st.dataframe(results_df, use_container_width=True)
            
            # Check stopping condition
            if score >= 8:
                st.success("🎉 Simulation complete! The interviewee passed with a good answer.")
                break
                
            iter_count += 1
            current_prompt = f"Ask a follow-up question about: {user_topic}"
        
        if iter_count == 6:
            st.warning("Simulation ended - maximum rounds reached")
