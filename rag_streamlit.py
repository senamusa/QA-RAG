import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/user_query"

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit app
st.title("RAG QA")

with st.container():
    with st.chat_message("assistant"):
        st.write("Hi!👋 How can I help you today?")

    # load convo
    for convo in st.session_state["chat_history"]:
        convo_prompt = convo[0]
        convo_completion = convo[1]
        with st.chat_message("you", avatar="🧜‍♀"):
            st.write(convo_prompt)

        with st.chat_message("assistant"):
            st.write(convo_completion)

    # if new convo
    prompt = st.chat_input(
        "Say something",
    )

    if prompt:
        # retrieve completion
        # call the FastAPI endpoint
        response = requests.post(API_URL, json={"query": prompt})
        
        if response.status_code == 200:
            result = response.json().get("result")
            completion = result

        st.session_state["chat_history"].append(
            (
                prompt,
                completion,
            )
        )

        with st.chat_message("you", avatar="🧜‍♀"):
            st.write(prompt)

        with st.chat_message("assistant"):
            st.write(completion)
        prompt = None
        st.rerun()

st.divider()
with st.container():
    reset_b = st.button('reset chat')
    if reset_b:
        st.session_state['chat_history'] = []
        st.rerun()