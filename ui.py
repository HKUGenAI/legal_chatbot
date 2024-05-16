import streamlit as st
from run import Chat

NEXT_Q = ""
MOCK_USER_ANS = ""
DUMMY_RESPONSE = ""
CURR_RESPONSE = ""
SIMILARITY = 0

def write_meta():
    #### Current Response:
    #{CURR_RESPONSE}

    col2.write(
        f"""#### Next Question: 
{NEXT_Q}
#### Mock User Answer:
{MOCK_USER_ANS}
#### Dummy Response:
{DUMMY_RESPONSE}
#### Similarity:
{SIMILARITY}
            """)

st.set_page_config(layout="wide")

if "model" not in st.session_state:
    st.session_state["model"] = Chat()


with st.sidebar:
    show_meta = st.toggle("Show metadata", True)
    "[View the source code](https://github.com/HKUGenAI/legal_chatbot)"


if show_meta:
    col1, col2 = st.columns([2, 1])
    col2.write("## Metadata")
    if NEXT_Q != "":
        write_meta()
else:
    col1 = st

col1.title("💬 Legal Bot")
col1.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What legal problem do you face?"}]

for msg in st.session_state.messages:
    col1.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    col1.chat_message("user").write(prompt)

    with st.spinner('Processing...'):
        msg, metas, exit = st.session_state.model.complete(prompt)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    col1.chat_message("assistant").write(msg)

    if metas and show_meta:
        NEXT_Q, MOCK_USER_ANS, CURR_RESPONSE ,DUMMY_RESPONSE, SIMILARITY = metas
        write_meta()

    if exit:
        col1.info("Session ended!")
        st.stop()