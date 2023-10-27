import streamlit as st

from llama import msgs, retrieval_chain

st.set_page_config(page_title="Mistral-7b Chatbot with Retrieval", page_icon="📖")
st.title("📖 Mistral-7b Chatbot with Retrieval")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = retrieval_chain({"query": prompt})
    st.chat_message("ai").write(response["result"])


# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
