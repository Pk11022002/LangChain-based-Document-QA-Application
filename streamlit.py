import streamlit as st
from prompts import rag_chain

# Display heading on both login and chatbox pages
st.markdown(
    "<h1 style='text-align: center; margin-top: -24px;'>Wavenet Technologies Chatbot</h1>",
    unsafe_allow_html=True,
)

# Chat input at the bottom of the page
prompt = st.chat_input("Ask your NL2SQL Question...")

# Process input only if a prompt is provided
if prompt:
    st.session_state.prompt = prompt

    # Display the question in the chat format
    st.chat_message("user").markdown(f"**Question:** {prompt}")

    with st.spinner("Generating response..."):
        try:
            # Call the rag_chain to process the input
            output = rag_chain.invoke({"input": prompt})

            # Display the answer in the chat format
            st.chat_message("assistant").markdown(f"**Answer:** {output['answer']}")

        except Exception as e:
            # Handle any exceptions that might occur during the processing
            st.error(f"An error occurred: {str(e)}")
