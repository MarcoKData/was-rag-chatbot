import streamlit as st
import rag_backend as backend


st.title("RAG - Chat with the 'Attention is all you Need'-Paper by Google!")

if "vector_index" not in st.session_state:
    with st.spinner("Woooohoooo... Magic is happening..."):
        st.session_state.vector_index = backend.get_index()

if "llm" not in st.session_state:
    st.session_state.llm = backend.get_llm()

input_text = st.text_input("Frage...")
if input_text:
    response = backend.rag_response(
        index=st.session_state.vector_index,
        question=input_text,
        llm=st.session_state.llm
    )
    st.write(response)
