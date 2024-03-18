import streamlit as st
import rag_backend as backend


st.title("RAG - Red And Green... sth like that ;)")

if "vector_index" not in st.session_state:
    with st.spinner("Woooohoooo... Magic is happening..."):
        st.session_state.vector_index = backend.get_index()

if "llm" not in st.session_state:
    st.session_state.llm = backend.get_llm()

input_text = st.text_input("Questione...")
if input_text:
    with st.spinner("Coole Fakten, tolle KI, sexy Entwickler... ;)"):
        response = backend.rag_response(
            index=st.session_state.vector_index,
            question=input_text,
            llm=st.session_state.llm
        )
        st.write(response)
