import streamlit as st
from bot import load_documents, split_text, vector_store, create_qa_chain

st.set_page_config(page_title="PDF Chat Assistant", layout="centered")

st.title("📄 Document QA Assistant")
st.markdown("Query your `AI.pdf` document using Google Gemini.")

# Initialize the QA chain in session state so it doesn't reload on every click
if "qa_chain" not in st.session_state:
    with st.spinner("Initializing system and processing PDF..."):
        try:
            docs = load_documents()
            chunks = split_text(docs)
            db = vector_store(chunks)
            st.session_state.qa_chain = create_qa_chain(db)
            st.success("System Ready!")
        except Exception as e:
            st.error(f"Error during initialization: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask something about the document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({"query": prompt})
            response = result["result"]
            
            # Extract page sources
            sources = set([str(doc.metadata.get('page', 'N/A')) for doc in result["source_documents"]])
            source_text = f"\n\n*Sources: Page(s) {', '.join(sources)}*"
            
            full_response = response + source_text
            st.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})