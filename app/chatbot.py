import streamlit as st
from PIL import Image

from retrieve_pipeline import answer_question_from_pdf

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="TripSafe Insurance Assistant",
    page_icon="/Users/arvindpanwar/workspace/insurance-rag-system/data/tripsafe.avif",
    layout="centered"
)

logo = Image.open("/Users/arvindpanwar/workspace/insurance-rag-system/data/tripsafe.avif")
st.image(logo,width=120)
st.title("TripSafe Insurance Assistant")
st.caption("    Ask questions about your policy. Answers are based only on the uploaded document.")

# -----------------------------
# Session state for chat
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# User input
# -----------------------------
user_query = st.chat_input("Ask a question about the policy...")

if user_query:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Call RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question_from_pdf(user_query)

            answer = result["answer"]
            citations = result["citations"]
            confidence = result["confidence"]

            # Render answer
            st.markdown(answer)

            # Confidence bar
            st.markdown("**Confidence**")
            st.progress(confidence)

            # Citations (collapsible)
            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        st.markdown(f"- {c}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
