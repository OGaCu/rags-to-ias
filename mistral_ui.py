import streamlit as st
from rag_pipeline import chunk_and_retrieve, retrieve_info, generate_rag_response, model
from rag_pipeline import convert_from_path, get_top_image, rag_images, generate_caption
from rag_pipeline import mistral_generate_rag_response, mistral_model

# Run once to build vector DB and embeddings
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge():
    return chunk_and_retrieve()

embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed = load_knowledge()

# App title
st.title("EECS 487 Chatbot")

# Add model selection in the sidebar
st.sidebar.title("Model Settings")
selected_model = st.sidebar.radio(
    "Choose a model",
    ["Gemini", "Mistral"],
    help="Select which model to use for generating responses"
)

show_images = st.sidebar.checkbox("Show related images", value=True)


# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask something about the course..."):
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Retrieve relevant docs and generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_docs = retrieve_info(user_query, embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed)
            
            # Use selected model to generate response
            if selected_model == "Mistral":
                response = mistral_generate_rag_response(user_query, retrieved_docs, mistral_model)
            else:  # Default model
                response = generate_rag_response(user_query, retrieved_docs, model)
                
            st.markdown(response)
            
            ####### START #######
            # comment out the following lines for more efficient reponse
            if show_images:
                images, top_img_score = rag_images(user_query + response)
                # img_caption = generate_caption(images[0])
                if top_img_score > 10:
                    top_img = get_top_image(images)
                    st.image(top_img)
            ####### END #######
      
        st.session_state.messages.append({"role": "assistant", "content": response})