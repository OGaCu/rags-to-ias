import streamlit as st
import base64
from io import BytesIO
import asyncio
import concurrent.futures
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

# Helper function to convert PIL image to base64 (for storing in session state)
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Helper function to convert base64 back to PIL image (for displaying from session state)
def base64_to_image(base64_str):
    from PIL import Image
    import io
    
    if base64_str:
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))
    return None

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_images and message.get("image"):
            img = base64_to_image(message["image"])
            if img:
                st.image(img)

# Function to generate response based on selected model
def generate_model_response(query, docs, model_choice):
    if model_choice == "Mistral":
        return mistral_generate_rag_response(query, docs, mistral_model)
    else:  # Default to Gemini
        return generate_rag_response(query, docs, model)

# Function to process RAG images
def process_images(query):
    if show_images:
        return rag_images(query)
    else: return

# Accept user input
if user_query := st.chat_input("Ask something about the course..."):
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Retrieve relevant docs and generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve relevant documents
            retrieved_docs = retrieve_info(user_query, embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed)
            
            # Run model and image processing concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Always start image processing regardless of toggle state
                # We'll pass just the query first, and update with response later
                future_images = executor.submit(process_images, user_query)
                # Start response generation
                future_response = executor.submit(
                    generate_model_response, 
                    user_query, 
                    retrieved_docs, 
                    selected_model
                )
                
                # Get response
                response = future_response.result()
                st.markdown(response)
                
                # Message data to save
                message_data = {"role": "assistant", "content": response}
                
                # Process images
                # Cancel the first image processing if it's still running
                if not future_images.done():
                    future_images.cancel()
                
                # Start image processing with both query and response
                if show_images:
                    images, top_img_score = process_images(user_query)
                    if top_img_score > 10:
                        # top_img = get_top_image(images)
                        top_img = images[0]
                        # st.image(top_img)
                        message_data["image"] = image_to_base64(top_img)
                        
            # Save the message to session state
            st.session_state.messages.append(message_data)