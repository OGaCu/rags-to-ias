import streamlit as st
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
from rag_pipeline import chunk_and_retrieve, retrieve_info, generate_rag_response, model
from rag_pipeline import get_top_image, rag_images, generate_caption
from rag_pipeline import mistral_generate_rag_response, mistral_model

# Apply nest_asyncio to allow asyncio to work within Streamlit
nest_asyncio.apply()

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge():
    return chunk_and_retrieve()

embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed = load_knowledge()

# App title
st.title("EECS 487 Chatbot")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            # st.image(message["image"], caption=message["caption"])
            st.image(message["image"])

async def async_retrieve_info(query, embedding_model, db, docs):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, retrieve_info, query, embedding_model, db, docs
        )

async def async_generate_response(query, docs, model):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, generate_rag_response, query, docs, model
        )

async def async_rag_images(query):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, rag_images, query
        )

# Accept user input
if user_query := st.chat_input("Ask something about the course..."):
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Retrieve relevant docs and generate response
    with st.chat_message("assistant"):
        # Create containers for text response and image
        text_container = st.container()
        image_container = st.container()
        

        # Define the main async function
        async def process_query():
            # Start both tasks concurrently
            docs_task = async_retrieve_info(
                user_query, embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed
            )
            images_task = async_rag_images(user_query)
            # Await both tasks concurrently
            retrieved_docs, images = await asyncio.gather(docs_task, images_task)
            images, top_image_score = images
            
            response = await async_generate_response(user_query, retrieved_docs, model)
            with text_container:
                st.markdown(response)
            
            show_image = None
            if images and len(images) > 0 and top_image_score > 12:
                top_img = get_top_image(images)
                with image_container:
                    st.image(top_img)
                    show_image = True
            else: 
                show_image = False
            
            # Update session state
            if show_image:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response, 
                    "image": top_img 
                })
            else:
                st.session_state.messages.append({"role": "assistant","content": response})
            
        # Run the async function
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_query())
