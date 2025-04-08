import time
import uuid
import streamlit as st
from datetime import datetime
import travel_agency_bot_engine as chatbot
from PIL import Image
import memory as db

chatbot_instance = chatbot.TravelAgencyBot()
db_instance = db.ChatHistoryDB("chat_history.db")
db_instance.create_table("chat_history")

icon = Image.open('static_files/logo.png')
# Set page configuration
st.set_page_config(
    page_title="Your travel assistant - Nomad AI",
    page_icon=icon,
    layout="centered"
)

# Custom CSS for better chat appearance
st.markdown("""
<style>
    .user-message {
        background-color: #F5F5F5;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-end;
        float: right;
        clear: both;
    }
    .ai-message {
        background-color: #E9E9E9;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-start;
        float: left;
        clear: both;
    }
    .chat-container {
        padding: 20px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    .message-container {
        width: 100%;
        overflow: hidden;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# list of created chats (history)
if "chats" not in st.session_state:
    history = db_instance.read_all_chats()
    if history:
        st.session_state["chats"] = history
    else:
        st.session_state["chats"] = []
    
    # st.session_state["chats"]  = [
    #    {
    #        "conversation_id": uuid.uuid4(),
    #        "header": "Gdzie pojechać...",
    #        "create_date": "2025-03-01T12:00:00Z",
    #        "history": [
    #            {
    #                  "role": "human",
    #                  "content": "Gdzie pojechać na wakacje?",
    #                  "create_date": "2025-03-01T12:00:00Z"
    #             },
    #             {
    #                  "role": "assistant",
    #                  "content": "Na wakacje polecam wybrać się do Grecji lub Hiszpanii. Oba kraje oferują piękne plaże i wiele atrakcji turystycznych.",
    #                  "create_date": "2025-03-01T12:00:00Z"
    #            }
    #        ]
    #    },
    #    {
    #        "conversation_id": uuid.uuid4(),
    #        "header": "Gdzie tanie loty?",
    #        "create_date": "2025-03-02T12:00:00Z",
    #    }
    # ]

def create_new_chat(default_header="New conversation"):
    new_chat = {
        "conversation_id": uuid.uuid4(),
        "header": default_header if len(default_header) <= 30 else default_header[:30] + "...",
        "create_date": datetime.now().isoformat(),
        "history": []
    }
    st.session_state["chats"].append(new_chat)
    st.session_state["current_chat"] = new_chat

def chatbot_response(user_input, conversation_id):
    # Simulate a response from the chatbot
    chatbot_instance.process_user_input(user_input)
    response = {
        "conversation_id": conversation_id,
        "response": f"{chatbot_instance.provide_answer()}"
    }
    return response

######################################################################################################
# Sidebar for chat history and new conversation creation
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.image(icon, width=80, use_container_width=False)
    with col2:
        st.markdown("<h1 style='text-align: center;'>Nomad AI</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Create new conversation"):
        create_new_chat()
    st.header("Chat history")
    sorted_chats = sorted(st.session_state["chats"], key=lambda x: x["create_date"], reverse=True)
    for chat in sorted_chats:
        if st.button(f"{chat['header']}", key=chat["conversation_id"]):
            st.session_state["current_chat"] = chat
            st.rerun()

#####################################################################################################
# Main chat area
st.title("Your travel assistant - Nomad AI")
st.write("Hello traveler! I am your travel assistant. How can I help you today?")

with st.container():
    messages_html = '<div class="chat-messages">'
    if "current_chat" in st.session_state:
        # Display chat messages
        if "history" in st.session_state["current_chat"]:
            for message in st.session_state["current_chat"]["history"]:
                role = message["role"]
                content = message["content"]

                with st.container():
                    if role == "human":
                        messages_html += f'<div class="message-container"><div class="user-message">{content}</div></div>'
                    else:
                        messages_html += f'<div class="message-container"><div class="ai-message">{content}</div></div>'

    st.markdown(messages_html, unsafe_allow_html=True)
    # Input for new message
    with st.container():
        user_input = st.chat_input("Type your message here...")

        if user_input:
            if "current_chat" not in st.session_state:
                create_new_chat(user_input)
            else:
                # Check if the current chat is empty
                if not st.session_state["current_chat"]["history"]:
                    st.session_state["current_chat"]["header"] = user_input[:30] + "..." if len(user_input) > 30 else user_input

            # Add user message to the display
            st.session_state["current_chat"]["history"].append({"role": "human", "content": user_input, "create_date": datetime.now().isoformat()})

            # Create a placeholder for the AI's response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")

                # Get response from the chatbot
                response = chatbot_response(user_input, st.session_state["current_chat"]["conversation_id"])

                # Save the conversation ID
                st.session_state["current_chat"]["conversation_id"] = response["conversation_id"]

                # Add AI message to the display
                st.session_state["current_chat"]["history"].append({"role": "assistant", "content": response["response"], "create_date": datetime.now().isoformat()})
                
                #Save chat history to the database (passing it as a list)
                db_instance.save_chat_history([st.session_state["current_chat"]])

                # Display typing effect
                full_response = response["response"]
                simulated_response = ""

                for chunk in full_response.split():
                    simulated_response += chunk + " "
                    message_placeholder.markdown(simulated_response + "▌")
                    time.sleep(0.2)

                st.rerun()
###################################################################################################################