import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any
import time
import os

# Configuration
# API_BASE_URL = "http://localhost:8000"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="PC Parts Chatbot",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #FFEBEE;
        border-left: 4px solid #E53935;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SESSION STATE INITIALIZATION ====================
def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'


# ==================== API FUNCTIONS ====================
def signup_user(email: str, password: str) -> Dict[str, Any]:
    """Sign up a new user"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/signup",
            json={"email": email, "password": password},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Login user"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}


def load_pdf_data(file_path: str, clear_existing: bool = False) -> Dict[str, Any]:
    """Load PDF data via admin endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/admin/load-data",
            params={"clear_existing": clear_existing},
            timeout=300  # 5 minutes timeout for large files
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}


def send_chat_message(message: str, token: str):
    """Send chat message and get streaming response"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message},
            headers=headers,
            stream=True,
            timeout=60
        )
        
        if response.status_code == 401:
            return None, "Authentication failed. Please login again."
        
        return response, None
    except requests.exceptions.RequestException as e:
        return None, f"Connection error: {str(e)}"


def get_chat_history(token: str) -> Dict[str, Any]:
    """Get chat history"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/history",
            headers=headers,
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}


def delete_chat_history(token: str) -> Dict[str, Any]:
    """Delete chat history"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.delete(
            f"{API_BASE_URL}/chat/history",
            headers=headers,
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}


def check_health() -> Dict[str, Any]:
    """Check API health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}


# ==================== UI COMPONENTS ====================
def show_login_page():
    """Display login page"""
    st.markdown('<h1 class="main-header"> PC Parts Chatbot</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("###  Login to Your Account")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_button = st.form_submit_button("Login", use_container_width=True)
            with col_b:
                signup_link = st.form_submit_button("Create Account", use_container_width=True)
            
            if login_button:
                if not email or not password:
                    st.error("Please fill in all fields")
                else:
                    with st.spinner("Logging in..."):
                        result = login_user(email, password)
                    
                    if result.get("success"):
                        st.session_state.authenticated = True
                        st.session_state.token = result.get("token")
                        st.session_state.user_email = result.get("email")
                        st.session_state.user_id = result.get("user_id")
                        st.session_state.current_page = 'chat'
                        st.success(" Login successful!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f" {result.get('message', 'Login failed')}")
            
            if signup_link:
                st.session_state.current_page = 'signup'
                st.rerun()


def show_signup_page():
    """Display signup page"""
    st.markdown('<h1 class="main-header"> PC Parts Chatbot</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("###  Create New Account")
        
        with st.form("signup_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                signup_button = st.form_submit_button("Sign Up", use_container_width=True)
            with col_b:
                login_link = st.form_submit_button("Back to Login", use_container_width=True)
            
            if signup_button:
                if not email or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    with st.spinner("Creating account..."):
                        result = signup_user(email, password)
                    
                    if result.get("success"):
                        st.session_state.authenticated = True
                        st.session_state.token = result.get("token")
                        st.session_state.user_email = result.get("email")
                        st.session_state.user_id = result.get("user_id")
                        st.session_state.current_page = 'chat'
                        st.success(" Account created successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f" {result.get('message', 'Signup failed')}")
            
            if login_link:
                st.session_state.current_page = 'login'
                st.rerun()


def show_sidebar():
    """Display sidebar with chat history and controls"""
    with st.sidebar:
        st.markdown("###  User Info")
        st.info(f"**Email:** {st.session_state.user_email}")
        
        st.markdown("---")
        
        # Admin Panel
        with st.expander(" Admin Panel", expanded=False):
            st.markdown("**Upload Product Data**")
            st.caption("Upload PDF or Excel files containing product information")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'xlsx', 'xls'],
                help="Supported formats: PDF, Excel"
            )
            
            clear_existing = st.checkbox("Clear existing data before upload", value=False)
            
            if st.button(" Load Data", use_container_width=True):
                if uploaded_file is None:
                    st.error("Please select a file first")
                else:
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Note: The actual file needs to be in the data/ folder on the server
                        # This is a simplified version - in production, you'd upload the file to the server first
                        st.warning(" Please place the file in the `data/` folder on the server, then click 'Load Data'")
                        
                        with st.spinner("Loading data... This may take a few minutes..."):
                            result = load_pdf_data(tmp_path, clear_existing)
                        
                        if result.get("success"):
                            st.success(f" {result.get('message')}")
                            st.info(f"Files processed: {result.get('files_processed')}")
                            st.info(f"Total chunks: {result.get('total_chunks')}")
                            if result.get('files_loaded'):
                                st.write("**Loaded files:**")
                                for file in result.get('files_loaded'):
                                    st.write(f"- {file}")
                        else:
                            st.error(f" {result.get('message')}")
                            if result.get('errors'):
                                st.write("**Errors:**")
                                for error in result.get('errors'):
                                    st.write(f"- {error}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        st.markdown("---")
        
        # Chat History
        st.markdown("###  Chat History")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Refresh", use_container_width=True):
                load_chat_history()
        with col2:
            if st.button(" Clear All", use_container_width=True):
                if st.session_state.chat_messages:
                    with st.spinner("Deleting..."):
                        result = delete_chat_history(st.session_state.token)
                    
                    if result.get("success"):
                        st.session_state.chat_messages = []
                        st.success("Chat history cleared!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Failed to clear history: {result.get('message')}")
        
        # Display chat history
        if st.session_state.chat_messages:
            st.caption(f"Total messages: {len(st.session_state.chat_messages)}")
            
            # Show recent conversations
            for i, msg in enumerate(reversed(st.session_state.chat_messages[-20:])):  # Show last 20 messages
                if msg['role'] == 'user':
                    with st.container():
                        st.markdown(f"**You:** {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}")
                        st.markdown("---")
        else:
            st.info("No chat history yet. Start a conversation!")
        
        st.markdown("---")
        
        # Health Check
        if st.button(" Check System Status", use_container_width=True):
            with st.spinner("Checking..."):
                health = check_health()
            
            if health.get("status") == "healthy":
                st.success(" System is healthy")
                st.info(f"Database loaded: {health.get('database_loaded')}")
                st.info(f"Total documents: {health.get('total_documents')}")
            else:
                st.error(" System is unhealthy")
                if 'error' in health:
                    st.error(health['error'])
        
        # Logout
        st.markdown("---")
        if st.button(" Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.user_email = None
            st.session_state.user_id = None
            st.session_state.chat_messages = []
            st.session_state.current_page = 'login'
            st.rerun()


def load_chat_history():
    """Load chat history from backend"""
    result = get_chat_history(st.session_state.token)
    
    if result.get("success"):
        st.session_state.chat_messages = result.get("messages", [])
    else:
        st.error(f"Failed to load chat history: {result.get('message')}")


def show_chat_page():
    """Display main chat interface"""
    st.markdown('<h1 class="main-header"> PC Parts Assistant</h1>', unsafe_allow_html=True)
    
    # Load chat history if not loaded
    if not st.session_state.chat_messages:
        load_chat_history()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
    
    # Chat input
    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Your message",
                placeholder="Ask about PC parts, pricing, or build recommendations...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("Send ", use_container_width=True)
        
        if send_button and user_input:
            # Add user message to chat
            user_message = {
                "role": "user",
                "content": user_input
            }
            st.session_state.chat_messages.append(user_message)
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Create placeholder for assistant response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
            
            # Get streaming response
            response_obj, error = send_chat_message(user_input, st.session_state.token)
            
            if error:
                st.error(f" {error}")
                if "Authentication failed" in error:
                    st.session_state.authenticated = False
                    time.sleep(2)
                    st.rerun()
            else:
                full_response = ""
                
                # Stream the response
                for line in response_obj.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        
                        # Parse SSE format
                        if line_text.startswith('data: '):
                            data_str = line_text[6:]  # Remove 'data: ' prefix
                            
                            try:
                                data = json.loads(data_str)
                                content = data.get('content', '')
                                done = data.get('done', False)
                                is_error = data.get('error', False)
                                
                                if is_error:
                                    st.error(f" {content}")
                                    break
                                
                                if content:
                                    full_response += content
                                    
                                    # Update the placeholder with accumulated response
                                    response_placeholder.markdown(full_response)
                                
                                if done:
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                
                # Add assistant response to chat history
                if full_response:
                    assistant_message = {
                        "role": "assistant",
                        "content": full_response
                    }
                    st.session_state.chat_messages.append(assistant_message)
            
            # Rerun to update the chat
            time.sleep(0.5)
            st.rerun()


# ==================== MAIN APP ====================
def main():
    """Main application"""
    init_session_state()
    
    # Route to appropriate page
    if not st.session_state.authenticated:
        if st.session_state.current_page == 'signup':
            show_signup_page()
        else:
            show_login_page()
    else:
        # Show sidebar and chat page
        show_sidebar()
        show_chat_page()


if __name__ == "__main__":
    main()