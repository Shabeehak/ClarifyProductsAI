"""
ClarifyProducts.AI - Professional Streamlit Frontend
Product Review Intelligence Platform

A clean, professional interface for product review analysis
without unnecessary decorations or fake metrics.
"""

import streamlit as st
import requests
from PIL import Image
import io
import os
from typing import Optional, Dict, Any
import time

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ClarifyProducts.AI - Review Intelligence",
    page_icon="assets/favicon.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# =============================================================================
# PROFESSIONAL STYLING
# =============================================================================

st.markdown(
    """
<style>
    /* Clean, professional theme */
    .main {
        background-color: #ffffff;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Professional cards */
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    /* Buttons */
    .stButton > button {
        background: #667eea;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    .stButton > button:hover {
        background: #5568d3;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }

    /* Hide default menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Enhanced Clickable Chatbot Card - More Visible & Engaging */
    .chatbot-card {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 50px;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6);
        cursor: pointer;
        transition: all 0.3s ease;
        animation: bounce 2s ease-in-out infinite;
        display: flex;
        align-items: center;
        gap: 12px;
        max-width: 280px;
    }

    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6);
        }
        50% {
            transform: translateY(-8px);
            box-shadow: 0 12px 32px rgba(102, 126, 234, 0.8);
        }
    }

    .chatbot-card:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.9);
        animation: none;
    }

    .chatbot-card-icon {
        font-size: 32px;
        flex-shrink: 0;
        animation: wave 1.5s ease-in-out infinite;
    }

    @keyframes wave {
        0%, 100% {
            transform: rotate(0deg);
        }
        25% {
            transform: rotate(-15deg);
        }
        75% {
            transform: rotate(15deg);
        }
    }

    .chatbot-card-text {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .chatbot-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.2;
    }

    .chatbot-card-subtitle {
        font-size: 0.8rem;
        opacity: 0.95;
        line-height: 1.2;
    }

    /* Pulsing glow effect */
    .chatbot-card::before {
        content: '';
        position: absolute;
        inset: -2px;
        border-radius: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        animation: pulse-glow 2s ease-in-out infinite;
        z-index: -1;
    }

    @keyframes pulse-glow {
        0%, 100% {
            opacity: 0;
            transform: scale(1);
        }
        50% {
            opacity: 0.5;
            transform: scale(1.05);
        }
    }

    /* Full-Screen Chatbot Overlay (Claude AI / ChatGPT style) */
    .chatbot-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 9998;
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(5px);
    }

    .chatbot-container {
        position: fixed;
        top: 0;
        right: 0;
        width: 50%;
        height: 100%;
        background: #ffffff;
        box-shadow: -4px 0 24px rgba(0,0,0,0.15);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        animation: slideIn 0.3s ease-out;
    }

    @keyframes slideIn {
        from {
            transform: translateX(100%);
        }
        to {
            transform: translateX(0);
        }
    }

    .chatbot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 24px 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .chatbot-header-content {
        flex: 1;
    }

    .chatbot-title {
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 8px;
    }

    .chatbot-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        line-height: 1.4;
        font-style: italic;
    }

    .chatbot-close {
        background: rgba(255,255,255,0.2);
        border: none;
        color: white;
        font-size: 28px;
        cursor: pointer;
        padding: 8px 12px;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        transition: all 0.2s;
    }

    .chatbot-close:hover {
        background: rgba(255,255,255,0.3);
        transform: rotate(90deg);
    }

    .chatbot-body {
        padding: 32px;
        overflow-y: auto;
        flex: 1;
        background: #f9fafb;
    }

    .chatbot-welcome {
        text-align: center;
        padding: 60px 20px;
        max-width: 600px;
        margin: 0 auto;
    }

    .chatbot-welcome-icon {
        font-size: 64px;
        margin-bottom: 24px;
        opacity: 0.8;
    }

    .chatbot-welcome-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 16px;
    }

    .chatbot-welcome-text {
        font-size: 1rem;
        color: #6b7280;
        line-height: 1.6;
        margin-bottom: 32px;
    }

    .chatbot-suggestions {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        max-width: 500px;
        margin: 0 auto;
    }

    .chatbot-suggestion-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.2s;
        text-align: left;
    }

    .chatbot-suggestion-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }

    .chatbot-input-container {
        padding: 24px 32px;
        background: white;
        border-top: 1px solid #e5e7eb;
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .chatbot-container {
            width: 100%;
        }

        .chatbot-card {
            bottom: 20px;
            right: 20px;
            padding: 12px 18px;
            max-width: 220px;
        }

        .chatbot-card-icon {
            font-size: 28px;
        }

        .chatbot-card-title {
            font-size: 0.85rem;
        }

        .chatbot-card-subtitle {
            font-size: 0.75rem;
        }
    }

    @media (max-width: 480px) {
        .chatbot-card {
            padding: 10px 16px;
            max-width: 200px;
        }

        .chatbot-card-icon {
            font-size: 24px;
        }

        .chatbot-card-title {
            font-size: 0.8rem;
        }

        .chatbot-card-subtitle {
            font-size: 0.7rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR
# =============================================================================


def render_sidebar():
    """Render simplified sidebar with only essential information"""
    with st.sidebar:
        st.markdown("### System")

        # API Health Check
        try:
            response = requests.get(
                f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=2
            )
            if response.status_code == 200:
                st.success("API: Connected")
            else:
                st.error("API: Error")
        except:
            st.warning("API: Offline")

        st.markdown("---")

        # About (minimal)
        st.markdown("### About")
        st.markdown(
            """
This platform analyzes product reviews from multiple sources using AI to provide:

- Comprehensive review summaries
- Sentiment analysis
- Purchase recommendations
- Multi-source data aggregation
        """
        )


# =============================================================================
# HEADER
# =============================================================================


def render_header():
    """Render application header"""
    st.markdown(
        """
    <div class="app-header">
        <div class="app-title">ClarifyProducts.AI</div>
        <div class="app-subtitle">
            Product Review Intelligence Platform
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 1: PRODUCT SEARCH
# =============================================================================


def render_product_search_tab():
    """Render product search functionality"""
    st.subheader("Product Search")
    st.markdown("Search for products by name to view aggregated review analysis.")

    # Search input with form for Enter key support
    with st.form(key="search_form"):
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            product_name = st.text_input(
                "Product Name",
                placeholder="Enter product name (e.g., iPhone 15 Pro, CeraVe Lotion)",
                label_visibility="collapsed",
                key="product_search",
            )
        with col2:
            search_button = st.form_submit_button("Search", use_container_width=True, type="primary")
        with col3:
            clear_button = st.form_submit_button("Clear", use_container_width=True)

    # Handle clear button
    if clear_button:
        st.rerun()

    # Process search
    if search_button:
        if product_name:
            with st.spinner("Searching and analyzing reviews..."):
                # Show what's happening
                status = st.empty()
                status.info("Step 1/3: Searching product database...")
                time.sleep(0.5)
                status.info("Step 2/3: Aggregating reviews from multiple sources...")

                result = get_review_consensus(product_name)

                if result:
                    status.info("Step 3/3: Analyzing sentiment and generating insights...")
                    time.sleep(0.3)
                    status.empty()
                    display_product_results(result)
                else:
                    status.empty()
        else:
            st.warning("Please enter a product name.")


# =============================================================================
# TAB 2: IMAGE RECOGNITION
# =============================================================================


def render_image_recognition_tab():
    """Render image recognition functionality"""
    st.subheader("Image Recognition")
    st.markdown("Upload a product image for automatic identification and review analysis.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Product Image",
            type=["jpg", "jpeg", "png"],
            help="Supported: JPG, JPEG, PNG (max 10MB)",
            label_visibility="collapsed",
        )

    if uploaded_file:
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image", use_container_width=True, type="primary"):
            with st.spinner("Processing image..."):
                status = st.empty()
                status.info("Step 1/3: Analyzing image with AI vision...")
                time.sleep(0.5)

                result = identify_and_get_reviews(image)

                if result:
                    status.info("Step 2/3: Fetching product reviews...")
                    time.sleep(0.5)
                    status.info("Step 3/3: Generating insights...")
                    time.sleep(0.3)
                    status.empty()

                    st.success(f"Product identified: {result.get('product_name', 'Unknown')}")
                    display_product_results(result)
                else:
                    status.empty()
    else:
        st.info("Please upload a product image to begin analysis.")


# =============================================================================
# FLOATING CHATBOT
# =============================================================================


def render_chatbot_button():
    """Render beautiful chatbot card button in bottom-right corner"""

    # Wrapper div with unique ID for specific CSS targeting
    st.markdown('<div id="chatbot-button-wrapper">', unsafe_allow_html=True)

    # Functional button with new text
    if st.button("💬 Want to know more about any product? Ask our AI Assistant!", key="chatbot_toggle_btn", help="Chat with AI Assistant"):
        st.session_state.chatbot_open = True
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Beautiful chatbot-specific CSS (doesn't affect search button)
    st.markdown(
        """
        <style>
        /* Target ONLY the chatbot button using wrapper ID */
        #chatbot-button-wrapper button {
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            z-index: 998 !important;

            /* Beautiful card styling */
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 16px 24px !important;
            border-radius: 50px !important;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6) !important;

            /* Text styling */
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            white-space: nowrap !important;

            /* Animation */
            animation: chatBounce 2s ease-in-out infinite !important;
            transition: all 0.3s ease !important;
        }

        /* Bounce animation */
        @keyframes chatBounce {
            0%, 100% {
                transform: translateY(0);
                box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6);
            }
            50% {
                transform: translateY(-8px);
                box-shadow: 0 12px 32px rgba(102, 126, 234, 0.8);
            }
        }

        /* Hover effect - scale up and stop bounce */
        #chatbot-button-wrapper button:hover {
            transform: scale(1.1) translateY(-5px) !important;
            box-shadow: 0 12px 36px rgba(102, 126, 234, 0.9) !important;
            animation: none !important;
        }

        /* Responsive - Mobile */
        @media (max-width: 768px) {
            #chatbot-button-wrapper button {
                bottom: 20px !important;
                right: 20px !important;
                padding: 14px 20px !important;
                font-size: 0.85rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chatbot_fullscreen():
    """Render full-screen chatbot interface (completely replaces main content)"""
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Custom styling for chatbot-only view
    st.markdown(
        """
        <style>
        /* Full-screen chatbot styling */
        .chatbot-header-bar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 16px 32px;
            color: white;
            margin: 0 -1rem 1rem -1rem;
            border-radius: 0;
            position: relative;
        }

        .chatbot-header-bar h1 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
        }

        .chatbot-header-bar p {
            margin: 6px 0 0 0;
            font-size: 0.9rem;
            opacity: 0.95;
            font-style: italic;
        }

        /* Close button styling - in same row as header */
        .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border: none !important;
            font-size: 24px !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
            transition: all 0.2s !important;
        }

        .stButton > button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.3) !important;
            transform: rotate(90deg) !important;
        }

        /* Welcome screen styling - Minimal to fit screen */
        .welcome-container {
            max-width: 600px;
            margin: 0 auto;
            text-align: center;
            padding: 10px;
        }

        .welcome-icon {
            font-size: 48px;
            margin-bottom: 12px;
        }

        .welcome-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 8px;
        }

        .welcome-text {
            font-size: 0.95rem;
            color: #6b7280;
            line-height: 1.4;
            margin-bottom: 16px;
        }

        /* Chat container */
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 10px 20px;
        }

        /* Responsive Design - Mobile & Tablet */
        @media (max-width: 768px) {
            .chatbot-header-bar {
                padding: 12px 20px;
                margin: 0 -1rem 0.5rem -1rem;
            }

            .chatbot-header-bar h1 {
                font-size: 1.2rem;
            }

            .chatbot-header-bar p {
                font-size: 0.8rem;
                margin: 4px 0 0 0;
            }

            .welcome-container {
                padding: 5px;
            }

            .welcome-icon {
                font-size: 40px;
                margin-bottom: 8px;
            }

            .welcome-title {
                font-size: 1.2rem;
                margin-bottom: 6px;
            }

            .welcome-text {
                font-size: 0.85rem;
                line-height: 1.3;
                margin-bottom: 10px;
            }

            .chat-container {
                padding: 5px 15px;
            }
        }

        @media (max-width: 480px) {
            .chatbot-header-bar {
                padding: 10px 15px;
            }

            .chatbot-header-bar h1 {
                font-size: 1.1rem;
            }

            .chatbot-header-bar p {
                font-size: 0.75rem;
            }

            .welcome-icon {
                font-size: 36px;
                margin-bottom: 6px;
            }

            .welcome-title {
                font-size: 1.1rem;
            }

            .welcome-text {
                font-size: 0.8rem;
                margin-bottom: 8px;
            }
        }

        /* Adjust Streamlit's default spacing for chatbot view */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 3rem !important;
            max-height: 100vh !important;
            overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header with close button - fixed together
    col_header, col_close = st.columns([11, 1])

    with col_header:
        st.markdown(
            """
            <div class="chatbot-header-bar">
                <h1>AI Product Assistant</h1>
                <p>"Any doubt with product or help to identify best product for you based on reviews"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_close:
        st.markdown("<div style='margin-top: 12px;'>", unsafe_allow_html=True)
        if st.button("✕", key="close_chatbot", help="Close and return to Product Search", type="secondary"):
            st.session_state.chatbot_open = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Main chat area
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Welcome screen (if no messages) - Compact, no scrolling needed
    if not st.session_state.messages:
        # Minimal spacing to fit everything on screen
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-icon">🤖</div>
                <div class="welcome-title">How can I help you today?</div>
                <div class="welcome-text">
                    Ask me anything about products, comparisons, features, or get
                    personalized recommendations based on thousands of reviews.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display chat messages
    else:
        # Back button at top of chat
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back to Search", key="back_to_search"):
                st.session_state.chatbot_open = False
                st.rerun()

        st.markdown("---")

        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for idx, source in enumerate(message["sources"][:3], 1):
                            metadata = source.get("metadata", {})
                            st.markdown(
                                f"""
**Source {idx}:** {metadata.get('source', 'Unknown')}
**Rating:** {metadata.get('rating', 'N/A')}/5
**Excerpt:** {source.get('text', '')[:150]}...
                            """
                            )

        # Action buttons
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("🗑️ Clear Chat", key="clear_chat_history"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🏠 Back Home", key="back_home"):
                st.session_state.chatbot_open = False
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def handle_chat_input():
    """Handle chat input at root level (outside any containers)"""
    # Chat input (MUST be at root level)
    if st.session_state.chatbot_open:
        prompt = st.chat_input(
            "Ask me anything about products...",
            key="chat_input_main",
        )

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Generating response..."):
                response_data = query_rag_chatbot(prompt)

                if response_data:
                    ai_response = response_data.get(
                        "response", "Unable to process request."
                    )
                    message_data = {"role": "assistant", "content": ai_response}

                    if response_data.get("sources"):
                        message_data["sources"] = response_data["sources"]

                    st.session_state.messages.append(message_data)
                else:
                    error_msg = "Error processing request. Please try again."
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

            st.rerun()


# =============================================================================
# API FUNCTIONS
# =============================================================================


def get_review_consensus(product_name: str) -> Optional[Dict[str, Any]]:
    """Get product review consensus from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/smart-search/",
            params={"q": product_name, "use_cache": True, "include_ml": True},
            timeout=60,  # Increased to 60 seconds for summarization
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error("Product not found. Please try a different search term.")
        elif response.status_code == 500:
            st.error("Server error. Please try again later.")
        else:
            st.error(f"Request failed (Status: {response.status_code})")
        return None

    except requests.Timeout:
        st.error(
            """
**Request Timeout**

This usually happens when:
- Product has many reviews (processing takes longer)
- Backend server is slow to respond

**What to try:**
- Search for a more specific product name
- Try again in a few moments
- Check if backend server is running
        """
        )
        return None
    except requests.ConnectionError:
        st.error(
            """
**Connection Error**

Cannot connect to backend server.

**Please ensure:**
- Backend is running at http://localhost:8000
- No firewall blocking the connection

**Start backend:** `uvicorn app.main:app --reload`
        """
        )
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


def identify_and_get_reviews(image: Image.Image) -> Optional[Dict[str, Any]]:
    """Identify product from image"""
    try:
        # Prepare image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        files = {"image": ("image.png", img_byte_arr, "image/png")}

        response = requests.post(f"{API_BASE_URL}/recognition/", files=files, timeout=60)

        if response.status_code == 200:
            result = response.json()

            if result.get("success"):
                primary_match = result.get("primary_match", {})
                product_name = primary_match.get("product_name", "unknown")
                confidence = primary_match.get("confidence", "medium")

                # Show info message if confidence is low (only category detected)
                if confidence == "low":
                    st.info(f"Could not identify exact product name. Showing results for category: **{product_name}**. For exact product identification, please upload an image of the product along with its package.")

                # Get reviews
                review_data = get_review_consensus(product_name)

                if review_data:
                    review_data["recognition"] = result
                    review_data["product_name"] = product_name
                    return review_data
                else:
                    return {"product_name": product_name, "recognition": result}
            else:
                st.error(result.get("message", "Unable to identify product from image."))
                st.info(
                    """
**Tips for better recognition:**
- Ensure good lighting
- Capture product label clearly
- Avoid blurry images
- Show brand name if visible
                """
                )
                return None
        else:
            st.error(f"Recognition failed (Status: {response.status_code})")
            return None

    except requests.Timeout:
        st.error(
            """
**Image Processing Timeout**

Image recognition is taking longer than expected (>60 seconds).

**What to try:**
- Use a clearer, higher quality image
- Try a different product image
- Check backend server load
        """
        )
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def query_rag_chatbot(query: str) -> Optional[Dict[str, Any]]:
    """Query RAG chatbot"""
    try:
        payload = {"query": query, "n_results": 5}
        response = requests.post(f"{API_BASE_URL}/rag/query", json=payload, timeout=90)

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except requests.Timeout:
        st.error(
            """
**Response Timeout**

AI is taking longer than expected to generate a response.

**What to try:**
- Ask a more specific question
- Try a different product
- Wait a moment and try again
        """
        )
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def display_product_results(result: Dict[str, Any]):
    """Display product results professionally"""

    product = result.get("product", {})
    product_name = product.get("name", result.get("product_name", "Unknown Product"))

    st.markdown(f"## {product_name}")

    # Typo correction notice
    if result.get("suggestions"):
        original_query = result.get("original_query", "")
        if original_query and original_query.lower() != product_name.lower():
            st.info(f"Showing results for: **{product_name}** (auto-corrected)")

    # Product images
    if product.get("images"):
        cols = st.columns(min(len(product["images"]), 4))
        for idx, img in enumerate(product["images"][:4]):
            with cols[idx]:
                st.image(img.get("url"), use_column_width=True)

    st.divider()

    # Key metrics (removed "Status", kept only important ones)
    col1, col2, col3 = st.columns(3)

    with col1:
        rating = product.get("overall_rating", 0)
        st.metric("Average Rating", f"{rating:.1f}/5.0")

    with col2:
        review_count = product.get("total_reviews", 0)
        st.metric("Total Reviews", f"{review_count:,}")

    with col3:
        recommendation = product.get("recommendation", {})
        rec_badge = recommendation.get("badge", "N/A")
        st.metric("Recommendation", rec_badge)

    # Recommendation details
    if recommendation:
        decision = recommendation.get("decision")
        confidence = recommendation.get("confidence", 0)
        reason = recommendation.get("reason", "")

        if decision == "buy":
            st.success(f"**{rec_badge}** (Confidence: {confidence:.0f}%)\n\n{reason}")
        elif decision == "consider":
            st.info(f"**{rec_badge}** (Confidence: {confidence:.0f}%)\n\n{reason}")
        elif decision == "wait":
            st.warning(f"**{rec_badge}** (Confidence: {confidence:.0f}%)\n\n{reason}")
        elif decision == "skip":
            st.error(f"**{rec_badge}** (Confidence: {confidence:.0f}%)\n\n{reason}")

    # Review Consensus
    consensus = result.get("consensus", {})
    summary_text = consensus.get("summary", "")
    sentiment = consensus.get("sentiment", {})

    # Debug logging (uncomment to see what data is received)
    # st.write("DEBUG - Full result keys:", list(result.keys()))
    # st.write("DEBUG - Consensus:", consensus)
    # st.write("DEBUG - Summary:", summary_text)

    # Show if we have sentiment data or a summary
    has_analysis = (sentiment and sentiment.get("positive_percent") is not None) or (summary_text and summary_text != "No reviews available for analysis.")

    if has_analysis:
        st.divider()
        st.subheader("Review Analysis")

        # Summary - Always try to show something if we have data
        if summary_text and summary_text != "No reviews available for analysis.":
            st.markdown("**Summary**")
            st.info(summary_text)
        elif sentiment and sentiment.get("total_analyzed", 0) > 0:
            # Fallback: If no summary text but we have analyzed reviews, show basic info
            total = sentiment.get("total_analyzed", 0)
            pos = sentiment.get("positive_percent", 0)
            neg = sentiment.get("negative_percent", 0)
            st.markdown("**Summary**")
            if pos > 60:
                st.info(f"Based on {total} reviews analyzed, customers generally have positive sentiment ({pos:.0f}% positive).")
            elif neg > 60:
                st.info(f"Based on {total} reviews analyzed, customers express concerns ({neg:.0f}% negative).")
            else:
                st.info(f"Based on {total} reviews analyzed, customer opinions are mixed ({pos:.0f}% positive, {neg:.0f}% negative).")

        # Sentiment
        if sentiment and sentiment.get("positive_percent") is not None:
            st.markdown("**Sentiment Distribution**")
            st.markdown(f"*Based on {sentiment.get('total_analyzed', 0)} reviews analyzed*")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Positive", f"{sentiment.get('positive_percent', 0):.0f}%")
            with col2:
                st.metric("Neutral", f"{sentiment.get('neutral_percent', 0):.0f}%")
            with col3:
                st.metric("Negative", f"{sentiment.get('negative_percent', 0):.0f}%")

        # Pros and Cons
        if consensus.get("pros") or consensus.get("cons"):
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Strengths**")
                if consensus.get("pros"):
                    for pro in consensus["pros"]:
                        st.markdown(f"- {pro}")
                else:
                    st.write("_No specific strengths extracted_")

            with col2:
                st.markdown("**Weaknesses**")
                if consensus.get("cons"):
                    for con in consensus["cons"]:
                        st.markdown(f"- {con}")
                else:
                    st.write("_No specific weaknesses extracted_")

    # Source Statistics
    sources = result.get("sources", {})
    if sources:
        st.divider()
        st.subheader("Data Sources")

        col1, col2, col3 = st.columns(3)

        youtube_stats = sources.get("youtube", {})
        if youtube_stats.get("has_data"):
            with col1:
                st.markdown("**YouTube**")
                st.write(f"Videos: {youtube_stats.get('video_count', 0)}")
                st.write(f"Total Views: {youtube_stats.get('total_views', 0):,}")

        reddit_stats = sources.get("reddit", {})
        if reddit_stats.get("has_data"):
            with col2:
                st.markdown("**Reddit**")
                st.write(f"Posts: {reddit_stats.get('post_count', 0)}")
                st.write(f"Comments: {reddit_stats.get('comment_count', 0)}")

        twitter_stats = sources.get("twitter", {})
        if twitter_stats.get("has_data"):
            with col3:
                st.markdown("**Twitter**")
                st.write(f"Tweets: {twitter_stats.get('tweet_count', 0)}")
                st.write(f"Likes: {twitter_stats.get('total_likes', 0):,}")

    # YouTube Review Videos
    review_videos = result.get("review_videos", [])
    if review_videos:
        st.divider()
        st.subheader("YouTube Reviews")
        st.markdown("*Watch what others are saying about this product*")

        for video in review_videos[:5]:  # Show top 5 videos
            col1, col2 = st.columns([1, 3])
            with col1:
                # Show thumbnail if available
                if video.get("thumbnail"):
                    st.image(video["thumbnail"], use_column_width=True)
            with col2:
                st.markdown(f"**[{video.get('title', 'Video')}]({video.get('url')})**")
                st.caption(f"👁️ {video.get('view_count', 0):,} views • {video.get('channel', 'Unknown Channel')}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Main application"""

    # Initialize chatbot state
    if "chatbot_open" not in st.session_state:
        st.session_state.chatbot_open = False

    # IF CHATBOT IS OPEN: Show ONLY chatbot interface (full screen)
    if st.session_state.chatbot_open:
        render_chatbot_fullscreen()
        handle_chat_input()
        return  # Don't render anything else

    # ELSE: Show normal product search/image interface
    render_sidebar()
    render_header()

    # Main tabs (removed AI Assistant tab)
    tab1, tab2 = st.tabs(["Product Search", "Upload Image Search"])

    with tab1:
        render_product_search_tab()

    with tab2:
        render_image_recognition_tab()

    # Floating chatbot button (always visible)
    render_chatbot_button()


if __name__ == "__main__":
    main()
