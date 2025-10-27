"""
ClarifyProducts.AI - Modern Streamlit Frontend
Professional Product Review Consensus Platform

Features:
- Product Search with AI-powered analysis
- Image Recognition for product identification
- Interactive AI Chatbot for product queries
- Real-time review aggregation from multiple sources
"""

import streamlit as st
import requests
from PIL import Image
import io
import os
from typing import Optional, Dict, Any

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ClarifyProducts.AI - Product Review Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/clarifyproducts",
        "Report a bug": "https://github.com/your-repo/clarifyproducts/issues",
        "About": "ClarifyProducts.AI - Making product decisions easier with AI-powered review analysis",
    },
)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# =============================================================================
# MODERN CUSTOM STYLING
# =============================================================================

st.markdown(
    """
<style>
    /* Main app styling */
    .main {
        background-color: #f8f9fa;
    }

    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .app-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Card styling */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Alert styling */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #dee2e6;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR
# =============================================================================


def render_sidebar():
    """Render sidebar with navigation and info"""
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        st.markdown("---")

        # API Status indicator
        st.markdown("### 🔌 System Status")
        try:
            response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.warning("⚠️ API Offline")

        st.markdown("---")

        # Quick stats
        st.markdown("### 📈 Platform Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Products", "10K+")
        with col2:
            st.metric("Reviews", "1M+")

        st.markdown("---")

        # About section
        st.markdown("### ℹ️ About")
        st.markdown(
            """
        **ClarifyProducts.AI** helps you make informed purchase decisions by:

        - 🤖 Analyzing thousands of reviews
        - 📸 Identifying products from images
        - 💬 Answering specific questions
        - 🎯 Providing actionable insights
        """
        )

        st.markdown("---")
        st.markdown("### 🔗 Resources")
        st.markdown("[📖 Documentation](#)")
        st.markdown("[💡 GitHub](#)")
        st.markdown("[📧 Support](#)")


# =============================================================================
# HEADER
# =============================================================================


def render_header():
    """Render application header"""
    st.markdown(
        """
    <div class="app-header">
        <div class="app-title">🔍 ClarifyProducts.AI</div>
        <div class="app-subtitle">
            Discover what real people think about products • Powered by AI
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
    st.markdown("### 🔎 Search by Product Name")
    st.markdown("Enter a product name to get comprehensive review analysis from multiple sources.")

    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        product_name = st.text_input(
            "Product Name",
            placeholder="e.g., CeraVe Moisturizing Lotion, iPhone 15 Pro, AirPods Pro...",
            label_visibility="collapsed",
            key="product_search_input",
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True, type="primary")

    # Process search
    if search_button and product_name:
        with st.spinner(f"🔍 Analyzing reviews for **{product_name}**..."):
            result = get_review_consensus(product_name)
            if result:
                display_product_results(result)
    elif search_button:
        st.warning("⚠️ Please enter a product name to search.")


# =============================================================================
# TAB 2: IMAGE RECOGNITION
# =============================================================================


def render_image_recognition_tab():
    """Render image recognition functionality"""
    st.markdown("### 📸 Identify Product from Image")
    st.markdown("Upload a product image and our AI will identify it and fetch reviews automatically.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Product Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            label_visibility="collapsed",
        )

    if uploaded_file:
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Recognition button
        if st.button("🔍 Identify & Analyze", use_container_width=True, type="primary"):
            with st.spinner("🤖 Identifying product using AI vision..."):
                result = identify_and_get_reviews(image)
                if result:
                    st.success(f"✅ Identified: **{result.get('product_name', 'Unknown')}**")
                    display_product_results(result)
    else:
        st.info("👆 Upload an image to get started")


# =============================================================================
# TAB 3: AI CHATBOT
# =============================================================================


def render_chatbot_display():
    """Render AI chatbot display (without input - input is outside tabs)"""
    st.markdown("### 💬 Chat with AI Assistant")
    st.markdown("Ask anything about products, compare features, or get personalized recommendations.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome message if no chat history
    if not st.session_state.messages:
        st.info(
            """
        👋 **Hi! I'm your AI product assistant.**

        I can help you with:
        - Product comparisons
        - Feature analysis
        - Review summaries
        - Purchase recommendations

        **Type your question below!** ⬇️
        """
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for idx, source in enumerate(message["sources"][:3], 1):
                        metadata = source.get("metadata", {})
                        st.markdown(
                            f"""
                        **Source {idx}:** {metadata.get('source', 'Unknown')}
                        **Rating:** ⭐ {metadata.get('rating', 'N/A')}/5
                        **Excerpt:** {source.get('text', '')[:200]}...
                        """
                        )

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def handle_chat_input():
    """Handle chat input (must be outside tabs due to Streamlit limitation)"""
    # Chat input - this MUST be at root level, not in tabs
    if prompt := st.chat_input("Ask me anything about products..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response_data = query_rag_chatbot(prompt)

            if response_data:
                ai_response = response_data.get("response", "I couldn't process that request.")

                # Store message with sources
                message_data = {"role": "assistant", "content": ai_response}

                if response_data.get("sources"):
                    message_data["sources"] = response_data["sources"]

                st.session_state.messages.append(message_data)
            else:
                error_msg = "Sorry, I encountered an error. Please try again."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

        # Rerun to show the new messages
        st.rerun()


# =============================================================================
# HELPER FUNCTIONS - API CALLS
# =============================================================================


def get_review_consensus(product_name: str) -> Optional[Dict[str, Any]]:
    """
    Get review consensus for a product using smart search

    Args:
        product_name: Name of the product to search

    Returns:
        Dictionary with product data or None if failed
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/smart-search/",
            params={"q": product_name, "use_cache": True, "include_ml": True},
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Search failed with status {response.status_code}")
            return None

    except requests.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Make sure the backend server is running at http://localhost:8000")
        return None


def identify_and_get_reviews(image: Image.Image) -> Optional[Dict[str, Any]]:
    """
    Identify product from image and get reviews

    Args:
        image: PIL Image object

    Returns:
        Dictionary with recognition and review data or None if failed
    """
    try:
        # Prepare image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        files = {"image": ("image.png", img_byte_arr, "image/png")}

        # Call recognition API
        response = requests.post(
            f"{API_BASE_URL}/recognition/", files=files, timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("success"):
                primary_match = result.get("primary_match", {})
                product_name = primary_match.get("product_name", "unknown product")

                # Get reviews for identified product
                review_data = get_review_consensus(product_name)

                # Merge recognition and review data
                if review_data:
                    review_data["recognition"] = result
                    review_data["product_name"] = product_name
                    return review_data
                else:
                    return {"product_name": product_name, "recognition": result}
            else:
                st.error(f"❌ {result.get('message', 'Recognition failed')}")
                return None
        else:
            st.error(f"❌ Recognition failed with status {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        st.error("⏱️ Recognition timed out. Please try again with a clearer image.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def query_rag_chatbot(query: str) -> Optional[Dict[str, Any]]:
    """
    Query the RAG chatbot

    Args:
        query: User question

    Returns:
        Dictionary with response and sources or None if failed
    """
    try:
        payload = {"query": query, "n_results": 5}

        response = requests.post(
            f"{API_BASE_URL}/rag/query", json=payload, timeout=120
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Chat failed with status {response.status_code}")
            return None

    except requests.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def display_product_results(result: Dict[str, Any]):
    """Display product search results in a structured format"""

    # Product header
    product = result.get("product", {})
    product_name = product.get("name", result.get("product_name", "Unknown Product"))

    st.markdown(f"## 📦 {product_name}")

    # Show if typo was corrected
    if result.get("suggestions") and result["product_name"].lower() != product_name.lower():
        st.info(
            f"💡 Showing results for: **{result['product_name']}** (corrected from your search)"
        )

    # Product images
    if product.get("images"):
        cols = st.columns(min(len(product["images"]), 4))
        for idx, img in enumerate(product["images"][:4]):
            with cols[idx]:
                st.image(img.get("url"), use_column_width=True)

    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rating = product.get("overall_rating", 0)
        st.metric("⭐ Rating", f"{rating:.1f}/5.0")

    with col2:
        review_count = product.get("total_reviews", 0)
        st.metric("📊 Reviews", f"{review_count:,}")

    with col3:
        recommendation = product.get("recommendation", {})
        rec_badge = recommendation.get("badge", "N/A")
        st.metric("💡 Verdict", rec_badge)

    with col4:
        cache_status = "🚀" if result.get("metadata", {}).get("from_cache") else "🆕"
        st.metric("Status", f"{cache_status} {'Cached' if cache_status == '🚀' else 'Fresh'}")

    # Recommendation details
    if recommendation:
        decision = recommendation.get("decision")
        confidence = recommendation.get("confidence", 0)
        reason = recommendation.get("reason", "")

        if decision == "buy":
            st.success(f"✅ **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")
        elif decision == "consider":
            st.info(f"👍 **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")
        elif decision == "wait":
            st.warning(f"⏳ **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")
        elif decision == "skip":
            st.error(f"❌ **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")

    # Review Consensus
    consensus = result.get("consensus", {})
    if consensus and consensus.get("summary"):
        st.markdown("---")
        st.markdown("### 🤖 Review Consensus")

        # Summary
        if consensus.get("summary"):
            st.markdown("#### 📝 What People Are Saying")
            st.info(consensus["summary"])

        # Sentiment Distribution
        sentiment = consensus.get("sentiment", {})
        if sentiment:
            st.markdown("#### 📊 Sentiment Distribution")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "👍 Positive",
                    f"{sentiment.get('positive_percent', 0):.0f}%",
                    delta=None,
                )
            with col2:
                st.metric("😐 Neutral", f"{sentiment.get('neutral_percent', 0):.0f}%")
            with col3:
                st.metric(
                    "👎 Negative",
                    f"{sentiment.get('negative_percent', 0):.0f}%",
                    delta=None,
                )

        # Pros and Cons
        if consensus.get("pros") or consensus.get("cons"):
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Pros")
                if consensus.get("pros"):
                    for pro in consensus["pros"]:
                        st.markdown(f"• {pro}")
                else:
                    st.write("_No specific pros extracted_")

            with col2:
                st.markdown("#### ❌ Cons")
                if consensus.get("cons"):
                    for con in consensus["cons"]:
                        st.markdown(f"• {con}")
                else:
                    st.write("_No specific cons extracted_")

    # YouTube Reviews
    review_videos = result.get("review_videos", [])
    if review_videos:
        st.markdown("---")
        st.markdown("### 🎥 YouTube Review Videos")

        for video in review_videos[:3]:
            with st.expander(
                f"🎬 {video.get('title', 'Video')} - {video.get('channel', 'Unknown')}"
            ):
                col1, col2 = st.columns([1, 2])

                with col1:
                    if video.get("thumbnail"):
                        st.image(video["thumbnail"], use_column_width=True)

                with col2:
                    st.markdown(f"**Channel:** {video.get('channel', 'Unknown')}")
                    st.markdown(f"**Views:** {video.get('views', 0):,}")
                    st.markdown(f"**URL:** [{video.get('url', '')}]({video.get('url', '')})")
                    if video.get("has_transcript"):
                        st.success("✅ Has transcript")

    # Source Statistics
    sources = result.get("sources", {})
    if sources:
        st.markdown("---")
        st.markdown("### 📚 Review Sources")

        col1, col2, col3 = st.columns(3)

        youtube_stats = sources.get("youtube", {})
        if youtube_stats.get("has_data"):
            with col1:
                st.markdown("#### 📺 YouTube")
                st.write(f"**Videos:** {youtube_stats.get('video_count', 0)}")
                st.write(f"**Total Views:** {youtube_stats.get('total_views', 0):,}")

        reddit_stats = sources.get("reddit", {})
        if reddit_stats.get("has_data"):
            with col2:
                st.markdown("#### 🔶 Reddit")
                st.write(f"**Posts:** {reddit_stats.get('post_count', 0)}")
                st.write(f"**Comments:** {reddit_stats.get('comment_count', 0)}")

        twitter_stats = sources.get("twitter", {})
        if twitter_stats.get("has_data"):
            with col3:
                st.markdown("#### 🐦 Twitter")
                st.write(f"**Tweets:** {twitter_stats.get('tweet_count', 0)}")
                st.write(f"**Total Likes:** {twitter_stats.get('total_likes', 0):,}")


# =============================================================================
# FOOTER
# =============================================================================


def render_footer():
    """Render application footer"""
    st.markdown(
        """
    <div class="custom-footer">
        <p>
            <strong>ClarifyProducts.AI</strong> • Powered by Advanced AI & Real-time Data Aggregation<br>
            Made with ❤️ using Streamlit • © 2025 ClarifyProducts.AI
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Main application entry point"""

    # Render sidebar
    render_sidebar()

    # Render header
    render_header()

    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔎 Product Search", "📸 Image Recognition", "💬 AI Assistant"]
    )

    with tab1:
        render_product_search_tab()

    with tab2:
        render_image_recognition_tab()

    with tab3:
        # Render chatbot display (without input - it's below tabs)
        render_chatbot_display()

    # Chat input MUST be outside tabs (Streamlit limitation)
    # It appears at the bottom of the page, always visible
    handle_chat_input()

    # Render footer
    render_footer()


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
