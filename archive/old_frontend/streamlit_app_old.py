"""
ClarifyProducts.AI - Streamlit Frontend
Product Review Consensus Platform
"""
import streamlit as st
import requests
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="ClarifyProducts.AI - Review Consensus",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Custom CSS for modern interface
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }

    /* Logo and title */
    .logo-container {
        text-align: center;
        margin-bottom: 3rem;
    }
            

    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 300;
    }

    /* Search box container */
    .search-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }

    /* Result container */
    .result-box {
        background: white;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }

    .consensus-positive {
        color: #10b981;
        font-weight: bold;
    }

    .consensus-negative {
        color: #ef4444;
        font-weight: bold;
    }

    .consensus-neutral {
        color: #f59e0b;
        font-weight: bold;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        width: 100%;
    }

    .stButton > button:hover {
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }

    /* Input fields */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
    }

    /* Feature cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    /* Modern Chatbot Widget Styles */
    .chatbot-widget-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
    }

    .chat-toggle-button {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
        font-size: 28px;
    }

    .chat-toggle-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.7);
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .chatbot-widget-container {
            bottom: 20px;
            right: 20px;
        }

        .chat-toggle-button {
            width: 50px;
            height: 50px;
            font-size: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application"""

    # Logo and Title
    st.markdown("""
    <div class="logo-container">
        <div class="main-title"> ClarifyProducts.AI</div>
        <div class="subtitle">Discover what real people think about products</div>
    </div>
    """, unsafe_allow_html=True)

    # Main search container
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)

        st.markdown("### Enter Product Name")

        # Product name input
        product_name = st.text_input(
            "Enter product name",
            placeholder="e.g., CeraVe Moisturizing Lotion, iPhone 15 Pro, AirPods Pro...",
            label_visibility="collapsed",
            key="product_input"
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            search_button = st.button(" Get Review Consensus", use_container_width=True, type="primary")

        st.markdown('</div>', unsafe_allow_html=True)

    # Process search
    if search_button and product_name:
        get_review_consensus(product_name)

    # Alternative: Image Upload
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown("### Or upload a product image")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Take a photo or upload an image of the product",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    if uploaded_file:
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button(" Identify & Get Reviews", use_container_width=True):
            identify_and_get_reviews(image)

    st.markdown('</div>', unsafe_allow_html=True)

    # Chatbot Widget Button
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown("### Need more help?")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(" Chat with AI Assistant", use_container_width=True, key="chat_toggle"):
            st.session_state.show_chatbot = not st.session_state.get('show_chatbot', False)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Show chatbot modal if requested
    if st.session_state.get('show_chatbot', False):
        show_modern_chatbot()

    # Features Section
    st.markdown("---")
    st.markdown("### Why ClarifyProducts.AI?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered Analysis</div>
            <p>Advanced RAG technology analyzes thousands of reviews to give you accurate consensus</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Image Recognition</div>
            <p>Just snap a photo - our AI identifies the product and fetches reviews instantly</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Ask Anything</div>
            <p>Chat with our AI to get specific answers about product features and concerns</p>
        </div>
        """, unsafe_allow_html=True)


def get_review_consensus(product_name: str):
    """Get review consensus for a product using smart search"""
    with st.spinner(f"Searching for {product_name}..."):
        try:
            # First try smart search (with typo tolerance)
            response = requests.get(
                f"{API_BASE_URL}/smart-search/",
                params={
                    "q": product_name,
                    "use_cache": True,
                    "include_ml": True
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                # Display results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)

                # Show if typo was corrected
                if result.get('suggestions') and result['product_name'].lower() != product_name.lower():
                    st.info(f"💡 Showing results for: **{result['product_name']}** (corrected from '{product_name}')")

                # Product header
                product = result.get('product', {})
                st.markdown(f"## 🔍 {product.get('name', result.get('product_name', 'Unknown'))}")

                # Show product images if available
                if product.get('images'):
                    cols = st.columns(min(len(product['images']), 4))
                    for idx, img in enumerate(product['images'][:4]):
                        with cols[idx]:
                            st.image(img.get('url'), use_column_width=True)

                st.markdown("---")

                # Rating, Review Count, and Recommendation
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    rating = product.get('overall_rating', 0)
                    st.metric("⭐ Rating", f"{rating:.1f}/5.0")
                with col2:
                    review_count = product.get('total_reviews', 0)
                    st.metric("📊 Total Reviews", f"{review_count:,}")
                with col3:
                    recommendation = product.get('recommendation', {})
                    rec_badge = recommendation.get('badge', 'N/A')
                    st.metric("💡 Verdict", rec_badge)
                with col4:
                    cache_status = "🚀 Cached" if result.get('metadata', {}).get('from_cache', False) else "🆕 Fresh"
                    st.metric("Status", cache_status)

                # Show recommendation details
                if recommendation:
                    decision = recommendation.get('decision')
                    confidence = recommendation.get('confidence', 0)
                    reason = recommendation.get('reason', '')

                    if decision == 'buy':
                        st.success(f"✅ **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")
                    elif decision == 'consider':
                        st.info(f"👍 **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")
                    elif decision == 'wait':
                        st.warning(f"⏳ **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")
                    elif decision == 'skip':
                        st.error(f"❌ **{rec_badge}** ({confidence:.0f}% confidence)\n\n{reason}")
                    else:
                        st.info(f"ℹ️ **{rec_badge}**\n\n{reason}")

                # Review Consensus Section
                consensus = result.get('consensus', {})
                if consensus and consensus.get('summary'):
                    st.markdown("---")
                    st.markdown("### 🤖 Review Consensus")

                    # Summary
                    if consensus.get('summary'):
                        st.markdown("#### 📝 What People Are Saying")
                        st.info(consensus['summary'])

                    # Sentiment Breakdown
                    sentiment = consensus.get('sentiment', {})
                    if sentiment:
                        st.markdown("#### 📊 Sentiment Distribution")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("👍 Positive", f"{sentiment.get('positive_percent', 0):.0f}%",
                                     delta=None, delta_color="normal")
                        with col2:
                            st.metric("😐 Neutral", f"{sentiment.get('neutral_percent', 0):.0f}%")
                        with col3:
                            st.metric("👎 Negative", f"{sentiment.get('negative_percent', 0):.0f}%",
                                     delta=None, delta_color="inverse")

                    # Pros and Cons
                    if consensus.get('pros') or consensus.get('cons'):
                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### ✅ Pros")
                            if consensus.get('pros'):
                                for pro in consensus['pros']:
                                    st.write(f"• {pro}")
                            else:
                                st.write("_No specific pros extracted_")

                        with col2:
                            st.markdown("#### ❌ Cons")
                            if consensus.get('cons'):
                                for con in consensus['cons']:
                                    st.write(f"• {con}")
                            else:
                                st.write("_No specific cons extracted_")

                # YouTube Review Videos
                review_videos = result.get('review_videos', [])
                if review_videos:
                    st.markdown("---")
                    st.markdown("### 🎥 YouTube Review Videos")

                    for video in review_videos[:3]:  # Show top 3
                        with st.expander(f"🎬 {video.get('title', 'Video')} - {video.get('channel', 'Unknown')}"):
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                if video.get('thumbnail'):
                                    st.image(video['thumbnail'], use_column_width=True)

                            with col2:
                                st.markdown(f"**Channel:** {video.get('channel', 'Unknown')}")
                                st.markdown(f"**Views:** {video.get('views', 0):,}")
                                st.markdown(f"**URL:** [{video.get('url', '')}]({video.get('url', '')})")
                                if video.get('has_transcript'):
                                    st.success("✅ Has transcript")
                                else:
                                    st.info("ℹ️ No transcript available")

                # Show source statistics
                sources = result.get('sources', {})
                if sources:
                    st.markdown("---")
                    st.markdown("### 📚 Review Sources")

                    col1, col2, col3 = st.columns(3)

                    # YouTube stats
                    youtube_stats = sources.get('youtube', {})
                    if youtube_stats.get('has_data'):
                        with col1:
                            st.markdown("#### Youtube")
                            st.write(f"**Videos:** {youtube_stats.get('video_count', 0)}")
                            st.write(f"**Total Views:** {youtube_stats.get('total_views', 0):,}")

                    # Reddit stats
                    reddit_stats = sources.get('reddit', {})
                    if reddit_stats.get('has_data'):
                        with col2:
                            st.markdown("#### Reddit")
                            st.write(f"**Posts:** {reddit_stats.get('post_count', 0)}")
                            st.write(f"**Comments:** {reddit_stats.get('comment_count', 0)}")

                    # Twitter stats
                    twitter_stats = sources.get('twitter', {})
                    if twitter_stats.get('has_data'):
                        with col3:
                            st.markdown("#### Twitter")
                            st.write(f"**Tweets:** {twitter_stats.get('tweet_count', 0)}")
                            st.write(f"**Total Likes:** {twitter_stats.get('total_likes', 0):,}")

                # Alternative suggestions
                if result.get('suggestions') and len(result['suggestions']) > 1:
                    st.markdown("---")
                    st.markdown("### 🔄 Related Searches")
                    suggestions_str = " • ".join([f"**{s}**" for s in result['suggestions'][:5]])
                    st.markdown(suggestions_str)

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.error(f"❌ Search failed: {response.status_code}")
                st.info("💡 Falling back to RAG search...")
                # Fallback to original RAG-based search
                fallback_rag_search(product_name)

        except requests.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Make sure the backend server is running at http://localhost:8000")


def fallback_rag_search(product_name: str):
    """Fallback to RAG-based search if smart search fails"""
    try:
        query = f"What is the review consensus about {product_name}?"
        payload = {"query": query, "n_results": 10}

        response = requests.post(
            f"{API_BASE_URL}/rag/query",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f"## 📊 Review Consensus: {product_name}")
            st.markdown("---")
            st.markdown("### What People Are Saying:")
            st.markdown(result['response'])

            if result.get('sources') and len(result['sources']) > 0:
                st.markdown("---")
                st.markdown(f"### 📚 Based on {result['context_count']} Reviews")
                for source in result['sources'][:3]:
                    rating = source['metadata'].get('rating', 'N/A')
                    source_name = source['metadata'].get('source', 'Unknown')
                    with st.expander(f"⭐ {rating}/5.0 - {source_name}"):
                        st.write(source['text'])

            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"RAG search also failed: {str(e)}")


def identify_and_get_reviews(image: Image.Image):
    """Identify product from image and get reviews using Complete Pipeline (CLIP + OCR + Gemini)"""
    with st.spinner("🔍 Analyzing image (AI Vision + Text Recognition)..."):
        try:
            # Prepare image
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            files = {"image": ("image.png", img_byte_arr, "image/png")}

            response = requests.post(
                f"{API_BASE_URL}/recognition/",
                files=files,
                timeout=60  # Increased timeout for complete pipeline
            )

            if response.status_code == 200:
                result = response.json()

                # Check if recognition was successful
                if result.get('success'):
                    primary_match = result.get('primary_match', {})
                    product_label = primary_match.get('product_name', 'unknown product')
                    confidence = primary_match.get('confidence', 0)
                    category = primary_match.get('category', 'unknown')
                    message = result.get('message', '')

                    # Display result with enhanced UI
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Determine confidence emoji
                        if confidence >= 0.8:
                            emoji = "✅"
                            conf_color = "green"
                        elif confidence >= 0.6:
                            emoji = "🟡"
                            conf_color = "orange"
                        else:
                            emoji = "⚠️"
                            conf_color = "red"

                        st.markdown(f"### {emoji} **{product_label.title()}**")
                        st.markdown(f"**Category:** {category.capitalize()}")

                    with col2:
                        # Confidence meter
                        st.metric("Confidence", f"{confidence*100:.0f}%")

                    # Show method and details
                    if "multimodal" in message.lower():
                        st.info("💡 " + message)
                    elif "generic category" in message.lower():
                        st.warning("⚠️ " + message)
                    elif message:
                        st.info(message)

                    # Show alternative matches if available
                    alternatives = result.get('alternative_matches', [])
                    if alternatives:
                        with st.expander("View alternative matches"):
                            for alt in alternatives[:3]:
                                st.write(f"• {alt['product_name']}: {alt['confidence']*100:.1f}%")

                    # Automatically get reviews for this product (same as text input)
                    product_name = product_label
                    st.markdown("---")
                    st.info(f"🔍 Fetching reviews for **{product_name.title()}**...")
                    get_review_consensus(product_name)
                else:
                    # Recognition failed
                    error_message = result.get('message', 'Recognition failed')
                    st.error(f"❌ {error_message}")

                    # Show helpful tips
                    with st.expander("💡 Tips for better recognition"):
                        st.markdown("""
                        - Ensure good lighting
                        - Capture product label clearly
                        - Avoid blurry images
                        - For products without text (shoes, electronics), upload images showing logos/distinctive features
                        """)
            else:
                st.error(f"❌ Recognition failed: {response.status_code}")

        except requests.exceptions.Timeout:
            st.error("⏱️ Recognition timed out. Please try again with a clearer image.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def show_modern_chatbot():
    """Display modern production-style chatbot interface"""
    st.markdown("---")
    st.markdown('<div class="result-box" style="padding: 0;">', unsafe_allow_html=True)

    # Header with close button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("## 💬 AI Product Assistant")
    with col2:
        if st.button("✕", key="close_chat", help="Close chat"):
            st.session_state.show_chatbot = False
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome message with suggested prompts if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px;">
            <h3 style="color: #667eea; margin-bottom: 10px;">👋 Hi! How can I help you today?</h3>
            <p style="color: #666; margin-bottom: 20px;">I can help you understand products, compare features, and analyze reviews</p>
        </div>
        """, unsafe_allow_html=True)

    # Display chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources", expanded=False):
                        for idx, source in enumerate(message["sources"][:3], 1):
                            metadata = source.get('metadata', {})
                            st.markdown(f"""
                            **Source {idx}:** {metadata.get('source', 'Unknown')}
                            **Rating:** ⭐ {metadata.get('rating', 'N/A')}/5
                            **Content:** {source.get('text', '')[:200]}...
                            """)

    # Chat input
    if prompt := st.chat_input("Ask me anything about products...", key="chat_input_main"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching through reviews and analyzing..."):
                try:
                    payload = {
                        "query": prompt,
                        "n_results": 5
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/rag/query",
                        json=payload,
                        timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result['response']

                        st.markdown(ai_response)

                        # Store message with sources
                        message_data = {
                            "role": "assistant",
                            "content": ai_response
                        }
                        if result.get('sources'):
                            message_data["sources"] = result['sources']

                        st.session_state.messages.append(message_data)

                        # Show context info
                        if result.get('context_count', 0) > 0:
                            st.caption(f"✅ Answer based on {result['context_count']} real-time reviews")

                        # Show sources inline
                        if result.get('sources'):
                            with st.expander("📚 View Sources", expanded=False):
                                for idx, source in enumerate(result['sources'][:3], 1):
                                    metadata = source.get('metadata', {})
                                    st.markdown(f"""
                                    **Source {idx}:** {metadata.get('source', 'Unknown')}
                                    **Rating:** ⭐ {metadata.get('rating', 'N/A')}/5
                                    **Content:** {source.get('text', '')[:200]}...
                                    """)
                    else:
                        error_msg = "I apologize, but I couldn't process that request. Please try rephrasing your question or ask about a specific product."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except Exception as e:
                    error_msg = f"Connection error: {str(e)}. Please try again."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Chat actions
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
