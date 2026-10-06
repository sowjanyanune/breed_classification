import json
import os
import time
import textwrap
import streamlit as st
from PIL import Image

from utils.predictor import predict_breed_details, class_names

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CanineVision • AI Dog Breed Classifier",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero-container {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(236, 72, 153, 0.1) 100%);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 24px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 12px 36px -10px rgba(99, 102, 241, 0.2);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: #ffffff;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}

.hero-title {
    font-size: 2.5rem;
    line-height: 1.15;
    font-weight: 800;
    background: linear-gradient(135deg, #FFFFFF 30%, #C4B5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    max-width: 720px;
    line-height: 1.6;
    margin: 0;
}

.custom-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
    margin-bottom: 1.2rem;
}

.prediction-hero-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1.5px solid rgba(139, 92, 246, 0.4);
    border-radius: 24px;
    padding: 1.8rem;
    box-shadow: 0 15px 35px -5px rgba(99, 102, 241, 0.25);
    margin-bottom: 1.5rem;
}

.breed-title {
    font-size: 2rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.breed-tagline {
    font-size: 0.95rem;
    color: #A78BFA;
    font-weight: 500;
    margin-top: 4px;
    margin-bottom: 1.2rem;
}

.confidence-meter-bg {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    height: 14px;
    width: 100%;
    overflow: hidden;
    margin: 10px 0 4px 0;
    position: relative;
}

.confidence-meter-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6, #10B981);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
}

.trait-chip {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3);
    color: #DDD6FE;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
    margin-right: 6px;
    margin-bottom: 6px;
}

.info-tile {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    min-height: 95px;
}

.info-tile-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    font-weight: 700;
    margin-bottom: 4px;
}

.info-tile-val {
    font-size: 0.95rem;
    color: #F1F5F9;
    font-weight: 600;
    line-height: 1.4;
}

.fun-fact-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
}

.fun-fact-title {
    color: #FBBF24;
    font-weight: 700;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}

.fun-fact-text {
    color: #E2E8F0;
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    font-size: 0.95rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Breed Information
# ---------------------------------------------------------
@st.cache_data
def load_breed_database():
    with open("data/breed_info.json", "r", encoding="utf-8") as f:
        return json.load(f)

breed_db = load_breed_database()


# ---------------------------------------------------------
# Sidebar: Breed Encyclopedia & System Specs
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🐾 CanineVision AI")
    st.caption("Deep Learning Dog Breed Classifier")
    st.divider()

    st.markdown("#### 📖 Breed Encyclopedia")
    st.write("Explore all 10 supported dog breeds and their characteristics:")

    all_breed_keys = list(breed_db.keys())
    selected_explore_breed = st.selectbox(
        "Select a breed to inspect",
        options=all_breed_keys,
        format_func=lambda x: f"{breed_db[x].get('emoji', '🐶')} {breed_db[x].get('name', x.replace('_', ' '))}"
    )

    if selected_explore_breed:
        info_sidebar = breed_db[selected_explore_breed]
        with st.expander(f"ℹ️ {info_sidebar.get('name', selected_explore_breed)} Details", expanded=True):
            st.markdown(f"*{info_sidebar.get('tagline', '')}*")
            st.write(info_sidebar.get("description", ""))
            st.markdown(f"**Size:** {info_sidebar.get('size', 'N/A')}")
            st.markdown(f"**Temperament:** {info_sidebar.get('temperament', 'N/A')}")
            st.markdown(f"**Life Span:** {info_sidebar.get('life_span', 'N/A')}")
            st.markdown(f"**Energy Level:** {info_sidebar.get('energy_level', 'N/A')}")

    st.divider()
    st.markdown("#### ⚙️ Model Information")
    st.caption("• **Architecture:** MobileNetV2 (Transfer Learning)")
    st.caption("• **Input Resolution:** 224 × 224 pixels")
    st.caption("• **Inference Speed:** Real-time (< 200ms)")
    st.caption("• **Classes Supported:** 10 Canine Breeds")

    st.divider()
    st.caption("Built with TensorFlow & Streamlit")


# ---------------------------------------------------------
# Main Page Hero Banner
# ---------------------------------------------------------
hero_html = textwrap.dedent("""
<div class="hero-container">
<div class="hero-badge">⚡ Powered by MobileNetV2 Deep Learning</div>
<h1 class="hero-title">Dog Breed Identification</h1>
<p class="hero-subtitle">
Upload a photo, capture a live snapshot, or test with curated samples.
Our neural network analyzes distinctive facial features and fur patterns to predict the exact breed with deep personality insights.
</p>
</div>
""")
st.markdown(hero_html, unsafe_allow_html=True)


# ---------------------------------------------------------
# Input Methods: Tabs (Upload, Camera, Samples)
# ---------------------------------------------------------
tab_upload, tab_camera, tab_samples = st.tabs([
    "📁 Upload Image",
    "📸 Live Camera Capture",
    "✨ Instant Sample Showcase"
])

selected_image = None
image_source_name = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a dog picture (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="dog_uploader",
        help="For best accuracy, upload a clear, well-lit photo showing the dog's face and body."
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file)
            image_source_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading image: {e}")

with tab_camera:
    camera_photo = st.camera_input("Take a photo of your dog")
    if camera_photo is not None:
        try:
            selected_image = Image.open(camera_photo)
            image_source_name = "Camera Snapshot"
        except Exception as e:
            st.error(f"Error loading camera snapshot: {e}")

with tab_samples:
    st.markdown("Select one of the sample images below for an instant demonstration:")
    sample_cols = st.columns(4)

    sample_items = [
        {"name": "Golden Retriever", "file": "assets/samples/golden_retriever.jpg", "key": "sample_gr"},
        {"name": "Siberian Husky", "file": "assets/samples/husky.jpg", "key": "sample_husky"},
        {"name": "Pug", "file": "assets/samples/pug.jpg", "key": "sample_pug"},
        {"name": "Beagle", "file": "assets/samples/beagle.jpg", "key": "sample_beagle"},
    ]

    for idx, item in enumerate(sample_items):
        with sample_cols[idx]:
            if os.path.exists(item["file"]):
                thumb = Image.open(item["file"])
                st.image(thumb, use_container_width=True)
                if st.button(f"Test {item['name']}", key=item["key"]):
                    st.session_state["active_sample"] = item["file"]
                    st.session_state["active_sample_name"] = item["name"]

    if "active_sample" in st.session_state and os.path.exists(st.session_state["active_sample"]):
        selected_image = Image.open(st.session_state["active_sample"])
        image_source_name = f"Sample: {st.session_state.get('active_sample_name', 'Dog')}"


# ---------------------------------------------------------
# Image Processing & Prediction Section
# ---------------------------------------------------------
if selected_image is not None:
    st.markdown("---")

    col_img, col_pred = st.columns([1, 1.3], gap="large")

    with col_img:
        st.markdown("### 🖼️ Input Image")
        st.image(
            selected_image,
            use_container_width=True,
            caption=f"Source: {image_source_name or 'Uploaded Image'}"
        )
        st.caption(f"Dimensions: {selected_image.width} × {selected_image.height} px | Mode: {selected_image.mode}")

    with col_pred:
        st.markdown("### 🔍 Classification Analysis")

        # Perform inference with progress spinner
        with st.spinner("🧠 Analyzing canine traits & running neural inference..."):
            t_start = time.time()
            top_breed, top_confidence, top_k_list, all_preds = predict_breed_details(selected_image, top_k=4)
            inference_time_ms = int((time.time() - t_start) * 1000)

        # Retrieve breed data
        breed_data = breed_db.get(top_breed, {
            "name": top_breed.replace("_", " "),
            "emoji": "🐶",
            "tagline": "Distinctive canine breed",
            "description": "Information not available for this breed.",
            "size": "N/A",
            "temperament": "N/A",
            "origin": "N/A",
            "life_span": "N/A",
            "energy_level": "N/A",
            "good_with": "N/A",
            "care_tips": "N/A",
            "fun_fact": "Every dog is unique and special!",
            "traits": ["Loyal", "Friendly"]
        })

        # Confidence status color & badge
        if top_confidence >= 80:
            status_color = "#10B981"
            status_text = "High Confidence"
        elif top_confidence >= 50:
            status_color = "#F59E0B"
            status_text = "Moderate Confidence"
        else:
            status_color = "#EF4444"
            status_text = "Low Confidence / Ambiguous"

        # Top Result Hero Card (written with textwrap.dedent to prevent markdown code block parsing)
        display_breed_name = breed_data.get("name", top_breed.replace("_", " "))
        emoji_icon = breed_data.get("emoji", "🐶")
        tagline = breed_data.get("tagline", "")
        meter_fill_pct = min(max(top_confidence, 5), 100)

        card_html = textwrap.dedent(f"""
<div class="prediction-hero-card">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
<div class="breed-title">
<span>{emoji_icon}</span> {display_breed_name}
</div>
<div style="background: rgba(255,255,255,0.1); border: 1px solid {status_color}; color: {status_color}; padding: 4px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;">
{status_text} • {inference_time_ms}ms
</div>
</div>
<div class="breed-tagline">{tagline}</div>
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; color: #E2E8F0; font-weight: 600;">
<span>Match Probability</span>
<span style="color: {status_color}; font-size: 1.25rem; font-weight: 800;">{top_confidence:.2f}%</span>
</div>
<div class="confidence-meter-bg">
<div class="confidence-meter-fill" style="width: {meter_fill_pct}%;"></div>
</div>
</div>
""")
        st.markdown(card_html, unsafe_allow_html=True)

        # Top 4 candidate distribution
        st.markdown("#### 📊 Candidate Distribution (Top 4)")
        for cand in top_k_list:
            c_name = cand["display_name"]
            c_conf = cand["confidence"]
            cand_info = breed_db.get(cand["breed"], {})
            cand_emoji = cand_info.get("emoji", "🐾")

            col_name, col_bar, col_pct = st.columns([1.6, 3, 0.9])
            with col_name:
                st.markdown(f"**{cand_emoji} {c_name}**")
            with col_bar:
                st.progress(float(c_conf / 100.0))
            with col_pct:
                st.markdown(f"`{c_conf:.1f}%`")

    # ---------------------------------------------------------
    # Comprehensive Breed Profile Cards
    # ---------------------------------------------------------
    st.markdown("### 🐕 Comprehensive Breed Profile")

    # Traits pill chips
    traits = breed_data.get("traits", [])
    if traits:
        chips_html = "".join([f'<span class="trait-chip">✦ {t}</span>' for t in traits])
        st.markdown(f'<div style="margin-bottom: 1rem;">{chips_html}</div>', unsafe_allow_html=True)

    # Overview description
    st.write(breed_data.get("description", ""))

    # 4-Tile Characteristic Grid
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(textwrap.dedent(f"""
<div class="info-tile">
<div class="info-tile-label">📏 Size & Build</div>
<div class="info-tile-val">{breed_data.get('size', 'N/A')}</div>
</div>
"""), unsafe_allow_html=True)

    with c2:
        st.markdown(textwrap.dedent(f"""
<div class="info-tile">
<div class="info-tile-label">🧠 Temperament</div>
<div class="info-tile-val">{breed_data.get('temperament', 'N/A')}</div>
</div>
"""), unsafe_allow_html=True)

    with c3:
        st.markdown(textwrap.dedent(f"""
<div class="info-tile">
<div class="info-tile-label">⚡ Energy & Exercise</div>
<div class="info-tile-val">{breed_data.get('energy_level', 'N/A')}</div>
</div>
"""), unsafe_allow_html=True)

    with c4:
        st.markdown(textwrap.dedent(f"""
<div class="info-tile">
<div class="info-tile-label">🌍 Origin & Lifespan</div>
<div class="info-tile-val">{breed_data.get('origin', 'N/A')}<br><small style="color:#94A3B8;">Life: {breed_data.get('life_span', 'N/A')}</small></div>
</div>
"""), unsafe_allow_html=True)

    # Additional Insights: Care tips & Fun Fact
    col_care, col_fact = st.columns([1, 1])

    with col_care:
        care_html = textwrap.dedent(f"""
<div class="custom-card" style="margin-top: 1.2rem; height: 100%;">
<div style="font-weight: 700; color: #A78BFA; font-size: 0.95rem; margin-bottom: 6px;">
✂️ Grooming & Care Advice
</div>
<p style="color: #E2E8F0; font-size: 0.9rem; line-height: 1.5; margin: 0;">
{breed_data.get('care_tips', 'Provide regular exercise, healthy nutrition, and routine vet checkups.')}
</p>
<div style="margin-top: 8px; font-size: 0.85rem; color: #94A3B8;">
<strong>Best Suited For:</strong> {breed_data.get('good_with', 'Loving households')}
</div>
</div>
""")
        st.markdown(care_html, unsafe_allow_html=True)

    with col_fact:
        fact_html = textwrap.dedent(f"""
<div class="fun-fact-box" style="margin-top: 1.2rem; height: 100%;">
<div class="fun-fact-title">
💡 Did You Know? (Fun Fact)
</div>
<p class="fun-fact-text">
{breed_data.get('fun_fact', 'Dogs have a sense of time and miss you when you are away!')}
</p>
</div>
""")
        st.markdown(fact_html, unsafe_allow_html=True)

else:
    # Empty State Welcome Card
    empty_html = textwrap.dedent("""
<div class="custom-card" style="text-align: center; padding: 3rem 2rem; border: 2px dashed rgba(168, 85, 247, 0.3);">
<div style="font-size: 3.5rem; margin-bottom: 0.8rem;">🐕✨</div>
<h3 style="color: #F8FAFC; margin-bottom: 0.4rem;">Ready to Identify a Breed</h3>
<p style="color: #94A3B8; max-width: 500px; margin: 0 auto 1.5rem auto; font-size: 0.95rem;">
Upload an image above, take a snapshot using your webcam, or pick one of our instant sample dogs in the <b>"Instant Sample Showcase"</b> tab!
</p>
<div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
<span class="trait-chip">🐶 10 Dog Breeds</span>
<span class="trait-chip">⚡ Instant Predictions</span>
<span class="trait-chip">📊 Top-4 Probability Breakdown</span>
<span class="trait-chip">📖 Full Care Guides</span>
</div>
</div>
""")
    st.markdown(empty_html, unsafe_allow_html=True)