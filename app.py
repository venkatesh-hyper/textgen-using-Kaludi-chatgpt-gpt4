import streamlit as st
from gpt_model import generate_text

# Custom CSS for a dynamic gradient background
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Dynamic background animation */
        body {
            background: linear-gradient(270deg, #6C63FF, #FF6C63, #63FFC6);
            background-size: 600% 600%;
            animation: gradientAnimation 15s ease infinite;
        }
        
        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Input box styling */
        .stTextInput > div > input {
            background-color: #FFFFFF;
            border: 2px solid #6C63FF;
            color: #333333;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #6C63FF;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        
        .stButton > button:hover {
            background-color: #5548c8;
            transition: 0.3s;
        }
        
        /* Generated text styling */
        .generated-text {
            background-color: #FFFFFF;
            padding: 20px;
            border: 2px solid #6C63FF;
            border-radius: 10px;
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to add CSS
add_custom_css()

# Streamlit app layout
st.title("✨ GPT-2 Text Generator ✨")

# Add description with inline styling
st.markdown(
    """
    <div style="text-align: center; font-size: 18px; color: #FFFFFF;">
    Enter a prompt, and GPT-2 will generate creative text for you! 🎉
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr style='border: 1px solid #FFFFFF;'>", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([3, 1])

# Text input from the user in the left column
with col1:
    prompt = st.text_input("💬 Input your prompt", "Once upon a time", help="Enter the text you want GPT-2 to continue")

# Slider to select maximum length of generated text in the right column
with col2:
    max_length = st.slider("✏️ Max length", min_value=100, max_value=1000, value=300, help="Maximum number of tokens for GPT-2 to generate")

# Add a generate button with colorful styling
if st.button("🚀 Generate Text"):
    if prompt:
        with st.spinner("✨ Generating text..."):
            generated_text = generate_text(prompt, max_length=max_length)
            st.subheader("📝 Generated Text:")
            
            # Display the generated text in a styled container
            st.markdown(
                f"""
                <div class="generated-text">{generated_text}</div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter a prompt to generate text!")

# Option to clear inputs
if st.button("🧹 Clear Prompt"):
    st.experimental_rerun()
