import streamlit as st
import subprocess
import time

# Make Streamlit use full width
st.set_page_config(layout="wide")

st.title("GPU Monitoring: NVIDIA-SMI")

# Add refresh button
if st.button("🔄 Refresh GPU Stats", type="primary"):
    st.rerun()

# Get current GPU stats
try:
    result = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf-8")
except Exception as e:
    result = f"Failed to run nvidia-smi: {e}"

# Display timestamp
st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Show output
st.code(result, language="bash")
