import streamlit as st
import subprocess
import time
import threading
import sys

# Make Streamlit use full width
st.set_page_config(layout="wide")

# Global state for caching nvidia-smi results, shared for all sessions.
class GpuSmiState:
    def __init__(self):
        self.last_update = 0
        self.interval = 10  # seconds
        self.data = ""
        self.lock = threading.Lock()
        self._start_background_thread()

    def _update(self):
        try:
            result = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf-8")
        except Exception as e:
            result = f"Failed to run nvidia-smi: {e}"
        with self.lock:
            self.data = result
            self.last_update = time.time()

    def _background_loop(self):
        while True:
            now = time.time()
            with self.lock:
                last = self.last_update
            if now - last >= self.interval:
                self._update()
            time.sleep(1)

    def _start_background_thread(self):
        t = threading.Thread(target=self._background_loop, daemon=True)
        t.start()

    def get_data(self):
        with self.lock:
            data = self.data
            last = self.last_update
        # In case data is not yet initialized, update immediately
        if time.time() - last > self.interval or not data:
            self._update()
            with self.lock:
                data = self.data
        return data

# Create module-level state (shared across all Streamlit sessions)
GPU_SMI_STATE = GpuSmiState()

st.title("GPU Monitoring: NVIDIA-SMI")
st.write(f"(Refreshed automatically every {GPU_SMI_STATE.interval} seconds for all viewers)")

# Add a placeholder for the code block
code_placeholder = st.empty()

# Autorefresh the display when the value changes or interval elapses
# Using Streamlit's rerun via a loop with st.experimental_rerun is not ideal. Instead, 
# We'll just update the placeholder in a lightweight loop using st.experimental_rerun or a hacky timer widget.

# Use Streamlit's built-in auto-refresh via st.experimental_rerun on a timer
# To avoid infinite rerun loops, wrap in a "while" with sleep is not advisable.
# Instead, use a dummy widget to force reruns as recommended in Streamlit docs.

refresh_interval = GPU_SMI_STATE.interval

# Add a Streamlit 'timer' using st.experimental_singleton to keep a stable session id
def get_session_id():
    import os
    import random
    return f"{os.getpid()}-{random.random()}"

SESSION_ID = get_session_id()
timer_placeholder = st.empty()

# Use a Streamlit hack: session-state-based counter with st.experimental_rerun using time.sleep
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = 0

def maybe_rerun():
    import time as _time
    # Only rerun after the refresh interval
    if _time.time() - st.session_state.last_update_time > refresh_interval:
        st.session_state.last_update_time = _time.time()
        st.experimental_rerun()

# Show output and force refresh
data = GPU_SMI_STATE.get_data()
code_placeholder.code(data, language="bash")

# Dummy widget invisibly triggers rerun after interval
timer_placeholder.markdown(
    f"<script>setTimeout(function(){{window.location.reload();}}, {int(refresh_interval * 1000)});</script>",
    unsafe_allow_html=True,
)

# --- To run on port 20000, suggest usage here if not run that way ---
if __name__ == "__main__" or getattr(sys, "argv", None):
    import os
    # If not running via 'streamlit run ... --server.port 20000', guide user
    port_arg = "--server.port"
    desired_port = "20000"
    # Only check if this is the main script invoked directly, not as Streamlit import
    if not any(port_arg in arg and desired_port in arg for arg in sys.argv):
        st.info(
            "To run this app on port 20000, use:\n"
            "`streamlit run gpu_monitoring_streamlit.py --server.port 20000`"
        )
