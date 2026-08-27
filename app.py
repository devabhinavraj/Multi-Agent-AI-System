"""
Streamlit UI for the Multi-Agent AI Research System.

This wraps pipeline.research_pipeline() — which runs the
search agent -> reader agent -> writer chain -> critic chain
pipeline — in a simple web interface.

Two separate approval steps are surfaced entirely inside Streamlit
(no terminal / VS Code interaction needed):

1. Before the pipeline starts at all — a "Confirm agent run" card.
2. Before each individual tool call the search agent makes (this
   replaces the input() prompt in tools.human_approval) — an
   "Approve / Deny" card that appears mid-run.

Run with:
    streamlit run app.py
"""

import threading
import time
import traceback

import streamlit as st

from pipeline.pipeline import research_pipeline
from tools import tools as tools_module


# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)


# ----------------------------------------------------------------------
# Approval broker
# ----------------------------------------------------------------------
class ApprovalBroker:
    """Thread-safe bridge between the background pipeline thread (which
    wants human approval before a tool call) and the Streamlit UI thread
    (which shows Approve/Deny buttons instead of blocking on input()).

    The pipeline thread calls request_approval() and blocks. The
    Streamlit thread polls .pending_tool every rerun, renders buttons
    when it's set, and calls submit_decision() when the user clicks —
    which unblocks the pipeline thread.
    """

    def __init__(self):
        self._decision_event = threading.Event()
        self.pending_tool = None
        self.request_id = 0
        self._decision = None

    def request_approval(self, tool_name: str) -> bool:
        self.request_id += 1
        self.pending_tool = tool_name
        self._decision_event.clear()
        self._decision_event.wait()
        self.pending_tool = None
        return bool(self._decision)

    def submit_decision(self, approved: bool) -> None:
        self._decision = approved
        self._decision_event.set()


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
_DEFAULTS = {
    "history": [],
    "pending_topic": None,   # topic waiting on the pre-run confirmation
    "run_thread": None,      # background Thread while a pipeline run is active
    "run_broker": None,      # ApprovalBroker for the active run
    "run_topic": None,       # topic of the active run
    "run_result_box": None,  # dict the worker thread writes {"state": ...} into
    "run_error_box": None,   # dict the worker thread writes {"error", "trace"} into
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Research System")
    st.markdown(
        "A 4-agent pipeline:\n"
        "1. **Search Agent** — finds recent, reliable sources\n"
        "2. **Reader Agent** — scrapes the best URL for detail\n"
        "3. **Writer Chain** — drafts the final report\n"
        "4. **Critic Chain** — reviews and gives feedback\n\n"
        "Every tool call the search agent makes needs your approval — "
        "you'll see an Approve/Deny card right here, no terminal needed."
    )
    st.divider()

    if st.session_state.history:
        st.subheader("Past Runs")
        topic_labels = [f"{i + 1}. {h['topic']}" for i, h in enumerate(st.session_state.history)]
        selected_label = st.radio(
            "Select a run to view",
            options=["Current"] + topic_labels,
            index=0,
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        selected_label = "Current"
        st.caption("No past runs yet.")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _generate_pdf_bytes(title: str, body: str) -> bytes:
    """Render a simple one-column PDF from plain text. Requires fpdf2
    (`pip install fpdf2` or `uv add fpdf2`)."""
    from fpdf import FPDF  # noqa: F401  (import here so app still runs without it)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    safe_title = title.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 10, safe_title)
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    # Core PDF fonts only support latin-1, so replace unsupported
    # characters (curly quotes, emoji, etc.) rather than crashing.
    safe_body = body.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 7, safe_body)

    raw = pdf.output()
    if isinstance(raw, str):
        raw = raw.encode("latin-1", "replace")
    return bytes(raw)


def _download_row(label_prefix: str, content: str, file_stub: str, unique_key: str) -> None:
    """Show side-by-side .txt and .pdf download buttons for a block of text."""
    col_txt, col_pdf = st.columns(2)
    with col_txt:
        st.download_button(
            f"⬇️ {label_prefix} (.txt)",
            data=content,
            file_name=f"{file_stub}.txt",
            mime="text/plain",
            key=f"dl-txt-{unique_key}",
            use_container_width=True,
        )
    with col_pdf:
        try:
            pdf_bytes = _generate_pdf_bytes(file_stub.replace("_", " "), content)
            st.download_button(
                f"⬇️ {label_prefix} (.pdf)",
                data=pdf_bytes,
                file_name=f"{file_stub}.pdf",
                mime="application/pdf",
                key=f"dl-pdf-{unique_key}",
                use_container_width=True,
            )
        except ImportError:
            st.caption("Run `uv add fpdf2` (or `pip install fpdf2`) to enable PDF downloads.")


def render_state(topic: str, state: dict) -> None:
    """Render the four pipeline outputs in tabs."""
    st.subheader(f"Results for: *{topic}*")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "📖 Scraped Content"]
    )

    stub = topic.strip().replace(" ", "_")

    with tab_report:
        report = state.get("report", "")
        st.markdown(report if report else "_No report generated._")
        if report:
            _download_row("Download report", report, f"{stub}_report", f"{topic}-{id(state)}-report")

    with tab_feedback:
        feedback = state.get("feedback", "")
        st.markdown(feedback if feedback else "_No feedback generated._")
        if feedback:
            _download_row("Download feedback", feedback, f"{stub}_feedback", f"{topic}-{id(state)}-feedback")

    with tab_search:
        st.text(state.get("search_result", "_No search results._"))

    with tab_scraped:
        st.text(state.get("scraped_content", "_No scraped content._"))


def start_pipeline(topic: str) -> None:
    """Kick off research_pipeline() in a background thread and wire its
    tool-approval prompts to a fresh ApprovalBroker instead of input()."""
    broker = ApprovalBroker()
    # Swap out the terminal-input approval handler for the broker's version.
    # This only affects the current process, so the plain `python
    # pipeline/pipeline.py` CLI usage still uses input() as before.
    tools_module.approval_handler = broker.request_approval

    result_box = {}
    error_box = {}

    def _worker():
        try:
            result_box["state"] = research_pipeline(topic)
        except Exception as exc:  # noqa: BLE001
            error_box["error"] = exc
            error_box["trace"] = traceback.format_exc()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    st.session_state.run_thread = thread
    st.session_state.run_broker = broker
    st.session_state.run_topic = topic
    st.session_state.run_result_box = result_box
    st.session_state.run_error_box = error_box


def busy() -> bool:
    return st.session_state.run_thread is not None


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("Multi-Agent AI Research System")
st.caption("Enter a topic and let the search → reader → writer → critic pipeline research it for you.")

with st.form("research_form", clear_on_submit=False):
    topic_input = st.text_input(
        "Research topic",
        placeholder="e.g. Latest advances in solid-state batteries",
    )
    submitted = st.form_submit_button(
        "🚀 Start Research",
        use_container_width=True,
        disabled=busy() or st.session_state.pending_topic is not None,
    )

if submitted:
    if not topic_input or not topic_input.strip():
        st.warning("Please enter a topic before starting.")
    else:
        # Don't call the agents yet — ask for confirmation first.
        st.session_state.pending_topic = topic_input.strip()

# ----------------------------------------------------------------------
# Step 1 approval: confirm running the pipeline at all
# ----------------------------------------------------------------------
if st.session_state.pending_topic:
    st.markdown("###")
    with st.container(border=True):
        st.markdown("### ⚠️ Confirm agent run")
        st.write(
            f"This will call the **search**, **reader**, **writer**, and **critic** "
            f"agents for the topic:\n\n> **{st.session_state.pending_topic}**\n\n"
            f"This may use API credits and take a minute or two. Proceed?"
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            confirm_yes = st.button(
                "✅ Yes, run the agents", use_container_width=True, type="primary"
            )
        with col_no:
            confirm_no = st.button("❌ No, cancel", use_container_width=True)

    if confirm_yes:
        topic_to_run = st.session_state.pending_topic
        st.session_state.pending_topic = None
        start_pipeline(topic_to_run)
        st.rerun()
    elif confirm_no:
        st.session_state.pending_topic = None
        st.info("Cancelled — no agents were called.")

st.divider()

# ----------------------------------------------------------------------
# Step 2 approval: per-tool-call approval while the pipeline is running
# ----------------------------------------------------------------------
if busy():
    thread = st.session_state.run_thread
    broker = st.session_state.run_broker

    if broker.pending_tool is not None:
        with st.container(border=True):
            st.markdown("### 🛠️ Tool call needs approval")
            st.write(
                f"The search agent wants to call **`{broker.pending_tool}`** "
                f"for topic *{st.session_state.run_topic}*. Approve?"
            )
            col_yes, col_no = st.columns(2)
            key_suffix = broker.request_id
            with col_yes:
                if st.button(
                    "✅ Approve", key=f"tool-yes-{key_suffix}",
                    use_container_width=True, type="primary",
                ):
                    broker.submit_decision(True)
                    st.rerun()
            with col_no:
                if st.button(
                    "❌ Deny", key=f"tool-no-{key_suffix}",
                    use_container_width=True,
                ):
                    broker.submit_decision(False)
                    st.rerun()

    elif thread.is_alive():
        st.info(
            f"⏳ Researching **{st.session_state.run_topic}**... "
            f"this page refreshes automatically."
        )
        time.sleep(1)
        st.rerun()

    else:
        # Thread finished — collect the result and clear run state.
        thread.join()
        error_box = st.session_state.run_error_box
        result_box = st.session_state.run_result_box
        topic_done = st.session_state.run_topic

        if error_box and "error" in error_box:
            st.error(f"Pipeline failed: {error_box['error']}")
            with st.expander("Show traceback"):
                st.code(error_box["trace"])
        else:
            st.success("Research complete ✅")
            st.session_state.history.insert(0, {"topic": topic_done, "state": result_box["state"]})

        st.session_state.run_thread = None
        st.session_state.run_broker = None
        st.session_state.run_topic = None
        st.session_state.run_result_box = None
        st.session_state.run_error_box = None

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
if selected_label != "Current" and st.session_state.history:
    idx = int(selected_label.split(".")[0]) - 1
    entry = st.session_state.history[idx]
    render_state(entry["topic"], entry["state"])
elif st.session_state.history:
    entry = st.session_state.history[0]
    render_state(entry["topic"], entry["state"])
else:
    st.info("No research run yet. Enter a topic above and click **Start Research**.")