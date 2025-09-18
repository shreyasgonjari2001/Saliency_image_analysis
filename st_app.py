import streamlit as st
import json
import numpy as np

# ==============================
# Load JSON data from temp copy (to avoid conflicts with ongoing writes)
# ==============================
json_file = "updated_analysis_results.json"  # Copy updated_analysis_results.json to this name
with open(json_file, "r") as f:
    data = json.load(f)

st.set_page_config(page_title="Ad Image Analysis Viewer", layout="wide")
st.title("📊 Ad Image Analysis Viewer")

# ==============================
# Collect numeric metrics dynamically
# ==============================
sample_metrics = list(data[0]["analysis"].keys())
numeric_metrics = [
    m for m in sample_metrics if isinstance(data[0]["analysis"][m], (int, float))
]

# ==============================
# Sidebar: Metric Selection (Dropdown Checklist)
# ==============================
st.sidebar.header("Select Metrics to Focus On")
selected_metrics = st.sidebar.multiselect(
    "Metrics", numeric_metrics, default=numeric_metrics
)

# ==============================
# Sidebar: Range Filters with Two Number Inputs (Smoother than Comma Textbox)
# ==============================
st.sidebar.header("Range Filters")
filters = {}
for metric in selected_metrics:
    values = [
        d["analysis"][metric]
        for d in data
        if isinstance(d["analysis"][metric], (int, float))
    ]
    if not values:
        continue
    min_val, max_val = float(np.min(values)), float(np.max(values))
    col1, col2 = st.sidebar.columns(2)
    with col1:
        low = st.number_input(f"{metric} Min", value=min_val, min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
    with col2:
        high = st.number_input(f"{metric} Max", value=max_val, min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
    # Ensure low <= high
    if low > high:
        low, high = high, low
    filters[metric] = (low, high)

# Collect all object labels (only if relevant metric is selected)
object_labels = set()
if "object_attention_scores" in selected_metrics:
    for item in data:
        if item["analysis"].get("object_attention_scores"):
            for obj in item["analysis"]["object_attention_scores"]:
                object_labels.add(obj["label"])
selected_labels = st.sidebar.multiselect(
    "Filter by Object Labels", sorted(list(object_labels))
)

# Sorting option (only for selected metrics)
sort_metric = st.sidebar.selectbox("Sort by metric", ["None"] + selected_metrics)
sort_order = st.sidebar.radio("Sort order", ["Descending", "Ascending"], index=0)

# Page size
page_size = st.sidebar.selectbox("Images per load", [12, 24, 48, 96], index=1)

# Reset filters button
if st.sidebar.button("Apply Filters"):
    st.session_state.page_num = 1
    st.rerun()

# ==============================
# Filtering
# ==============================
filtered = []
for item in data:
    a = item["analysis"]
    # Numeric filters (only for selected metrics)
    valid = True
    for metric, (low, high) in filters.items():
        if not (low <= a.get(metric, 0) <= high):
            valid = False
            break
    if not valid:
        continue
    # Label filter (if applicable)
    if selected_labels and "object_attention_scores" in selected_metrics:
        labels = [obj["label"] for obj in a.get("object_attention_scores", []) or []]
        if not any(lbl in labels for lbl in selected_labels):
            continue
    filtered.append(item)

# Sorting
if sort_metric != "None":
    reverse = sort_order == "Descending"
    filtered = sorted(
        filtered, key=lambda x: x["analysis"].get(sort_metric, 0), reverse=reverse
    )

# ==============================
# Pagination (Load More)
# ==============================
if "page_num" not in st.session_state:
    st.session_state.page_num = 1
start_idx = 0
end_idx = st.session_state.page_num * page_size
paged_items = filtered[start_idx:end_idx]
st.write(f"Showing {len(paged_items)} of {len(filtered)} filtered results")

# ==============================
# Helper: show image with clickable link
# ==============================
def show_image(url, width="100%"):
    st.markdown(
        f'<a href="{url}" target="_blank"><img src="{url}" width="{width}"></a>',
        unsafe_allow_html=True,
    )

# ==============================
# Render grid (only show selected metrics in caption)
# ==============================
cols = st.columns(3)
for i, item in enumerate(paged_items):
    with cols[i % 3]:
        show_image(item["url"])
        st.caption(
            " | ".join(
                f"{m}: {item['analysis'][m]:.2f}"
                for m in selected_metrics
                if isinstance(item["analysis"][m], (int, float))
            )
        )

# ==============================
# Load More button
# ==============================
if len(paged_items) < len(filtered):
    if st.button("⬇️ Load More"):
        st.session_state.page_num += 1
        st.rerun()
