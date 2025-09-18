import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. Load data ----------
try:
    with open("updated_analysis_results.json", "r") as f:  # Adjust filename if needed
        records = json.load(f)
except FileNotFoundError:
    st.error("JSON file not found. Please check the path.")
    st.stop()

# Flatten to DataFrame
df_records = []
for rec in records:
    flat = rec.get("analysis", {}).copy()  # Safely get analysis dict
    flat["url"] = rec.get("url", "N/A")  # Safely get url
    flat["ad_id"] = rec.get("ad_id", None)  # Safely get ad_id from root level
    flat["cluster"] = rec.get("cluster", None)  # Safely get cluster from root level
    df_records.append(flat)

df = pd.DataFrame(df_records)

# Load final_scores.json and merge final_score into df
try:
    with open("final_scores.json", "r") as f:
        final_scores = json.load(f)
    score_dict = {item['ad_id']: item.get('score') for item in final_scores if 'ad_id' in item}
    df['final_score'] = df['ad_id'].map(score_dict).fillna('N/A')  # Fill missing with 'N/A'
except FileNotFoundError:
    df['final_score'] = 'N/A'  # Fallback if file missing
    st.warning("final_scores.json not found. Final scores will show as N/A.")

# Debugging: Show columns in sidebar
with st.sidebar.expander("Data Columns (Debug)"):
    st.write("Columns:", list(df.columns))
    st.write("Data Types:", df.dtypes.to_dict())

# ---------- 2. Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Main Analysis", "Ad ID Details", "Cluster Visualization", "Final Score Analysis"])

with tab1:
    st.title("Metric Distribution and Bin Examples (1D or 2D)")

    # Cluster selection option
    available_clusters = sorted(df['cluster'].dropna().unique())
    use_cluster_filter = st.checkbox("Filter by cluster")
    if use_cluster_filter:
        selected_cluster_analysis = st.selectbox("Select Cluster for Analysis", available_clusters)
        df_analysis = df[df['cluster'] == selected_cluster_analysis]
    else:
        df_analysis = df.copy()

    num_cols = [c for c in df_analysis.select_dtypes(include="number").columns if c not in ["ad_id", "cluster"]]  # Exclude ad_id and cluster from metrics

    if not num_cols:
        st.error("No numeric columns found in data.")
        st.stop()

    # Select metrics; allow "None" for Y to enable 1D mode
    metric_x = st.selectbox("Select X metric", num_cols, index=0)
    metric_y_options = ["None"] + [col for col in num_cols if col != metric_x]
    metric_y = st.selectbox("Select Y metric (or None for 1D analysis)", metric_y_options, index=0)

    max_examples = st.number_input("Max images per bin (or intersection)", 3, 20, 5, 1)

    # Custom CSS for larger images
    st.markdown("""
    <style>
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- 3. Validate and prepare data ----------
    if metric_x not in df_analysis.columns or not pd.api.types.is_numeric_dtype(df_analysis[metric_x]):
        st.error(f"Invalid X metric '{metric_x}'. It must be a numeric column.")
        st.stop()

    if metric_y != "None":
        if metric_y not in df_analysis.columns or not pd.api.types.is_numeric_dtype(df_analysis[metric_y]):
            st.error(f"Invalid Y metric '{metric_y}'. It must be a numeric column.")
            st.stop()

        # Bivariate data
        bivar_data = df_analysis[[metric_x, metric_y]].dropna()
        if bivar_data.empty:
            st.error(f"No valid data for metrics '{metric_x}' and '{metric_y}'.")
            st.stop()
        x_data = bivar_data[metric_x]
        y_data = bivar_data[metric_y]
    else:
        # Univariate data
        x_data = df_analysis[metric_x].dropna()
        if x_data.empty:
            st.error(f"No valid data for metric '{metric_x}'.")
            st.stop()
        y_data = None

    # ---------- 4. Compute and display histogram ----------
    num_bins = 10  # Adjustable; could add a slider for this if desired

    if metric_y == "None":
        # 1D Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(x_data, bins=num_bins, edgecolor='black')
        ax.set_title(f"Distribution of {metric_x}")
        ax.set_xlabel(metric_x)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        # 2D Histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        h = ax.hist2d(x_data, y_data, bins=num_bins, cmap='Blues')
        ax.set_title(f"2D Distribution of {metric_x} vs {metric_y}")
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        fig.colorbar(h[3], ax=ax)  # Add colorbar for density
        st.pyplot(fig)

    # Compute bin edges and labels
    x_hist_counts, x_edges = np.histogram(x_data, bins=num_bins)
    x_bin_labels = [f"{x_edges[i]:.2f} - {x_edges[i+1]:.2f}" for i in range(num_bins)]

    if metric_y != "None":
        y_hist_counts, y_edges = np.histogram(y_data, bins=num_bins)
        y_bin_labels = [f"{y_edges[i]:.2f} - {y_edges[i+1]:.2f}" for i in range(num_bins)]

    # ---------- 5. Bin selection ----------
    if metric_y == "None":
        selected_x_bin = st.selectbox(f"Select {metric_x} bin", x_bin_labels)
        selected_y_bin = None
    else:
        selected_x_bin = st.selectbox(f"Select {metric_x} bin", x_bin_labels)
        selected_y_bin = st.selectbox(f"Select {metric_y} bin", y_bin_labels)

    if (metric_y == "None" and selected_x_bin) or (metric_y != "None" and selected_x_bin and selected_y_bin):
        # Determine filters
        x_bin_idx = x_bin_labels.index(selected_x_bin)
        x_bin_min = x_edges[x_bin_idx]
        x_bin_max = x_edges[x_bin_idx + 1]

        if metric_y != "None":
            y_bin_idx = y_bin_labels.index(selected_y_bin)
            y_bin_min = y_edges[y_bin_idx]
            y_bin_max = y_edges[y_bin_idx + 1]
            filter_cond = (
                (df_analysis[metric_x] >= x_bin_min) & (df_analysis[metric_x] < x_bin_max) &
                (df_analysis[metric_y] >= y_bin_min) & (df_analysis[metric_y] < y_bin_max)
            )
            dropna_cols = [metric_x, metric_y]
            subtitle = f"Intersection: {selected_x_bin} ({metric_x}) and {selected_y_bin} ({metric_y})"
            sort_col = metric_x  # Or use sum: df_analysis[metric_x] + df_analysis[metric_y]
            caption_func = lambda row: f"ad_id: {row['ad_id']}, {metric_x}: {row[metric_x]:.3f}, {metric_y}: {row[metric_y]:.3f}"
        else:
            filter_cond = (df_analysis[metric_x] >= x_bin_min) & (df_analysis[metric_x] < x_bin_max)
            dropna_cols = [metric_x]
            subtitle = f"Bin: {selected_x_bin} ({metric_x})"
            sort_col = metric_x
            caption_func = lambda row: f"ad_id: {row['ad_id']}, {metric_x}: {row[metric_x]:.3f}"

        # Filter data
        filtered_df = df_analysis[filter_cond].dropna(subset=dropna_cols)

        if filtered_df.empty:
            st.warning(f"No examples in selected bin(s).")
        else:
            # Sort descending and limit to max_examples
            sorted_filtered = filtered_df.sort_values(sort_col, ascending=False).head(max_examples)
            st.subheader(f"Examples from {subtitle} (Top {len(sorted_filtered)} by {sort_col})")
            cols = st.columns(min(max_examples, len(sorted_filtered)))
            for i, (_, row) in enumerate(sorted_filtered.iterrows()):
                with cols[i % len(cols)]:
                    st.image(row["url"], use_container_width=True, caption=caption_func(row))

with tab2:
    st.title("Ad ID Details")
    st.write("Paste an ad_id from the main page to view all associated metrics and image.")

    pasted_ad_id = st.text_input("Enter ad_id")

    if pasted_ad_id:
        try:
            ad_id_value = int(pasted_ad_id)  # Convert to int since ad_id is integer
            ad_id_data = df[df['ad_id'] == ad_id_value]
            if ad_id_data.empty:
                st.warning(f"No data found for ad_id {ad_id_value}")
            else:
                st.subheader(f"Metrics for ad_id: {ad_id_value}")
                st.dataframe(ad_id_data)
                # Show images associated with ad_id, one or more
                urls = ad_id_data['url'].unique()
                for url in urls:
                    st.image(url, caption=f"Image for ad_id {ad_id_value}", use_container_width=True)
        except ValueError:
            st.error("Please enter a valid integer ad_id.")

with tab3:
    st.title("Cluster-wise Analysis and Visualization")
    clusters = df['cluster'].dropna().unique()
    selected_cluster = st.selectbox("Select Cluster", sorted(clusters))  # Sorted for better UX

    cluster_data = df[df['cluster'] == selected_cluster]
    st.write(f"Showing data for cluster {selected_cluster} (Count: {len(cluster_data)})")

    if cluster_data.empty:
        st.warning(f"No data available for cluster {selected_cluster}")
    else:
        # Show descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(cluster_data.describe())

        # Show images in a grid (limited to prevent overload; adjust as needed)
        st.subheader(f"Sample Images in Cluster {selected_cluster}")
        sample_size = min(20, len(cluster_data))  # Limit to 20 images for performance
        sample_data = cluster_data.sample(sample_size) if sample_size < len(cluster_data) else cluster_data
        cols = st.columns(5)
        for idx, (_, row) in enumerate(sample_data.iterrows()):
            with cols[idx % 5]:
                st.image(row['url'], caption=f"ad_id: {row['ad_id']}", use_container_width=True)

with tab4:
    st.title("Final Score Analysis")
    if not final_scores:  # Check if list is empty (removes the warning)
        st.warning("No final scores data available.")
    else:
        # Convert list to DataFrame for easier display
        final_scores_df = pd.DataFrame(final_scores)
        st.dataframe(final_scores_df)
