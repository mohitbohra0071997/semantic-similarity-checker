import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import requests

# ---------------------------
# ✅ Page config
# ---------------------------
st.set_page_config(page_title="🧠 Semantic Similarity Checker", layout="wide")

# ---------------------------
# ✅ Load model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# ✅ Extract structured blocks
# ---------------------------
def extract_structured_blocks(soup):
    blocks = []
    if soup.title and soup.title.string:
        blocks.append(f"[TITLE]: {soup.title.string.strip()}")
    for tag in ['h1', 'h2', 'h3']:
        for el in soup.find_all(tag):
            text = el.get_text(strip=True)
            if text:
                blocks.append(f"[{tag.upper()}]: {text}")
    for p in soup.find_all('p')[:10]:  # limit to first 10
        text = p.get_text(strip=True)
        if text:
            blocks.append(f"[P]: {text}")
    return blocks

def extract_text_blocks(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        return extract_structured_blocks(soup)
    except Exception as e:
        return [f"ERROR: {e}"]

# ---------------------------
# ✅ Get similar blocks (batch)
# ---------------------------
def get_similar_blocks(blocks1, blocks2, mini_threshold=0.8):
    embeddings1 = model.encode(blocks1, convert_to_tensor=True)
    embeddings2 = model.encode(blocks2, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)

    similar = []
    for i in range(len(blocks1)):
        for j in range(len(blocks2)):
            score = similarity_matrix[i][j].item()
            if score >= mini_threshold:
                similar.append((blocks1[i], blocks2[j], round(score, 4)))
    return similar

# ---------------------------
# ✅ Extract visible text (fast mode)
# ---------------------------
def extract_text(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text if text else "ERROR: No content extracted"
    except Exception as e:
        return f"ERROR: {e}"

# ---------------------------
# ✅ Compute page similarity
# ---------------------------
def compute_similarity(text1, text2):
    try:
        e = model.encode([text1, text2], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(e[0], e[1]))
    except Exception as e:
        return f"ERROR: {e}"

# ---------------------------
# ✅ Streamlit UI
# ---------------------------
st.title("🧠 Semantic Similarity Checker")

with st.expander("📚 How to Use", expanded=True):
    st.markdown("""
    1. **Input URLs**: Paste up to 2,000 URLs or upload a file.
    2. **Select Mode**:
       - **Fast Mode**: One big text per page → quick check (1–2 seconds per pair).
       - **In-Depth Mode**: Block-level comparison → detailed match (5–20 seconds per pair).
    3. **Set Thresholds**.
    4. **Run**.
    5. **Review Results & Download**.
    """)

# ---------------------------
# ✅ URL Input
# ---------------------------
st.header("📥 Upload or Paste URLs")
input_type = st.radio("Choose input method", ["Paste URLs", "Upload File"], horizontal=True)

urls = []

if input_type == "Paste URLs":
    text = st.text_area("Paste up to 2000 URLs (one per line)", height=300)
    if text:
        urls = [line.strip() for line in text.splitlines() if line.strip()]
elif input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload .txt, .csv, or .xlsx", type=["txt", "csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".txt"):
                urls = [line.decode("utf-8").strip() for line in uploaded_file.readlines()]
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                urls = df.iloc[:, 0].dropna().astype(str).tolist()
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
                urls = df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Failed to read file: {e}")

urls = list(dict.fromkeys(urls))
if len(urls) > 2000:
    st.warning("⚠️ Limit is 2000 URLs. Truncating.")
    urls = urls[:2000]

if urls:
    st.success(f"✅ {len(urls)} unique URLs loaded.")
    st.dataframe(pd.DataFrame(urls, columns=["Loaded URLs"]).head(20))

# ---------------------------
# ✅ Mode selection
# ---------------------------
st.header("⚡ Choose Comparison Mode")
mode = st.radio("Mode:", ["Fast Mode (Top-Level)", "In-Depth Mode (Similar Blocks)"])

threshold = st.slider("🔎 Page Similarity Threshold", 0.5, 1.0, 0.95, 0.01)
if mode == "In-Depth Mode (Similar Blocks)":
    mini_threshold = st.slider("🧩 Block Similarity Threshold", 0.5, 1.0, 0.8, 0.01)

# ---------------------------
# ✅ Run comparison
# ---------------------------
if st.button("🚀 Run Semantic Comparison", use_container_width=True):
    if len(urls) < 2:
        st.error("⚠️ Please provide at least 2 URLs for comparison.")
    else:
        est_time = "1–2 sec per pair" if mode.startswith("Fast") else "5–20 sec per pair"
        st.info(f"Fetching pages and computing similarity. Estimated time: {est_time} per pair.")

        df_pairs = pd.DataFrame({
            "URL 1": urls[:-1],
            "URL 2": urls[1:]
        })

        results = []
        progress = st.progress(0)

        for i, row in df_pairs.iterrows():
            u1, u2 = row["URL 1"], row["URL 2"]

            if mode.startswith("Fast"):
                t1 = extract_text(u1)
                t2 = extract_text(u2)

                if t1.startswith("ERROR") or t2.startswith("ERROR"):
                    score = "ERROR"
                    flag = "❌ Fetch Error"
                else:
                    score = compute_similarity(t1, t2)
                    if isinstance(score, str):
                        flag = "❌ Encoding Error"
                    else:
                        score = round(score, 4)
                        flag = "🛑 Similar" if score >= threshold else ""

                results.append({
                    "URL 1": u1,
                    "URL 2": u2,
                    "Similarity Score": score,
                    "Flag": flag,
                    "Content 1 Preview": t1[:300],
                    "Content 2 Preview": t2[:300],
                    "Similar Blocks": []
                })

            else:  # In-Depth Mode
                blocks1 = extract_text_blocks(u1)
                blocks2 = extract_text_blocks(u2)

                if "ERROR" in blocks1[0] or "ERROR" in blocks2[0]:
                    score = "ERROR"
                    flag = "❌ Fetch Error"
                    similar_blocks = []
                else:
                    joined1 = " ".join(blocks1)
                    joined2 = " ".join(blocks2)
                    score = compute_similarity(joined1, joined2)
                    if isinstance(score, str):
                        flag = "❌ Encoding Error"
                    else:
                        score = round(score, 4)
                        flag = "🛑 Similar" if score >= threshold else ""
                    similar_blocks = get_similar_blocks(blocks1, blocks2, mini_threshold)

                results.append({
                    "URL 1": u1,
                    "URL 2": u2,
                    "Similarity Score": score,
                    "Flag": flag,
                    "Similar Blocks": similar_blocks
                })

            progress.progress((i + 1) / len(df_pairs))

        result_df = pd.DataFrame(results)
        st.success("✅ Done!")
        st.dataframe(result_df[["URL 1", "URL 2", "Similarity Score", "Flag"]], use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv, "semantic_similarity_results.csv", use_container_width=True)

        if mode.startswith("Fast"):
            with st.expander("🔍 Preview (Top-Level Text)"):
                for i, row in result_df.iterrows():
                    st.markdown(f"### 🔗 Pair {i+1}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**URL 1:** {row['URL 1']}")
                        st.text_area("Text", row['Content 1 Preview'], height=150, key=f"c1_{i}")
                    with col2:
                        st.markdown(f"**URL 2:** {row['URL 2']}")
                        st.text_area("Text", row['Content 2 Preview'], height=150, key=f"c2_{i}")
        else:
            with st.expander("🔍 Preview Similar Blocks Only"):
                for i, row in result_df.iterrows():
                    st.markdown(f"### 🔗 Pair {i+1} — Page Similarity: {row['Similarity Score']}")
                    similar_blocks = row['Similar Blocks']
                    if not similar_blocks:
                        st.write("No similar blocks found above mini threshold.")
                    else:
                        for idx, (b1, b2, s) in enumerate(similar_blocks):
                            st.markdown(f"**Block {idx+1} — Score: {s}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text_area("Page 1 Block", b1, height=100, key=f"b1_{i}_{idx}")
                            with col2:
                                st.text_area("Page 2 Block", b2, height=100, key=f"b2_{i}_{idx}")

else:
    st.info("ℹ️ Please provide URLs to begin.")
