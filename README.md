# 🧠 Semantic Similarity Checker for SEO & Content Analysis

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/amal-alexander-305780131/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat&logo=streamlit)](https://your-app-url.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)

A Streamlit-powered tool to **crawl web pages and compare semantic similarity** between adjacent URLs, designed for SEO audits, content optimization, and duplicate content detection.

![image](https://github.com/user-attachments/assets/158b0795-e8f7-40b2-ad59-295df1d7ee3c)


## 🔍 Features

- **URL List Processing**: Analyze up to 2,000 URLs in one batch
- **Semantic Analysis**: Uses `all-MiniLM-L6-v2` transformer model for accurate text comparisons
- **Threshold Control**: Adjustable similarity score (0.5-1.0) to flag near-duplicate content
- **Content Preview**: Shows extracted text snippets for manual verification
- **Multi-Input Support**: Paste URLs or upload TXT/CSV/XLSX files
- **Export Results**: Download full comparison data as CSV

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install streamlit pandas sentence-transformers beautifulsoup4 requests
