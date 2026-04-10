# Step 1: Imports & Config

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import os
from openai import OpenAI

# Load API Key safely from environment variables
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize the OpenAI client configured for OpenRouter
# This allows using Qwen models via the OpenAI-compatible SDK
GROQ_API_KEY = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# Example usage function to demonstrate calling the Qwen model
def get_qwen_response(prompt,system_role="You are a helpful assistant."):
    try:
        response = GROQ_API_KEY.client.chat.completions.create(
            # Using Qwen 2.5 72B Instruct - one of the most powerful Qwen models
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content":system_role},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Placeholder for Step 2: Database or Data Analysis Logic
# (You can continue adding your pandas/sqlite3 logic below)

if __name__ == "__main__":
    # Test the connection
    print("Testing Qwen Model via OpenRouter...")
    # test_prompt = "Write a short python function to calculate the area of a circle."
    # print(get_qwen_response(test_prompt))

#-------------------------------------------------------------------------------------

    # Step 2: Memory System

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    created_at TEXT
)
""")
conn.commit()

def save_memory(text):
    cursor.execute("INSERT INTO insights (content, created_at) VALUES (?, ?)",
                   (text, str(datetime.now())))
    conn.commit()

def load_memory():
    cursor.execute("SELECT content FROM insights ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    if not rows:
        return "No previous memory found."
    
    memory_text = " | ".join([row[0] for row in reversed(rows)])
    return memory_text

#-------------------------------------------------------------------------------------

# Step 3: Data Collection Agent

def load_file_content(file):
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'csv':
            df = pd.read_csv(file)
            return df.describe().to_string(), "data" 
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(file)
            return df.describe().to_string(), "data"
        elif ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text[:4000], "text"
    except Exception as e:
        return None, None
    
#-------------------------------------------------------------------------------------

# Step 4: Data Analysis Agent

def analyze_data(df):
    result = {}

    result['total_sales'] = df['sales'].sum()
    result['avg_sales'] = df['sales'].mean()

    result['top_products'] = df.groupby('product')['sales'].sum().sort_values(ascending=False)

    result['daily_sales'] = df.groupby('date')['sales'].sum()

    result['category_sales'] = df.groupby('category')['sales'].sum()

    return result

#-------------------------------------------------------------------------------------

# Step 5: Anomaly Detection (Z-score)

def detect_anomalies(daily_sales):
    anomalies = []

    mean = daily_sales.mean()
    std = daily_sales.std()

    for date, value in daily_sales.items():
        z = (value - mean) / std if std != 0 else 0
        if abs(z) > 1.5:
            anomalies.append((str(date), float(value)))

    return anomalies

#-------------------------------------------------------------------------------------

# Step 6: Planner Agent

def plan_tasks(user_input):
    tools = [
        "Load Data",
        "Analyze Data",
        "Detect Anomalies",
        "Generate Insights",
        "Make Decisions"
    ]
    
    prompt = f"""
    You are an AI Orchestrator. The user wants to: '{user_input}'.
    Based on this goal, select the necessary steps from this list: {tools}.
    Return the steps as a simple comma-separated list. 
    Example: Load Data, Analyze Data, Make Decisions
    """
    
    plan_raw = get_qwen_response(prompt, system_role="You are a Strategic Workflow Planner")

    steps = [step.strip() for step in plan_raw.split(',')]
    
    print(f"📋 Dynamic Plan Generated: {steps}")
    return steps

#-------------------------------------------------------------------------------------

# Step 7: LLM Reasoning (Qwen)

def generate_insights(summary, anomalies, memory):

    prompt = f"""
    You are an expert business analyst AI.
    
    Previous Context: {memory}
    
    Current Business Data:
    - Total Sales: {summary['total_sales']}
    - Avg Sales: {summary['avg_sales']}
    - Top Products: {summary['top_products']}
    - Category Sales: {summary['category_sales']}
    
    Identified Anomalies: {anomalies}
    
    Provide:
    1. Key Insights (What is happening?)
    2. Hidden Problems (Why is it happening?)
    3. Strategic Recommendations (What to do?)
    """
    
    return get_qwen_response(prompt, system_role="Expert Business Analyst")

#-------------------------------------------------------------------------------------

# Step 8: Decision Agent

def make_decisions(insights, summary):

    prompt = f"""
    You are a Senior Business Manager. Based on these insights:
    {insights}
    
    And this data summary:
    Total Sales: {summary['total_sales']}
    Top Products: {summary['top_products']}
    
    Task: Provide 3 concrete executive decisions to improve performance or fix issues. 
    Be specific and action-oriented.
    """
    
    return get_qwen_response(prompt, system_role="Senior Executive Decision Maker")

#-------------------------------------------------------------------------------------

# Step 9: Visualization

def plot_data(daily_sales):
    plt.figure()
    daily_sales.plot()
    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()

#-------------------------------------------------------------------------------------

# Step 10: Orchestrator

def load_pdf_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text[:10000] if text else "No text found in PDF"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    
def run_agent(file, user_input):
    file_ext = file.name.split('.')[-1].lower()
    
    if file_ext in ['csv', 'xlsx', 'xls']:
        steps = plan_tasks(user_input) 
        df = load_data(file)
        summary = analyze_data(df)
        anomalies = detect_anomalies(summary['daily_sales'])
        memory = load_memory()
        
        insights = generate_insights(summary, anomalies, memory)
        decisions = make_decisions(insights, summary) 
        
        save_memory(insights)
        pdf_file = generate_pdf_report(insights, decisions, anomalies)
        
        return steps, summary, anomalies, insights, decisions, pdf_file
        
    elif file_ext == 'pdf':
        steps = ["Extract Text", "AI Document Analysis"]
        raw_text = load_pdf_text(file)
        insights = get_qwen_response(f"Analyze this document: {raw_text[:2000]}", "Document Expert")
        decisions = "Decisions are inside the detailed PDF analysis."
        return steps, None, None, insights, decisions, None
    
#-------------------------------------------------------------------------------------

# Step 11: PDF Report Generator

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(insights, decisions, anomalies):

    file_name = "business_report.pdf"

    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(Paragraph("AI Business Report", styles['Title']))
    content.append(Spacer(1, 12))

    # Insights
    content.append(Paragraph("Insights:", styles['Heading2']))
    content.append(Paragraph(insights.replace("\n", "<br/>"), styles['Normal']))
    content.append(Spacer(1, 12))

    # Decisions
    content.append(Paragraph("Decisions:", styles['Heading2']))
    content.append(Paragraph(str(decisions), styles['Normal']))
    content.append(Spacer(1, 12))

    # Anomalies
    content.append(Paragraph("Anomalies:", styles['Heading2']))
    content.append(Paragraph(str(anomalies), styles['Normal']))
    content.append(Spacer(1, 12))

    doc.build(content)
    print(f"✅ Report saved as: {file_name}")

    return file_name

#-------------------------------------------------------------------------------------

# Step 12: Frontend

import streamlit as st
import pandas as pd
import plotly.express as px
import io
import PyPDF2
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- 1. Page Config ---
st.set_page_config(page_title="AI Business Agent", page_icon="🤖", layout="wide")

# --- 2. AI Client (Replace with your key) ---
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# --- 3. Professional CSS ---
st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; }
        div[data-baseweb="input"], .stFileUploader section {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px);
        }
        .result-card, .decision-card { 
            background: rgba(30, 41, 59, 0.7); padding: 25px; border-radius: 20px; 
            border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 4. Helper Functions ---
def load_data(file):
    try:
        ext = file.name.split('.')[-1].lower()
        if ext == 'csv': return pd.read_csv(file)
        elif ext in ['xls', 'xlsx']: return pd.read_excel(file)
    except: return None

def generate_pdf(insights, decisions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = [
        Paragraph("AI Business Report", styles['Title']),
        Spacer(1, 12),
        Paragraph("Insights:", styles['Heading2']),
        Paragraph(insights, styles['BodyText']),
        Spacer(1, 12),
        Paragraph("Decisions:", styles['Heading2']),
        Paragraph(decisions, styles['BodyText'])
    ]
    doc.build(content)
    buffer.seek(0)
    return buffer

# --- 5. The Orchestrator ---
def run_agent(user_query, df_summary=None):
    system_msg = "You are a Business AI. Provide 'Insights:' and 'Decisions:' sections."
    context = f"Data Summary:\n{df_summary}" if df_summary else "No data."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Query: {user_query}\nContext: {context}"}
            ]
        )
        res = response.choices[0].message.content
        if "Decisions" in res:
            p = res.split("Decisions")
            return p[0].replace("Insights", "").strip(": \n"), p[1].strip(": \n")
        return res, "Check insights for actions."
    except Exception as e:
        return f"Error: {str(e)}", "Check API Key."

# --- 6. UI Layout ---
left_co, cent_co, last_co = st.columns([1, 3, 1])
with cent_co:
    st.markdown("<h1 style='text-align: center; font-size: 70px;'>🤖</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Business AI Intelligence</h2>", unsafe_allow_html=True)
    
    st.markdown("##### 💬 What is your question?")
    user_query = st.text_input("query", placeholder="e.g. Analyze sales", label_visibility="collapsed", key="main_q")
    
    st.markdown("##### 📂 Upload Data")
    uploaded_file = st.file_uploader("upload", type=["csv", "xlsx", "pdf"], label_visibility="collapsed", key="main_up")
    
    run_pressed = st.button("🚀 Execute Analysis", use_container_width=True)

# --- 7. Logic Execution ---
if run_pressed:
    if uploaded_file and user_query:
        with st.spinner("Analyzing..."):
            content, file_type = load_file_content(uploaded_file)
            
            if content is not None:
                ins, dec = run_agent(user_query, df_summary=content)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="result-card"><h3>💡 Insights</h3>{ins}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="decision-card"><h3>✅ Decisions</h3>{dec}</div>', unsafe_allow_html=True)
                    st.download_button("📥 Download PDF", data=generate_pdf(ins, dec), file_name="Report.pdf")
            else:
                st.error("Could not read file content.")
    else:
        st.warning("Please provide both a question and a file.")