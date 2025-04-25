import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types

# Gemini-based LLM function
def generate_llm_insight(prompt):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction="You are a data analyst summarizing insurance user survey data."),
        contents=[prompt],
    )
    return response.text.strip()

@st.cache_data
def load_data():
    return pd.read_excel("combined_insurance_data.xlsx")

st.set_page_config(layout="wide")
st.title("🔁 Cross-Tab Insights")

df = load_data()

city_col, age_col, edu_col, job_col = "City", "Age Group", "Education", "Occupation"

with st.sidebar:
    st.header("📋 Filters")
    selected_city = st.selectbox("City", ["All"] + sorted(df[city_col].dropna().unique()))
    selected_age = st.selectbox("Age Group", ["All"] + sorted(df[age_col].dropna().unique()))
    selected_edu = st.selectbox("Education", ["All"] + sorted(df[edu_col].dropna().unique()))
    selected_job = st.selectbox("Occupation", ["All"] + sorted(df[job_col].dropna().unique()))

for col, selected in zip([city_col, age_col, edu_col, job_col], [selected_city, selected_age, selected_edu, selected_job]):
    if selected != "All":
        df = df[df[col] == selected]

selected_section = st.radio("📈 Select Section", ["📱 Digital Demographic", "🤝 Trust & Claims", "❤️ Loyalty Signals"], horizontal=True)

if selected_section == "📱 Digital Demographic":
    df['Digital - App'] = df['Digital - App'].replace({0: 'No', 1: 'Yes'})
    df['Digital - Website'] = df['Digital - Website'].replace({0: 'No', 1: 'Yes'})

    app_cross = pd.crosstab(df[age_col], df['Digital - App'])
    web_cross = pd.crosstab(df[age_col], df['Digital - Website'])
    edu_app_cross = pd.crosstab(df[edu_col], df['Digital - App'])

    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        fig1 = px.bar(
            app_cross.reset_index().melt(id_vars=age_col, var_name="App Usage", value_name="Count"),
            x=age_col,
            y="Count",
            color="App Usage",
            title="App Usage by Age Group",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            web_cross.reset_index().melt(id_vars=age_col, var_name="Website Usage", value_name="Count"),
            x=age_col,
            y="Count",
            color="Website Usage",
            title="Website Usage by Age Group",
            barmode="stack",
            color_discrete_sequence=px.colors.sequential.Magma
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.bar(
            edu_app_cross.reset_index().melt(id_vars=edu_col, var_name="App Usage", value_name="Count"),
            x=edu_col,
            y="Count",
            color="App Usage",
            title="App Usage by Education Level",
            barmode="stack",
            color_discrete_sequence=px.colors.sequential.Cividis
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🧠 Gemini-Generated Insights")
        prompt = f"""
        You are a data analyst summarizing digital usage patterns. 
        Analyze the following cross-tabulated data and provide 3 concise insights:
        App Usage by Age Group:
        {app_cross.to_string()}

        Website Usage by Age Group:
        {web_cross.to_string()}

        App Usage by Education:
        {edu_app_cross.to_string()}

        - Focus on trends, anomalies, and actionable findings.
        - Use simple language for a business analyst.
        """
        insight = generate_llm_insight(prompt)
        st.markdown(f"**Insight:**\n{insight}")



elif selected_section == "🤝 Trust & Claims":

    trust_cross = pd.crosstab(df[edu_col], df['Trust'])
    claim_cross = pd.crosstab(df['Claim Made'], df['Trust'])
    col1, col2 = st.columns([1.5, 1.5])

    with col1:
        fig1 = px.bar(
            trust_cross.reset_index().melt(id_vars=edu_col, var_name="Trust Level", value_name="Count"),
            x=edu_col,
            y="Count",
            color="Trust Level",
            title="Trust Level by Education",
            barmode="stack",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            claim_cross.reset_index().melt(id_vars='Claim Made', var_name="Trust Level", value_name="Count"),
            x='Claim Made',
            y="Count",
            color="Trust Level",
            title="Claim History vs Trust",
            barmode="stack",
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("🧠 Gemini-Generated Insights")
        prompt = f"""
        You are a data analyst summarizing digital usage patterns. 
        Analyze the following cross-tabulated data and provide 3 concise insights:

        Education vs Trust:
        {trust_cross.to_string()}

        Claim Made vs Trust:
        {claim_cross.to_string()}

        - Focus on trends, anomalies, and actionable findings.
        - Use simple language for a business analyst.
        """
        insight = generate_llm_insight(prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "❤️ Loyalty Signals":
    nps_cross = pd.crosstab(df[edu_col], df['NPS'])

    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        fig1 = px.bar(
            nps_cross.reset_index().melt(id_vars=edu_col, var_name="NPS Score", value_name="Count"),
            x=edu_col,
            y="Count",
            color="NPS Score",
            title="Education Level vs Likelihood to Recommend (NPS)",
            barmode="stack",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("🧠 Gemini-Generated Insights")
        prompt = f"""
        You are a data analyst summarizing digital usage patterns. 
        Analyze the following cross-tabulated data and provide 3 concise insights:

        Education vs NPS:
        {nps_cross.to_string()}

        - Focus on trends, anomalies, and actionable findings.
        - Use simple language for a business analyst.
        """
        insight = generate_llm_insight(prompt)
        st.markdown(f"**Insight:**\n{insight}")
