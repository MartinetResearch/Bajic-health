import streamlit as st
st.set_page_config(layout="wide")

from google import genai
from google.genai import types

import pandas as pd
import plotly.express as px

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

df = load_data()

st.title("👥 Insurance Demographics Overview")

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

selected_section = st.radio("📊 Select Section", ["🧬 Respondent Profile", "🏢 Insurer Profile","🗺️ Demographics by Location", "📈 Socioeconomic Patterns"], horizontal=True)

if selected_section == "🧬 Respondent Profile":
    col1, col2 = st.columns([1.5, 1.5])

    with col1:
        # Age Group Distribution (Pie Chart)
        age_counts = df[age_col].value_counts()
        fig1 = px.pie(
            names=age_counts.index,
            values=age_counts.values,
            title="Age Group Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Occupation Distribution (Horizontal Bar Chart)
        job_counts = df[job_col].value_counts().sort_values(ascending=True)
        fig2 = px.bar(
            x=job_counts.values,
            y=job_counts.index,
            orientation='h',
            title="Respondent Occupation Type",
            labels={'x': 'Count', 'y': 'Occupation'},
            color_discrete_sequence=['skyblue']
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Education Level Distribution (Horizontal Bar Chart)
        edu_counts = df[edu_col].value_counts().sort_values(ascending=True)
        fig3 = px.bar(
            x=edu_counts.values,
            y=edu_counts.index,
            orientation='h',
            title="Respondent Education Level",
            labels={'x': 'Count', 'y': 'Education Level'},
            color_discrete_sequence=['coral']
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        You are a data analyst summarizing insurance user survey data. Analyze the following demographic distributions and provide key insights in 3 concise bullet points. 
        Focus on trends, anomalies, or significant patterns.

        Age Group Distribution:
        {age_counts.to_string()}

        Occupation Distribution:
        {job_counts.to_string()}

        Education Level Distribution:
        {edu_counts.to_string()}
        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown("### Key Insights:")
        st.markdown(insight)    

elif selected_section == "🏢 Insurer Profile":
    col1, col2 = st.columns([1.5, 1.5])

    with col1:
        # Count how many responses each insurer received
        insurer_counts = df['Insurer'].value_counts().reset_index()
        insurer_counts.columns = ['Insurer', 'Count']

        # Create a Plotly horizontal bar chart with an improved color palette
        fig = px.bar(insurer_counts,
                     y='Insurer',
                     x='Count',
                     orientation='h',
                     title='Number of Respondents per Insurance Company',
                     text='Count',
                     color='Count',
                     color_continuous_scale=px.colors.sequential.Viridis)

        fig.update_layout(
            xaxis_title='Number of Respondents',
            yaxis_title='Insurance Company',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Note:** The chart shows the number of respondents for each insurance company, Appling Filters is recommended.")
        
    with col2:
        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        You are a data analyst summarizing insurance user survey data. Analyze the following demographic distributions and provide key insights in 3 concise bullet points. 
        Focus on trends, anomalies, or significant patterns.

        Insurer Company Distribution:
        {insurer_counts.to_string()}
        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown("### Key Insights:")
        st.markdown(insight)    

elif selected_section == "🗺️ Demographics by Location":
    col1, col2 = st.columns([1.5, 1.5])

    with col1:
        # Age Group by City
        age_city_df = pd.crosstab(df["City"], df["Age Group"]).reset_index().melt(id_vars="City", var_name="Age Group", value_name="Count")
        fig1 = px.bar(
            age_city_df, y="City", x="Count", color="Age Group",
            title="Age Group Distribution by City",
            orientation="h", barmode="stack",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig1, use_container_width=True)

         # Education by City
        edu_city_df = pd.crosstab(df["City"], df["Education"]).reset_index().melt(id_vars="City", var_name="Education", value_name="Count")
        fig2 = px.bar(
            edu_city_df, y="City", x="Count", color="Education",
            title="Education Level Distribution by City",
            orientation="h", barmode="stack",
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        You are a data analyst summarizing insurance user survey data. Analyze the following demographic distributions and provide key insights in 3 concise bullet points. 
        Focus on trends, anomalies, or significant patterns.

        Age Group Distribution by City:
        {age_city_df.to_string()}

        Education Level Distribution by City:
        {edu_city_df.to_string()}
        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown("### Key Insights:")
        st.markdown(insight)    

elif selected_section == "📈 Socioeconomic Patterns":
    col1, col2 = st.columns([1.5, 1.5])

    with col1:
        # Crosstab for Occupation vs Age Group
        occ_age = pd.crosstab(df["Age Group"], df["Occupation"]).reset_index().melt(
            id_vars="Age Group", var_name="Occupation", value_name="Count"
        )

        # Plot using Plotly
        fig = px.bar(
            occ_age,
            x="Age Group",
            y="Count",
            color="Occupation",
            barmode="group",
            title="Occupation Type by Age Group",
            orientation='v',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Number of Respondents",
            legend_title="Occupation",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        You are a data analyst summarizing insurance user survey data. Analyze the following demographic distributions and provide key insights in 3 concise bullet points. 
        Focus on trends, anomalies, or significant patterns.

        Crosstab for Occupation vs Age Group:
        {occ_age.to_string()}

        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown("### Key Insights:")
        st.markdown(insight)