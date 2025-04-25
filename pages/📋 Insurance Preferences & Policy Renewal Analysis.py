import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📋 Insurance Preferences & Policy Renewal Analysis")

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("combined_insurance_data.xlsx")

df = load_data()

# Weighted chart function
def plot_weighted_chart(rank_cols, weights, title):
    scores = {}
    for col, weight in zip(rank_cols, weights):
        counts = df[col].value_counts().fillna(0) * weight
        for option, score in counts.items():
            scores[option] = scores.get(option, 0) + score
    score_df = pd.DataFrame(list(scores.items()), columns=["Option", "Weighted Score"])
    score_df = score_df.sort_values(by="Weighted Score", ascending=False)
    fig = px.bar(score_df, x="Option", y="Weighted Score", title=title,
                 labels={"Option": "Policy Renewal Reason"}, height=500)
    fig.update_layout(xaxis_tickangle=-45)
    return fig, score_df

# Gemini-based LLM function
def generate_llm_insight(prompt):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction="You are a data analyst summarizing insurance user survey data."),
        contents=[prompt],
    )
    return response.text.strip()

# Filters
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

selected_section = st.radio("📊 Select Section", ["🎯 Policy Renewal Reasons", "📱 Digital Preferences", "📶 Digital Readiness"], horizontal=True)

if selected_section == "📱 Digital Preferences":
    col1, col2 = st.columns(2)
    with col1:
        
        # Select columns with digital channels
        digital_cols = [col for col in df.columns if any(x in col for x in ["WhatsApp", "App", "Website", "Chatbot", "Call Centre"])]

        # Summarize values
        digital_summary = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum().reset_index()
        digital_summary.columns = ["Channel", "Count"]

        # Sort the data for better visualization
        digital_summary = digital_summary.sort_values(by="Count", ascending=False)

        # Split by label length
        short_labels = digital_summary[digital_summary["Channel"].str.len() < 30]

        # Plot 1: Short labels
        fig1 = px.bar(short_labels, x="Channel", y="Count", title="Digital Channel Preference ", color="Channel")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        
        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        Based on this following:
        
        digital preference data (total counts):
        {digital_summary.to_string(index=False)}

        Summarize 2–3 key insights in simple language.
        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "🎯 Policy Renewal Reasons":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("🎯 Policy Renewal Reasons (Weighted)")
        # --------- Weighted Plot for Q10b Rankings ---------
        rank_columns = [
            '[Rank 1] - Q10b',
            '[Rank 2] - Q10b',
            '[Rank 3] - Q10b',
            '[Rank 4] - Q10b'
        ]

        weights = {
            '[Rank 1] - Q10b': 4,
            '[Rank 2] - Q10b': 3,
            '[Rank 3] - Q10b': 2,
            '[Rank 4] - Q10b': 1
        }

        weighted_totals = {}
        counts = {}
        for col in rank_columns:
            weight = weights[col]
            for item in df[col].dropna():
                item = item.strip()
                weighted_totals[item] = weighted_totals.get(item, 0) + weight
                counts[item] = counts.get(item, 0) + 1

        q10b_weighted = {k: weighted_totals[k] / counts[k] for k in weighted_totals}

        q10b_fig = px.bar(
            x=list(q10b_weighted.keys()),
            y=list(q10b_weighted.values()),
            title='Weighted Average Score for Renewal reasons Rankings',
            labels={'x': 'Attribute', 'y': 'Weighted Score'},
            color_discrete_sequence=['Pink']
        )
        q10b_fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(q10b_fig, use_container_width=True)
        # Original rating columns with full names
        full_rating_columns = [
            'Rating - Assistance by Hospital',
            'Rating - Cooperative Staff',
            'Rating - TPA Desk Assistance',
            'Rating - Overall Hospital Service'
        ]

        # Clean names for display
        clean_rating_columns = [
            'Assistance by Hospital',
            'Cooperative Staff',
            'TPA Desk Assistance',
            'Overall Hospital Service'
        ]

        # Convert rating columns to numeric
        df[full_rating_columns] = df[full_rating_columns].apply(pd.to_numeric, errors='coerce')
        rating_counts = {}
        for col in full_rating_columns:
            rating_counts[col] = df[col].value_counts().sort_index()
        rating_counts_df = pd.DataFrame(rating_counts).fillna(0).astype(int).sort_index()
        rating_counts_df = rating_counts_df.reset_index().rename(columns={'index': 'Rating'})

        # Rename columns for display
        rating_counts_df.columns = ['Rating'] + clean_rating_columns
        
        # --------- Weighted Average Plot using Plotly ---------
        # Multiply each rating value by count and divide by total responses for weighted average
        weighted_scores = {}
        for col in clean_rating_columns:
            weighted_scores[col] = (rating_counts_df['Rating'] * rating_counts_df[col]).sum() / rating_counts_df[col].sum()

        weighted_avg_fig = px.bar(
            x=list(weighted_scores.keys()),
            y=list(weighted_scores.values()),
            title='Weighted Average Ratings by Service',
            labels={'x': 'Service Attribute', 'y': 'Weighted Average Rating'},
            color_discrete_sequence=['Orange']
        )
        weighted_avg_fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(weighted_avg_fig, use_container_width=True)

        # --------- Stacked Bar Chart using Plotly ---------

        stacked_fig = px.bar(
            rating_counts_df,
            x='Rating',
            y=clean_rating_columns,
            title='Stacked Bar Chart: Rating Distribution by Service',
            labels={'value': 'Number of Respondents'},
        )
        st.plotly_chart(stacked_fig, use_container_width=True)



    with col2:

        # --------- Radar Plot using Plotly ---------
        avg_ratings = df[full_rating_columns].mean()
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=avg_ratings.tolist() + [avg_ratings.tolist()[0]],
            theta=clean_rating_columns + [clean_rating_columns[0]],
            fill='toself',
            name='Average Rating',
            line_color='orange'
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5])
            ),
            showlegend=False,
            title="Service Profile"
        )
        st.plotly_chart(radar_fig, use_container_width=True)

        st.subheader("🧠 Gemini Insights")
        renew_prompt = f"""
        Based on the following: 

        Weighted Average Score for Service Ratings:
        {weighted_scores}
        
        Weighted Average Score for Renewal reasons Ranking:
        {q10b_weighted}

        Rating Distribution by Service:
        {rating_counts_df.to_string(index=False)}

        Service Profile:
        {avg_ratings.to_string(index=False)}

        Summarize the key reasons users renew their policies.
        """
        insight = generate_llm_insight(renew_prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "📶 Digital Readiness":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("📶 Digital Readiness Score by Age")
        digital_cols = [col for col in df.columns if any(x in col for x in ["WhatsApp", "App", "Website", "Chatbot", "Call Centre"])]
        df["digital_score"] = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        readiness_scores = df.groupby(age_col)["digital_score"].mean().sort_values().reset_index()
        fig = px.bar(readiness_scores, x=age_col, y="digital_score", title="Average Digital Readiness Score by Age", color="digital_score", color_continuous_scale="Purples")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🧠 Gemini Insights")
        readiness_prompt = f"""
        Based on the average digital readiness score by age group:
        {readiness_scores.to_string(index=False)}
        Provide 2-3 insights about digital channel adoption.
        """
        insight = generate_llm_insight(readiness_prompt)
        st.markdown(f"**Insight:**\n{insight}")
