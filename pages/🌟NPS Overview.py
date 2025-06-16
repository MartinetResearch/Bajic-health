import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import plotly.express as px
import re
from google import genai
from google.genai import types

def extract_numeric(val):
    if pd.isnull(val):
        return None
    match = re.search(r'\d+', str(val))
    return int(match.group()) if match else None

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_NPS_Data.csv")
    nps_columns = ['Q1', 'Q2', 'Q5a', 'Q8', 'Q13']
    for col in nps_columns:
        df[col] = df[col].apply(extract_numeric)
    df = df.dropna(subset=nps_columns, how='all')
    return df

def calculate_nps(series):
    series = series.dropna()
    if len(series) == 0:
        return None
    promoters = (series >= 9).sum()
    detractors = (series <= 6).sum()
    total = len(series)
    return round(((promoters - detractors) / total) * 100, 2)

def generate_llm_insight(prompt):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a data analyst summarizing NPS scores from a health insurance survey."),
        contents=[prompt],
    )
    return response.text.strip()

# Labels and full question descriptions
question_labels = {
    "Q1": "Overall experience & likelihood to recommend",
    "Q2": "Satisfaction during policy purchase process",
    "Q5a": "Satisfaction with claims process interaction",
    "Q8": "Satisfaction with assistance during claim settlement",
    "Q13": "Satisfaction during policy renewal process"
}

full_question_texts = {
    "Q1": "Taking into consideration your overall experience, like Pre & Post Sales experience with Agent, Services provided by the company till now, Renewals / Claims exp, how likely is it that you would recommend the insurer to a friend or to a colleague?",
    "Q2": "Kindly tell me how you would rate your overall satisfaction with the insurer during the entire process of purchasing the policy on a scale 1 to 10.",
    "Q5a": "Keeping in mind the interaction during the CLAIMS PROCESS (ease, comfort, and convenience), please rate your satisfaction on a scale of 1 to 10.",
    "Q8": "On a scale of 1 to 10, please tell me how satisfied are you with the assistance provided by the insurer during the claims settlement process.",
    "Q13": "Kindly tell me how you would rate your overall satisfaction with the insurer during the entire process of renewal on a scale 1 to 10."
}

# --- Streamlit App ---

st.title("🌟Net Promoter Score (NPS) Overview")

df = load_data()

with st.sidebar:
    st.header("🔍 Filters")
    selected_city = st.selectbox("City", ["All"] + sorted(df["City"].dropna().unique()))
    selected_insurer = st.selectbox("Insurer", ["All"] + sorted(df["Insurer"].dropna().unique()))

filtered_df = df.copy()
if selected_city != "All":
    filtered_df = filtered_df[filtered_df["City"] == selected_city]
if selected_insurer != "All":
    filtered_df = filtered_df[filtered_df["Insurer"] == selected_insurer]

nps_columns = ['Q1', 'Q2', 'Q5a', 'Q8', 'Q13']
nps_results = []
summary_text = ""

for col in nps_columns:
    score = calculate_nps(filtered_df[col])
    if score is not None:
        nps_results.append({
            "Question": col,
            "Label": question_labels[col],
            "NPS Score": score
        })
        summary_text += f"{col}: {score}\n"

col1, col2 = st.columns([1.5, 1.5])

if nps_results:
    with col1:
        st.subheader("📊 NPS Scores by Question")
        
        df_scores = pd.DataFrame(nps_results)

        fig = px.bar(
            df_scores,
            x="Question",  # Use Q codes as axis labels
            y="NPS Score",
            text="NPS Score",
            hover_name="Label",  # Descriptions shown on hover
            color="NPS Score",
            color_continuous_scale=px.colors.sequential.Tealgrn,
            title=f"NPS Scores (Filtered by: {selected_insurer if selected_insurer != 'All' else 'All Insurers'}, {selected_city if selected_city != 'All' else 'All Cities'})"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(template="plotly_white", yaxis_title="NPS Score", xaxis_title="Question")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Question Descriptions"):
            for q in question_labels:
                st.markdown(f"**{q}** - {question_labels[q]}")

        with st.expander("📜 Full Question Texts"):
            for q in nps_columns:
                st.markdown(f"**{q}**: {full_question_texts[q]}")

    with col2:
        st.subheader("🧠 Gemini Insights")
        try:
            prompt = f"""
            You are a data analyst. Provide 3 key insights from the following NPS scores from a health insurance feedback survey.
            Filters: City = {selected_city}, Insurer = {selected_insurer}

            NPS Scores:
            {summary_text.strip()}
            """
            insight = generate_llm_insight(prompt)
            st.markdown("### Key Insights:")
            st.markdown(insight)
        except Exception as e:
            st.warning("⚠️ Gemini insights could not be generated. Check your internet connection.")
else:
    st.warning("No valid NPS data found for selected filters.")
