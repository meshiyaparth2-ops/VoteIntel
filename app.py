import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from streamlit_option_menu import option_menu

# Load Model
model = pickle.load(open("voteintel_model.pkl", "rb"))

st.set_page_config(page_title="VoteIntel", layout="wide")

# Sidebar
with st.sidebar:
    selected = option_menu(
        "VoteIntel Dashboard",
        ["Manual Prediction", "Bulk Scanner", "Data Analysis"],
        icons=["person", "upload", "bar-chart", "info-circle"],
        default_index=0
    )
# ---------------------------
# 1. MANUAL PREDICTION
# ---------------------------
if selected == "Manual Prediction":
    st.title("🗳️ Manual Prediction")

    # Load dataset
    df = pd.read_csv("indian_election_data.csv")

    # Rename columns (IMPORTANT)
    df.rename(columns={
        'st_name': 'State',
        'pc_name': 'Constituency',
        'cand_name': 'Candidate_Name',
        'cand_sex': 'Gender',
        'partyname': 'Party',
        'totvotpoll': 'Votes'
    }, inplace=True)

    # Dropdown values
    states = sorted(df['State'].dropna().unique())
    parties = sorted(df['Party'].dropna().unique())

    # UI Inputs
    state = st.selectbox("State", states)

    # Filter constituency based on state
    filtered_const = df[df['State'] == state]['Constituency'].dropna().unique()
    constituency = st.selectbox("Constituency", sorted(filtered_const))

    party = st.selectbox("Party", parties)
    gender = st.selectbox("Gender", ["Male", "Female"])
    votes = st.number_input("Votes", min_value=0)

    # ---------------------------
    # Prediction Button
    # ---------------------------
    if st.button("Predict"):
        try:
            # Input dataframe
            input_df = pd.DataFrame({
                "Party": [party],
                "State": [state],
                "Constituency": [constituency],
                "Gender": [gender],
                "Votes": [votes]
            })

            # Prediction
            pred = model.predict(input_df)[0]

            # Probability (IMPORTANT UPGRADE)
            prob = model.predict_proba(input_df)[0][1] * 100

            st.markdown("### 🧠 AI Prediction Result")

            # Progress bar
            st.progress(int(prob))

            # Metric
            st.metric("Winning Probability", f"{prob:.2f}%")

            # Smart result display
            if prob > 70:
                st.success("🔥 Strong Winning Candidate")
            elif prob > 50:
                st.info("⚖️ Competitive Candidate")
            else:
                st.error("❌ Low Chance of Winning")

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------
# 2. BULK SCANNER
# ---------------------------
elif selected == "Bulk Scanner":
    st.title("📂 Bulk Prediction")

    # ---------------------------
    # SAMPLE FILE DOWNLOAD
    # ---------------------------
    st.subheader("📥 Download Sample File")

    sample_data = pd.DataFrame({
        "Party": ["BJP", "INC", "AAP", "BJP", "INC"],
        "State": ["Gujarat", "Maharashtra", "Delhi", "UP", "Rajasthan"],
        "Constituency": ["Surat", "Mumbai North", "New Delhi", "Varanasi", "Jaipur"],
        "Gender": ["Male", "Male", "Female", "Male", "Female"],
        "Votes": [500000, 300000, 250000, 600000, 350000]
    })

    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_bulk_data.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # ---------------------------
    # FILE UPLOAD
    # ---------------------------
    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "json"])

    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                df = pd.read_json(file)

            st.write("📊 Preview:", df.head())

            # Prediction
            preds = model.predict(df)
            df["Prediction"] = preds
            df["Prediction"] = preds
            df["Result"] = df["Prediction"].map({1: "Winner 🎉", 0: "Loser ❌"})


            st.success("✅ Prediction Completed")
            st.write(df[['Party', 'State', 'Constituency', 'Votes', 'Result']])

            # Download result
            st.download_button(
                "📥 Download Results CSV",
                df.to_csv(index=False),
                "prediction_results.csv"
            )

        except Exception as e:
            st.error("❌ The uploaded dataset is not accurate according to the model")
            st.error(f"Error: {e}")

# ---------------------------
# 3. DATA ANALYSIS
# ---------------------------
elif selected == "Data Analysis":
    st.title("📊 Data Analysis")

    # Load + Rename (IMPORTANT)
    df = pd.read_csv("indian_election_data.csv")

    df.rename(columns={
        'st_name': 'State',
        'pc_name': 'Constituency',
        'cand_name': 'Candidate_Name',
        'cand_sex': 'Gender',
        'partyname': 'Party',
        'totvotpoll': 'Votes'
    }, inplace=True)

    # ---------------------------
    # Create Winner column (IMPORTANT)
    # ---------------------------
    df['Winner'] = df.groupby(['Constituency', 'year'])['Votes'] \
                     .transform(lambda x: (x == x.max()).astype(int))

    # ---------------------------
    # 1. Party-wise wins
    # ---------------------------
    fig, ax = plt.subplots()
    df['Party'].value_counts().head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top Parties")
    st.pyplot(fig)

    # ---------------------------
    # 2. State-wise results
    # ---------------------------
    fig, ax = plt.subplots()
    df['State'].value_counts().head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top States")
    st.pyplot(fig)

    # ---------------------------
    # 3. Votes distribution
    # ---------------------------
    fig, ax = plt.subplots()
    sns.histplot(df['Votes'], bins=30, ax=ax)
    ax.set_title("Votes Distribution")
    st.pyplot(fig)

    # ---------------------------
    # 4. Gender vs Winning
    # ---------------------------
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='Winner', data=df, ax=ax)
    ax.set_title("Gender vs Winning")
    st.pyplot(fig)

    # ---------------------------
    # 5. Votes vs Winner (REPLACED Age ❌)
    # ---------------------------
    fig, ax = plt.subplots()
    sns.boxplot(x='Winner', y='Votes', data=df, ax=ax)
    ax.set_title("Votes vs Winning")
    st.pyplot(fig)

    # ---------------------------
    # 6. Correlation Heatmap (FIXED)
    # ---------------------------
    fig, ax = plt.subplots()
    sns.heatmap(
        df.select_dtypes(include=['int64', 'float64']).corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

