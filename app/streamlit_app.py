import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px

st.set_page_config(
    page_title="Football Transfer Value Predictor",
    page_icon="⚽",
    layout="wide"
)

#custom styling
st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg, #061426 0%, #0B1F3A 50%, #061426 100%);
        color: #FFFFFF;
    }

    [data-testid="stMarkdownContainer"] h1 {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #FFFFFF !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #B8C7DC;
    }

    .stTextInput input,
    .stNumberInput input {
        background-color: #142944;
        color: #FFFFFF;
        border: 1px solid #29476B;
        border-radius: 10px;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #142944;
        color: #FFFFFF;
        border: 1px solid #29476B;
        border-radius: 10px;
    }

    
    /* =========================
   HERO SECTION
   ========================= */

.hero-section {
    background: linear-gradient(135deg, #0d1b2a, #142944);
    padding: 35px 40px;
    border-radius: 18px;
    margin-bottom: 25px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.hero-title {
    font-size: 38px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 20px;
    font-weight: 500;
    color: #8ecae6;
    margin-bottom: 12px;
}

.hero-description {
    font-size: 15px;
    color: #c7d3df;
    max-width: 750px;
    margin: 0 auto;
    line-height: 1.6;
}


/* =========================
   PROJECT STAT CARDS
   ========================= */

.stat-card {
    background-color: #142944;
    padding: 22px 15px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
    min-height: 100px;
}

.stat-number {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}

.stat-label {
    font-size: 14px;
    color: #aebdcc;
}
/* =========================
   SECTION CARDS
   ========================= */

.section-card {
    background-color: #142944;
    padding: 18px 22px;
    border-radius: 14px;
    margin-top: 20px;
    margin-bottom: 15px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 5px;
}

.section-description {
    font-size: 14px;
    color: #aebdcc;
}

</style>
""", unsafe_allow_html=True)

# Load trained model
model = joblib.load(
    "models/transfer_value_model.pkl"
)

#load player lookup dataset
player_lookup=pd.read_csv(
    "data/player_lookup.csv"
)
# Load SHAP background data
shap_background = np.load(
    "models/shap_background.npy"
)
# Create model-agnostic SHAP explainer
explainer = shap.Explainer(
    model.named_steps["model"].predict,
    shap_background
)




# =========================
# APP HEADER
# =========================

st.markdown("""
<div class="hero-section">
<div class="hero-title">
⚽ Football Transfer Value Predictor
</div>
<div class="hero-subtitle">
Machine-learning powered football market valuation and player scouting
</div>
<div class="hero-description">
Predict player market values using performance statistics,
player characteristics and historical market data.
</div>
</div>
""", unsafe_allow_html=True)


# =========================
# PROJECT STATS
# =========================

stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">80K+</div>
        <div class="stat-label">Player-Season-Records</div>
    </div>
    """, unsafe_allow_html=True)

with stat2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">0.798</div>
        <div class="stat-label">Test R² Score</div>
    </div>
    """, unsafe_allow_html=True)

with stat3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">2012–2025</div>
        <div class="stat-label">Data Coverage</div>
    </div>
    """, unsafe_allow_html=True)



# =========================
# PLAYER INFORMATION
# =========================

st.markdown("""
<div class="section-card">
    <div class="section-title">👤 Player Information</div>
    <div class="section-description">
        Enter the player's basic information.
    </div>
</div>
""", unsafe_allow_html=True)


# Row 1
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=15.0,
        max_value=45.0,
        value=24.0,
        step=0.1
    )

with col2:
    position = st.selectbox(
        "Position",
        ["Goalkeeper", "Defender", "Midfielder", "Attacker", "Unknown"]
    )


# Row 2
col1, col2 = st.columns(2)

with col1:
    sub_position = st.selectbox(
        "Sub-position",
        [
            "Centre-Back",
            "Left-Back",
            "Right-Back",
            "Defensive Midfield",
            "Central Midfield",
            "Attacking Midfield",
            "Left Midfield",
            "Right Midfield",
            "Left Winger",
            "Right Winger",
            "Centre-Forward",
            "Second Striker",
            "Goalkeeper",
            "Unknown"
        ]
    )

with col2:
    foot = st.selectbox(
        "Preferred Foot",
        ["Left", "Right", "Both", "Unknown"]
    )


# Row 3
col1, col2 = st.columns(2)

with col1:
    height = st.number_input(
        "Height (cm)",
        min_value=150,
        max_value=220,
        value=180,
        step=1
    )

with col2:
    country_of_citizenship = st.text_input(
        "Nationality",
        value="Unknown"
    )

# =========================
# PLAYER PERFORMANCE
# =========================

st.markdown("""
<div class="section-card">
    <div class="section-title">⚽ Player Performance</div>
    <div class="section-description">
        Enter the player's performance statistics for the season.
    </div>
</div>
""", unsafe_allow_html=True)


# Row 1
col1, col2 = st.columns(2)

with col1:
    appearances = st.number_input(
        "Appearances",
        min_value=0,
        max_value=60,
        value=30,
        step=1
    )

with col2:
    minutes_played = st.number_input(
        "Minutes Played",
        min_value=0,
        max_value=6000,
        value=2500,
        step=1
    )


# Row 2
col1, col2 = st.columns(2)

with col1:
    goals = st.number_input(
        "Goals",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

with col2:
    assists = st.number_input(
        "Assists",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


# Row 3
col1, col2 = st.columns(2)

with col1:
    yellow_cards = st.number_input(
        "Yellow Cards",
        min_value=0,
        max_value=30,
        value=3,
        step=1
    )

with col2:
    red_cards = st.number_input(
        "Red Cards",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

# =========================
# MARKET VALUE HISTORY
# =========================

st.markdown("""
<div class="section-card">
    <div class="section-title">📈 Market Value History</div>
    <div class="section-description">
        Enter information about the player's previous market value.
    </div>
</div>
""", unsafe_allow_html=True)


# Row 1
col1, col2 = st.columns(2)

with col1:
    season = st.number_input(
        "Season",
        min_value=2012,
        max_value=2025,
        value=2024,
        step=1
    )

with col2:
    previous_market_value = st.number_input(
        "Previous Market Value (€)",
        min_value=0,
        max_value=200_000_000,
        value=5_000_000,
        step=100_000,
        format="%d"
    )


# Row 2
col1, col2 = st.columns(2)

with col1:
    previous_value_growth = st.number_input(
        "Previous Value Growth (%)",
        min_value=-100.0,
        max_value=1000.0,
        value=0.0,
        step=1.0
    )

with col2:
    career_seasons = st.number_input(
        "Career Seasons",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )

# =========================
# PREVIOUS-SEASON PERFORMANCE
# =========================

st.markdown("""
<div class="section-card">
    <div class="section-title">📊 Previous-Season Performance</div>
    <div class="section-description">
        Enter the player's statistics from the previous season.
    </div>
</div>
""", unsafe_allow_html=True)


# Row 1
col1, col2 = st.columns(2)

with col1:
    previous_season_minutes = st.number_input(
        "Previous-Season Minutes",
        min_value=0,
        max_value=6000,
        value=2500,
        step=1
    )

with col2:
    previous_season_goals = st.number_input(
        "Previous-Season Goals",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


# Row 2
col1, col2 = st.columns(2)

with col1:
    previous_season_assists = st.number_input(
        "Previous-Season Assists",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

st.write("---")

if st.button("Predict Market Value"):

    # Calculate per-90 statistics
    if minutes_played > 0:
        goals_per_90 = goals / minutes_played * 90
        assists_per_90 = assists / minutes_played * 90
    else:
        goals_per_90 = 0
        assists_per_90 = 0

    # Convert percentage to decimal
    growth_decimal = previous_value_growth / 100

    # Create input for the model
    player_data = pd.DataFrame([{
        "season": season,
        "appearances": appearances,
        "minutes_played": minutes_played,
        "goals": goals,
        "assists": assists,
        "yellow_cards": yellow_cards,
        "red_cards": red_cards,
        "height_in_cm": height,
        "age": age,
        "goals_per_90": goals_per_90,
        "assists_per_90": assists_per_90,
        "previous_market_value_eur": previous_market_value,
        "previous_value_growth": growth_decimal,
        "career_seasons": career_seasons,
        "previous_season_minutes": previous_season_minutes,
        "previous_season_goals": previous_season_goals,
        "previous_season_assists": previous_season_assists,
        "position": position,
        "sub_position": sub_position,
        "foot": foot,
        "country_of_citizenship": country_of_citizenship
    }])

    # Make prediction
    prediction_log = model.predict(player_data)

    # Convert prediction back from log scale to euros
    predicted_value = np.expm1(prediction_log[0])

    # Prepare player data for SHAP
    player_processed = model.named_steps["preprocessor"].transform(
        player_data
    )

    

    # Calculate SHAP values
    player_shap = explainer(
        player_processed
    )

    #Calculate projected change
    if previous_market_value > 0:
        projected_change = (
            (predicted_value - previous_market_value)
            / previous_market_value
        ) * 100
    else:
        projected_change = 0

    
    # Display prediction card
    st.markdown(
        f"""
        <div style="
            background-color: #0B1F3A;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
        ">
            <h3 style="color: white;">Predicted Market Value</h3>
            <h1 style="color: white;">€{predicted_value:,.0f}</h1>
            <p style="color: #D9E2F2;">Estimated player market value</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display value comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Previous Market Value",
            f"€{previous_market_value:,.0f}",
            help="The player's previous market value used as a reference."
        )

    with col2:
        st.metric(
            "Predicted Market Value",
            f"€{predicted_value:,.0f}",
            help="The market value estimated by the machine learning model."
        )

    with col3:
        st.metric(
            "Projected Change",
            f"{projected_change:+.1f}%",
            help="Estimated percentage change from the previous market value."
        )

    st.subheader("📊 Market Value Comparison")

    chart_data = pd.DataFrame({
        "Type": ["Previous Value", "Predicted Value"],
        "Market Value": [previous_market_value, predicted_value]
    })

    fig = px.bar(
        chart_data,
        x="Type",
        y="Market Value",
        text="Market Value",
        labels={"Market Value": "Market Value (€)"}
    )

    fig.update_traces(
        texttemplate="€%{text:,.0f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>€%{y:,.0f}<extra></extra>"
    )

    fig.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=40),

        xaxis=dict(
            title="",
            tickfont=dict(size=14)
        ),

        yaxis=dict(
            title="Market Value (€)",
            tickformat=",.0f",
            gridcolor="rgba(255,255,255,0.15)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # SHAP explanation


    st.subheader("🧠 Why did the model make this prediction?")

    shap_values = player_shap.values[0]

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    # Create SHAP table
    shap_table = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values
    })


    # Convert encoded feature names back to original feature groups
    def clean_feature_name(feature):

        if feature.startswith("num__"):
            return feature.replace("num__", "")

        if feature.startswith("cat__position_"):
            return "Position"

        if feature.startswith("cat__sub_position_"):
            return "Sub-position"

        if feature.startswith("cat__foot_"):
            return "Preferred Foot"

        if feature.startswith("cat__country_of_citizenship_"):
            return "Nationality"

        return feature


    shap_table["Feature"] = shap_table["Feature"].apply(clean_feature_name)


    # Group one-hot encoded features back together
    shap_table = (
        shap_table
        .groupby("Feature", as_index=False)["SHAP Value"]
        .sum()
    )

    shap_table["Absolute Impact"] = shap_table["SHAP Value"].abs()


    # Get the 10 most influential features
    top_shap = (
        shap_table
        .sort_values("Absolute Impact", ascending=False)
        .head(10)
        .sort_values("SHAP Value")
    )


    # Create chart
    fig = px.bar(
        top_shap,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        text="SHAP Value",
        title="Top Factors Influencing the Prediction"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        xaxis_title="Impact on Prediction",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=50, t=60, b=20),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.15)",
            zerolinecolor="rgba(255,255,255,0.3)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# PLAYER LOOKUP
# =========================

st.write("---")

st.markdown("""
<div class="section-card">
    <div class="section-title">🔎 Player Lookup</div>
    <div class="section-description">
        Select a real player and season from the dataset to predict their market value.
    </div>
</div>
""", unsafe_allow_html=True)


# Select player
player_names = sorted(
    player_lookup["name"].dropna().unique()
)

selected_player = st.selectbox(
    "Select Player",
    player_names
)


# Get seasons available for selected player
player_seasons = player_lookup[
    player_lookup["name"] == selected_player
]["season"].sort_values().tolist()


selected_season = st.selectbox(
    "Select Season",
    player_seasons
)


# Get selected player-season
selected_row = player_lookup[
    (player_lookup["name"] == selected_player) &
    (player_lookup["season"] == selected_season)
].iloc[0]


if st.button("Predict Selected Player"):

    # Features used by the trained model
    model_features = [
        "season",
        "appearances",
        "minutes_played",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "height_in_cm",
        "age",
        "goals_per_90",
        "assists_per_90",
        "previous_market_value_eur",
        "previous_value_growth",
        "career_seasons",
        "previous_season_minutes",
        "previous_season_goals",
        "previous_season_assists",
        "position",
        "sub_position",
        "foot",
        "country_of_citizenship"
    ]

    # Prepare player data for the model
    lookup_data = selected_row[model_features].to_frame().T

    # Make prediction
    prediction_log = model.predict(lookup_data)

    predicted_value = np.expm1(prediction_log[0])

    # Prepare player data for SHAP
    lookup_processed = model.named_steps["preprocessor"].transform(
        lookup_data
    )

    # Calculate SHAP values
    lookup_shap = explainer(
        lookup_processed
    )


    # Display player
    st.subheader(
        f"⚽ {selected_player} — {selected_season} Season"
    )

    # Player statistics overview
    st.subheader("📊 Player Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Age",
            f"{selected_row['age']:.1f}"
        )

    with col2:
        st.metric(
            "Appearances",
            f"{selected_row['appearances']:.0f}"
        )

    with col3:
        st.metric(
            "Goals",
            f"{selected_row['goals']:.0f}"
        )

    with col4:
        st.metric(
            "Assists",
            f"{selected_row['assists']:.0f}"
        )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Minutes",
            f"{selected_row['minutes_played']:,.0f}"
        )

    with col2:
        st.metric(
            "Previous Value",
            f"€{selected_row['previous_market_value_eur']:,.0f}"
        )

    with col3:
        st.metric(
            "Goals / 90",
            f"{selected_row['goals_per_90']:.2f}"
        )

    with col4:
        st.metric(
            "Assists / 90",
            f"{selected_row['assists_per_90']:.2f}"
        )


    # Display prediction
    st.markdown(
        f"""
        <div style="
            background-color: #0B1F3A;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
        ">
            <h3 style="color: white;">Predicted Next-Season Market Value</h3>
            <h1 style="color: white;">€{predicted_value:,.0f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Display actual value if available
    actual_value = selected_row["target_market_value_eur"]

    if pd.notna(actual_value):

        difference = predicted_value - actual_value

        error_percentage = (
            difference / actual_value
        ) * 100 if actual_value != 0 else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Actual Next-Season Value",
                f"€{actual_value:,.0f}"
            )

        with col2:
            st.metric(
                "Predicted Value",
                f"€{predicted_value:,.0f}"
            )

        with col3:
            st.metric(
                "Prediction Difference",
                f"{error_percentage:+.1f}%"
            )
        # Predicted vs Actual chart
        st.subheader("📊 Predicted vs Actual Market Value")

        comparison_data = pd.DataFrame({
            "Type": ["Actual Value", "Predicted Value"],
            "Market Value": [actual_value, predicted_value]
        })

        fig = px.bar(
            comparison_data,
            x="Type",
            y="Market Value",
            text="Market Value",
            labels={"Market Value": "Market Value (€)"}
        )

        fig.update_traces(
            texttemplate="€%{text:,.0f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>€%{y:,.0f}<extra></extra>"
        )

        fig.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=40),
            xaxis=dict(
                title="",
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title="Market Value (€)",
                tickformat=",.0f",
                gridcolor="rgba(255,255,255,0.15)"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    else:

        st.info(
            "An actual next-season market value is not available for this player-season."
        )

    # SHAP explanation
    st.subheader("🧠 Why did the model make this prediction?")

    shap_values = lookup_shap.values[0]

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    shap_table = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values
    })

    # Convert encoded feature names back to original feature groups
    def clean_lookup_feature_name(feature):

        if feature.startswith("num__"):
            return feature.replace("num__", "")

        if feature.startswith("cat__position_"):
            return "Position"

        if feature.startswith("cat__sub_position_"):
            return "Sub-position"

        if feature.startswith("cat__foot_"):
            return "Preferred Foot"

        if feature.startswith("cat__country_of_citizenship_"):
            return "Nationality"

        return feature

    shap_table["Feature"] = shap_table["Feature"].apply(
        clean_lookup_feature_name
    )

    # Group one-hot encoded features together
    shap_table = (
        shap_table
        .groupby("Feature", as_index=False)["SHAP Value"]
        .sum()
    )

    shap_table["Absolute Impact"] = shap_table["SHAP Value"].abs()

    # Get the 10 most influential features
    top_lookup_shap = (
        shap_table
        .sort_values("Absolute Impact", ascending=False)
        .head(10)
        .sort_values("SHAP Value")
    )

    # Create SHAP chart
    fig = px.bar(
        top_lookup_shap,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        text="SHAP Value",
        title="Top Factors Influencing the Prediction"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        xaxis_title="Impact on Prediction",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=50, t=60, b=20),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.15)",
            zerolinecolor="rgba(255,255,255,0.3)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
    """
    <div style="
        background-color: #142944;
        padding: 12px 18px;
        border-radius: 10px;
        margin-top: 10px;
        text-align: center;
    ">
        <span style="color: #FFFFFF;">
            <b>How to read this chart:</b>
            Positive values pushed the prediction higher,
            while negative values pushed the prediction lower.
        </span>
    </div>
    """,
    unsafe_allow_html=True
    )

# =========================
# PLAYER SCREENING
# =========================

st.write("---")

st.markdown("""
<div class="section-card">
    <div class="section-title">🔎 Player Screening</div>
    <div class="section-description">
        Filter players based on age, performance, position and market value.
    </div>
</div>
""", unsafe_allow_html=True)

# Screening filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    screening_position = st.selectbox(
        "Position",
        ["All", "Goalkeeper", "Defender", "Midfield", "Attack"],
        key="screening_position"
    )

with col2:
    max_age = st.slider(
        "Maximum Age",
        min_value=16.0,
        max_value=40.0,
        value=25.0,
        step=1.0,
        key="screening_age"
    )

with col3:
    max_market_value = st.number_input(
        "Maximum Market Value (€)",
        min_value=0,
        max_value=200_000_000,
        value=10_000_000,
        step=500_000,
        format="%d",
        key="screening_market_value"
    )
with col4:
    screening_season= st.selectbox(
        "Season",
        ["All"] + sorted(
            player_lookup["season"].dropna().unique(),
            reverse=True
        ),
        key="screening_season"
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    min_appearances = st.number_input(
        "Minimum Appearances",
        min_value=0,
        max_value=60,
        value=20,
        step=1,
        key="screening_appearances"
    )

with col2:
    min_minutes = st.number_input(
        "Minimum Minutes",
        min_value=0,
        max_value=6000,
        value=1000,
        step=100,
        key="screening_minutes"
    )

with col3:
    min_goals = st.number_input(
        "Minimum Goals",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        key="screening_goals"
    )

with col4:
    min_assists = st.number_input(
        "Minimum Assists",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        key="screening_assists"
    )
# Apply screening filters

screened_players = player_lookup.copy()

# Season filter
if screening_season != "All":
    screened_players = screened_players[
        screened_players["season"] == screening_season
    ]

# Position filter
if screening_position != "All":
    screened_players = screened_players[
        screened_players["position"] == screening_position
    ]

# Age filter
screened_players = screened_players[
    screened_players["age"] <= max_age
]

# Market value filter
screened_players = screened_players[
    screened_players["previous_market_value_eur"] <= max_market_value
]

# Performance filters
screened_players = screened_players[
    (screened_players["appearances"] >= min_appearances) &
    (screened_players["minutes_played"] >= min_minutes) &
    (screened_players["goals"] >= min_goals) &
    (screened_players["assists"] >= min_assists)
]

# Remove rows with missing values in important fields
screened_players = screened_players.dropna(
    subset=[
        "name",
        "age",
        "position",
        "appearances",
        "minutes_played",
        "goals",
        "assists",
        "previous_market_value_eur"
    ]
)
# Display number of matching players
st.write(
    f"### 🔎 {len(screened_players):,} players found"
)

# Display matching players
display_columns = [
    "name",
    "season",
    "age",
    "position",
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90",
    "previous_market_value_eur",
    "predicted_next_value"
]
# Model features required for prediction
model_features = [
    "season",
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "height_in_cm",
    "age",
    "goals_per_90",
    "assists_per_90",
    "previous_market_value_eur",
    "previous_value_growth",
    "career_seasons",
    "previous_season_minutes",
    "previous_season_goals",
    "previous_season_assists",
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship"
]

# Predict next-season market value for screened players
if len(screened_players) > 0:

    with st.spinner("Predicting future market values..."):

        screening_predictions = model.predict(
            screened_players[model_features]
        )

        screened_players["predicted_next_value"] = np.expm1(
            screening_predictions
        )
screening_results = screened_players[display_columns].copy()

# Format market value
screening_results["previous_market_value_eur"] = (
    screening_results["previous_market_value_eur"]
    .apply(lambda x: f"€{x:,.0f}")
)
screening_results["predicted_next_value"] = (
    screening_results["predicted_next_value"]
    .apply(lambda x: f"€{x:,.0f}")
)

# Round performance statistics
screening_results["age"] = screening_results["age"].round(1)
screening_results["goals_per_90"] = screening_results["goals_per_90"].round(2)
screening_results["assists_per_90"] = screening_results["assists_per_90"].round(2)

# Rename columns for the app
screening_results = screening_results.rename(columns={
    "name": "Player",
    "season": "Season",
    "age": "Age",
    "position": "Position",
    "appearances": "Appearances",
    "minutes_played": "Minutes",
    "goals": "Goals",
    "assists": "Assists",
    "goals_per_90": "Goals/90",
    "assists_per_90": "Assists/90",
    "previous_market_value_eur": "Previous Market Value",
    "predicted_next_value": "Predicted Next Value"
})

st.dataframe(
    screening_results,
    use_container_width=True,
    hide_index=True
)
