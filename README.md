# ⚽ Football Transfer Value Predictor

A machine-learning project for predicting football player market values and supporting player scouting.

🔗 **Live Demo:** https://football-transfer-value-predictor-nr5zuglmflfnvfuneq4cex.streamlit.app/

## 📌 Project Overview

This project uses historical football player data to predict a player's market value for the following season.

The model combines:

- Player performance statistics
- Playing time and appearances
- Player characteristics
- Previous market value
- Historical market-value changes
- Position and nationality information

The project also includes an interactive Streamlit application that allows users to explore player valuations and screen players based on selected criteria.

## 🚀 Key Features

- Predicts next-season football player market values
- Uses historical player performance and market-value data
- Uses a chronological train/validation/test split to reduce data leakage
- Includes player performance, physical, positional and market-value features
- Provides model evaluation using MAE, RMSE and R²
- Uses SHAP to explain individual predictions
- Includes an interactive Streamlit dashboard
- Includes a player screening tool for scouting based on custom criteria

## 📊 Dataset

The project uses football data from the **Transfermarkt** website, provided through the **Transfermarkt Dataset** by David Cariboo.

The dataset contains information about:

- Players
- Player market valuations
- Match appearances and statistics
- Transfers
- Matches and competitions

The main datasets used in this project are:

- `players.csv`
- `player_valuations.csv`
- `appearances.csv`
- `transfers.csv`
- `games.csv`

The raw dataset is not included in this repository because of its large size. It can be obtained from the original dataset source and placed inside the `data/` directory.

## 🧠 Methodology

The prediction task is formulated as a **next-season market value prediction problem**.

For each player-season, information from the current season is used to predict the player's market value in the following season. This helps avoid using future information when training the model.

The main stages of the project are:

1. Clean and combine player, valuation, appearance and match data.
2. Aggregate performance statistics at player-season level.
3. Create historical features such as previous market value and previous-season performance.
4. Split the data chronologically into training, validation and test sets.
5. Train a machine-learning regression model on the logarithm of market value.
6. Evaluate predictions on unseen seasons.
7. Use SHAP to understand which features influence individual predictions.
8. Integrate the model into an interactive Streamlit application.

## 🧩 Model Features

The model uses a combination of numerical and categorical features.

### Performance & Usage

- Appearances
- Minutes played
- Goals
- Assists
- Yellow cards
- Red cards
- Goals per 90 minutes
- Assists per 90 minutes

### Player Characteristics

- Age
- Height
- Position
- Sub-position
- Preferred foot
- Country of citizenship

### Historical Market & Performance Features

- Previous market value
- Previous market-value growth
- Career seasons
- Previous-season minutes
- Previous-season goals
- Previous-season assists
- Season

## 🤖 Machine Learning Model

The project uses a `HistGradientBoostingRegressor` from Scikit-learn.

The model is trained on the logarithm of market value using `log1p()` to reduce the impact of the highly skewed distribution of football player market values.

The dataset is split chronologically:

- **Training:** Seasons up to 2022
- **Validation:** 2023
- **Test:** 2024

This chronological split ensures that the model is evaluated on later seasons that were not available during training.

## 📈 Model Performance

The final model achieved the following results on the unseen test set:

| Metric | Result |
|---|---:|
| R² | **0.798** |
| MAE | **€2.60M** |
| RMSE | **€6.59M** |

The R² score indicates that the model explains approximately 79.8% of the variance in the test-set target values.

Because football market values vary substantially between players, the MAE and RMSE provide additional information about the typical and larger prediction errors.

## 🔎 Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to analyse how individual features contribute to model predictions.

SHAP is used to:

- Identify the features that have the greatest influence on predictions
- Understand why individual player predictions differ
- Examine positive and negative feature contributions
- Make the model's predictions easier to interpret

The analysis showed that historical market value, age and playing time are among the features with substantial influence on the model's predictions.

## 🕵️ Player Scouting

The Streamlit application includes a player screening tool that allows users to filter players using criteria such as:

- Position
- Maximum age
- Maximum previous market value
- Season
- Minimum appearances
- Minimum minutes played
- Minimum goals
- Minimum assists

The model then provides predicted next-season market values for the players matching the selected criteria.

## 🌐 Streamlit Application

The project includes an interactive Streamlit dashboard for exploring the model and its predictions.

The application allows users to:

- Enter player information and generate a market-value prediction
- Look up existing players from the dataset
- View model predictions alongside player information
- Explore factors influencing individual predictions
- Screen players using custom scouting criteria
- Compare players based on performance and historical market-value data

The dashboard is designed to make the machine-learning model accessible without requiring users to interact directly with the Python code.

## 🛠️ Technology Stack

- **Python** — data processing and machine learning
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computing
- **Scikit-learn** — machine-learning model development and evaluation
- **SHAP** — model explainability
- **Plotly** — interactive visualisations
- **Streamlit** — interactive web application
- **Matplotlib & Seaborn** — data visualisation
- **Jupyter Notebook** — exploratory data analysis
- **Git & GitHub** — version control and project management

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/hayderlajil06-blip/football-transfer-value-predictor.git
cd football-transfer-value-predictor
## Create virtual environment: 
python -m venv .venv
## On windows activate it with:
.venv\Scripts\activate
## Install dependencies:
pip install -r requirements.txt
## Add the dataset by:
## Download the original Transfermarkt dataset and place the required CSV files inside the data/ directory.
## Run the streamlit application:
streamlit run app/streamlit_app.py

## 📁 Project Structure

```text
football-transfer-value-predictor/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── Raw Transfermarkt CSV files
│
├── models/
│   ├── transfer_value_model.pkl
│   └── shap_background.npy
│
├── notebooks/
│   └── 01-data-exploration.ipynb
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt

## 🚧 Limitations & Future Improvements

### Current Limitations

- Football market values are influenced by factors that are difficult to capture using historical statistics alone.
- The model predicts market value rather than actual transfer fees.
- Prediction errors are larger for very highly valued players because their market values can vary substantially.
- The raw dataset is not included in the repository because of its large file size.

### Future Improvements

- Add additional contextual features such as league strength and team performance.
- Incorporate transfer-fee data as a separate prediction target.
- Experiment with additional machine-learning algorithms such as XGBoost.
- Improve the model's treatment of extreme high-value players.
- Expand the Streamlit dashboard with additional scouting and comparison tools.

## 📚 Dataset Source & Attribution

The football data used in this project comes from the **Transfermarkt Dataset** by David Cariboo.

Source:
https://github.com/dcaribou/transfermarkt-datasets

The dataset is sourced from Transfermarkt and is provided under a CC0 license through the associated Kaggle dataset.

This project is for educational and portfolio purposes. The data is not redistributed as part of this repository.