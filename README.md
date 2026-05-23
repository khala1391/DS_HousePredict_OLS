# Bangalore House Price Prediction

A machine learning web app that predicts residential property prices in Bangalore using Linear Regression — with interactive location selection on a real map and data analysis visualizations.

**Live Demo:** *(coming soon)*

---

## Features

| Feature | Description |
| --- | --- |
| **Price Prediction** | Estimates house price (Lakh INR) based on location, area, BHK, and bathrooms |
| **Interactive Map** | Select location by clicking on a real Bangalore map — 57 geocoded areas |
| **Data Analysis** | 4 visualizations: price distribution, location comparison, model accuracy, error analysis |
| **Model Evaluation** | Actual vs Predicted scatter, Residuals histogram, Mean Error, MAE |

---

## Tech Stack

- **Frontend:** Streamlit + PyDeck (interactive map)
- **ML Model:** Scikit-learn Linear Regression
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Geocoding:** GeoPy (Nominatim)

---

## Model Performance

| Metric | Value |
| --- | --- |
| R² Score | 0.78 |
| Cross-validation (5-fold) mean | 0.81 |
| Training samples | ~10,000 |

---

## Getting Started

### Installation

```bash
git clone https://github.com/khala1391/DS_HousePredict_OLS.git
cd DS_HousePredict_OLS
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`

---

## Usage

1. **Select location** — click a dot on the Bangalore map (right panel)
2. **Fill in details** — Total Square Feet, BHK, Bathrooms
3. **Click Predict Price** — get estimated price in Lakh INR
4. **Explore Analysis tab** — switch between 5 views: map, scatter, box plot, actual vs predicted, residuals

---

## Project Structure

```text
house_prediction/
├── app.py                        # Streamlit app
├── train.py                      # Data preprocessing & model training
├── requirements.txt
├── data/
│   ├── bengaluru_house_prices.csv  # Raw dataset
│   └── location_coords.csv         # Geocoded location coordinates
├── model/
│   ├── model.pkl                   # Trained LinearRegression model
│   ├── columns.json                # Feature column names
│   └── eval_data.pkl               # Test set predictions for analysis
└── scripts/
    └── geocode_locations.py        # One-time geocoding script
```

---

## Author

Yuttapong M. — [linkedin.com/in/yuttapong-m](https://www.linkedin.com/in/yuttapong-m/)
