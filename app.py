"""Streamlit app: Bengaluru house price prediction.

Model artifacts are produced by train_model.py:
  banglore_home_prices_model.pickle  sklearn Pipeline (scaler + regressor)
  columns.json                       feature order used at training time
  metrics.json                       model selection report shown in the UI
"""

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Bengaluru House Price Estimator",
    page_icon="₹",
    layout="centered",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp { background: #faf6ee; }
html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; color: #2b2723; }

.block-container { max-width: 860px; padding-top: 4.2rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

.stMarkdown code, code {
  font-family: 'JetBrains Mono', monospace !important; font-size: 0.82em;
  background: #f0e7d7 !important; color: #8a3b2a !important;
  padding: 0.1em 0.35em !important; border-radius: 4px !important;
}

table.report { width: 100%; border-collapse: collapse; margin-top: 0.8rem;
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
table.report th { text-align: right; font-weight: 600; color: #6b6257;
  border-bottom: 1px solid #d9cdb8; padding: 0.45rem 0.5rem; }
table.report th:first-child, table.report td:first-child { text-align: left; }
table.report td { text-align: right; padding: 0.42rem 0.5rem; border-bottom: 1px solid #ece3d2; }
table.report tr.best td { background: #f6eddd; font-weight: 600; }

.eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: #a8432f; margin-bottom: 0.4rem;
}
h1.title { font-size: 2.25rem; line-height: 1.15; margin: 0 0 0.4rem 0; font-weight: 600; }
p.lede { color: #6b6257; font-size: 1.02rem; margin: 0 0 1.6rem 0; max-width: 40rem; }

.result-card {
  background: #fffdf8; border: 1px solid #e3d9c6; border-radius: 14px;
  padding: 1.5rem 1.6rem; box-shadow: 0 1px 0 #e3d9c6, 0 10px 24px -18px rgba(60,40,20,0.45);
}
.result-card .label {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: #8a7f72;
}
.result-card .value { font-size: 2.3rem; font-weight: 600; margin: 0.25rem 0 0.1rem 0; }
.result-card .sub { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #6b6257; }

.meta-row { display: flex; gap: 1.4rem; margin-top: 1.2rem; flex-wrap: wrap; }
.meta-row div { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #8a7f72;
  letter-spacing: 0.08em; text-transform: uppercase; }
.meta-row span { display: block; font-size: 0.95rem; color: #2b2723; letter-spacing: 0;
  text-transform: none; margin-top: 0.15rem; }

footer, #MainMenu { visibility: hidden; }
.note { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #8a7f72; margin-top: 1.4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open(ROOT / "banglore_home_prices_model.pickle", "rb") as handle:
        model = pickle.load(handle)
    columns = json.loads((ROOT / "columns.json").read_text())["data_columns"]
    metrics = json.loads((ROOT / "metrics.json").read_text())
    return model, columns, metrics


model, data_columns, metrics = load_artifacts()
locations = data_columns[3:]

st.markdown('<div class="eyebrow">Module 5 · Model deployment</div>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Bengaluru house price estimator</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="lede">Enter the property details. The price is predicted by the model chosen '
    'in <code>train_model.py</code> from '
    f'{metrics["rows_clean"]:,} cleaned listings across {metrics["n_locations"]} locations.</p>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.form("property"):
        location = st.selectbox(
            "Location", locations, index=locations.index("whitefield"), format_func=str.title
        )
        area = st.number_input("Total area (sqft)", min_value=300, max_value=10000, value=1200, step=50)
        bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=10, value=2, step=1)
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        submitted = st.form_submit_button("Estimate price", use_container_width=True)

with right:
    if submitted:
        row = np.zeros(len(data_columns))
        row[0], row[1], row[2] = area, bath, bhk
        if location in data_columns:
            row[data_columns.index(location)] = 1

        price = max(float(model.predict([row])[0]), 0.0)
        per_sqft = price * 1e5 / area

        st.markdown(
            f"""
            <div class="result-card">
              <div class="label">Estimated price</div>
              <div class="value">₹ {price:,.2f} L</div>
              <div class="sub">₹ {price / 100:,.2f} Cr · ₹ {per_sqft:,.0f} per sqft</div>
              <div class="meta-row">
                <div>Model<span>{metrics['selected_model']}</span></div>
                <div>Test MAE<span>₹ {metrics['holdout_MAE_lakhs']:,.0f} L</span></div>
                <div>Test RMSE<span>₹ {metrics['holdout_RMSE_lakhs']:,.0f} L</span></div>
                <div>Test R²<span>{metrics['holdout_R2']:.2f}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if area / bhk < 300:
            st.warning("Less than 300 sqft per bedroom is outside the training data range; treat this estimate with caution.")
        if bath > bhk + 2:
            st.warning("More than two extra bathrooms is unusual for this dataset; treat this estimate with caution.")
    else:
        st.markdown(
            """
            <div class="result-card">
              <div class="label">Estimated price</div>
              <div class="value">₹ -</div>
              <div class="sub">Fill the form and press Estimate price</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("How the model was chosen"):
    st.markdown(
        f"Candidates were tuned with `GridSearchCV` and compared with 5-fold cross-validation "
        f"on the training split; the winner was refit on all {metrics['rows_clean']:,} rows. "
        f"Test figures come from the 20% holdout split, scored before the refit."
    )
    rows = "".join(
        f"<tr class='{'best' if i == 0 else ''}'>"
        f"<td>{row['model']}</td>"
        f"<td>{row['cv_MAE_lakhs']:.2f}</td>"
        f"<td>{row['cv_RMSE_lakhs']:.2f}</td>"
        f"<td>{row['cv_R2']:.3f}</td>"
        f"<td>{row['test_RMSE_lakhs']:.2f}</td>"
        f"<td>{row['test_R2']:.3f}</td>"
        "</tr>"
        for i, row in enumerate(metrics["candidates"])
    )
    st.markdown(
        f"""
        <table class="report">
          <thead><tr>
            <th>Model</th><th>CV MAE (L)</th><th>CV RMSE (L)</th><th>CV R²</th>
            <th>Test RMSE (L)</th><th>Test R²</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Dataset: Kaggle Bengaluru house price data, {metrics['rows_raw']:,} raw rows. "
        f"Features: total sqft, bathrooms, bedrooms, location (one-hot). "
        f"Target: price in lakhs."
    )

st.markdown(
    '<div class="note">Estimates are statistical, not valuations. Re-run train_model.py to refresh '
    "the pickle, columns and metrics.</div>",
    unsafe_allow_html=True,
)
