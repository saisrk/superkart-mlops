"""SuperKart Sales Forecaster — Streamlit front-end served from Hugging Face Spaces."""
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# The model repo is injected as a Space variable by hosting.py
MODEL_REPO = os.getenv("MODEL_REPO", "your-hf-username/superkart-sales-model")
MODEL_FILE = "superkart_sales_model_v1.joblib"

FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
            "Product_Type", "Product_MRP", "Store_Establishment_Year", "Store_Size",
            "Store_Location_City_Type", "Store_Type", "Product_Category"]

# Profiles of the four existing SuperKart stores (used to pre-fill store inputs)
STORES = {
    "OUT001": dict(year=1987, size="High",   tier="Tier 2", type="Supermarket Type1"),
    "OUT002": dict(year=1998, size="Small",  tier="Tier 3", type="Food Mart"),
    "OUT003": dict(year=1999, size="Medium", tier="Tier 1", type="Departmental Store"),
    "OUT004": dict(year=2009, size="Medium", tier="Tier 2", type="Supermarket Type2"),
}
PRODUCT_TYPES = ["Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household",
                 "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks", "Breads",
                 "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"]
SIZES, TIERS = ["High", "Medium", "Small"], ["Tier 1", "Tier 2", "Tier 3"]
STORE_TYPES = ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]


@st.cache_resource
def load_model():
    """Download the registered model from the Hugging Face Model Hub (cached)."""
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)


st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Sales Forecaster")
st.caption(f"Predicts total revenue of a product in a store · model: `{MODEL_REPO}`")

model = load_model()
single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with single_tab:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Product details")
        category = st.selectbox("Product category", ["Food", "Drinks", "Non-Consumable"])
        product_type = st.selectbox("Product type", PRODUCT_TYPES)
        sugar_options = ["No Sugar"] if category == "Non-Consumable" else ["Low Sugar", "Regular", "No Sugar"]
        sugar = st.selectbox("Sugar content", sugar_options)
        weight = st.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.65, step=0.1)
        mrp = st.number_input("Product MRP", min_value=10.0, max_value=400.0, value=147.0, step=1.0)
        area = st.slider("Allocated display area (ratio)", 0.0, 0.35, 0.07, 0.005)
    with col2:
        st.subheader("Store details")
        store_id = st.selectbox("Store", list(STORES) + ["Custom store"])
        preset = STORES.get(store_id, STORES["OUT004"])
        custom = store_id == "Custom store"
        year = st.number_input("Establishment year", 1950, 2030, preset["year"], disabled=not custom)
        size = st.selectbox("Store size", SIZES, index=SIZES.index(preset["size"]), disabled=not custom)
        tier = st.selectbox("City type", TIERS, index=TIERS.index(preset["tier"]), disabled=not custom)
        stype = st.selectbox("Store type", STORE_TYPES, index=STORE_TYPES.index(preset["type"]), disabled=not custom)

    # Collect the inputs into a single-row DataFrame in the model's feature schema
    input_df = pd.DataFrame([{
        "Product_Weight": weight, "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": area, "Product_Type": product_type,
        "Product_MRP": mrp, "Store_Establishment_Year": int(year),
        "Store_Size": size, "Store_Location_City_Type": tier,
        "Store_Type": stype, "Product_Category": category,
    }])[FEATURES]

    st.markdown("**Model input**")
    st.dataframe(input_df, hide_index=True, use_container_width=True)

    if st.button("Predict sales", type="primary"):
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted Product_Store_Sales_Total: **{prediction:,.2f}**")

with batch_tab:
    st.write("Upload a CSV with these columns (a `Product_Id` column, if present, is used to "
             "derive `Product_Category`):")
    st.code(", ".join(FEATURES))
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        batch = pd.read_csv(upload)
        if "Product_Category" not in batch and "Product_Id" in batch:
            batch["Product_Category"] = batch["Product_Id"].str[:2].map(
                {"FD": "Food", "NC": "Non-Consumable", "DR": "Drinks"})
        if "Product_Sugar_Content" in batch:
            batch["Product_Sugar_Content"] = batch["Product_Sugar_Content"].replace({"reg": "Regular"})
        missing = [c for c in FEATURES if c not in batch]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            batch["Predicted_Sales"] = model.predict(batch[FEATURES]).round(2)
            st.dataframe(batch, use_container_width=True)
            st.metric("Total forecast revenue", f"{batch['Predicted_Sales'].sum():,.0f}")
            st.download_button("Download predictions", batch.to_csv(index=False),
                               "superkart_predictions.csv", "text/csv")
