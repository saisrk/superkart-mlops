---
title: SuperKart Sales Forecaster
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Predict product-store revenue for SuperKart
---

# SuperKart Sales Forecaster

Streamlit front-end for the SuperKart sales forecasting model. The model is loaded from the
Hugging Face Model Hub when the app starts. Enter product and store details to predict
`Product_Store_Sales_Total`, or upload a CSV for batch forecasts.

The app runs in the browser with [stlite](https://github.com/whitphx/stlite) (first load takes
~30 s). The included `Dockerfile` runs the same `app.py` on a Docker Space or any container host.
