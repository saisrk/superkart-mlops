# SuperKart — Sales Forecasting MLOps Pipeline

End-to-end MLOps pipeline with CI/CD that forecasts `Product_Store_Sales_Total` for SuperKart.

| Component | Link |
|---|---|
| Dataset (HF Dataset Hub) | https://huggingface.co/datasets/saisrk/superkart-sales-dataset |
| Model (HF Model Hub) | https://huggingface.co/saisrk/superkart-sales-model |
| Streamlit app (HF Space) | https://huggingface.co/spaces/saisrk/superkart-sales-forecaster |

## Pipeline (`.github/workflows/pipeline.yml`)
1. **register-dataset** → `superkart_project/model_building/data_register.py`
2. **data-prep** → `superkart_project/model_building/prep.py`
3. **model-training** → `superkart_project/model_building/train.py` (pushes `metrics.json` to `main`)
4. **deploy-hosting** → `superkart_project/hosting/hosting.py`

Runs on every push to `main`. Requires the repository secret `HF_TOKEN` (Hugging Face write token).
