# Iris Flower Classification Web App

## Files
- `app.py`: Streamlit website
- `iris_model.pkl`: your trained Random Forest model
- `requirements.txt`: Python dependencies

## Run locally
1. Install Python.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Deploy with Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload all three files to the repository root.
3. Open https://share.streamlit.io/
4. Sign in with GitHub and choose the repository.
5. Set the main file path to `app.py`, then deploy.

Note: The model was saved with scikit-learn 1.6.1, so requirements pins that version.
