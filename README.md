# Next Word Prediction

Next Word Prediction web app powered by a TensorFlow/Keras LSTM neural network, FastAPI, and an interactive frontend.

## Live Demo on Render

This repository is configured for one-click or automated deployment on [Render](https://render.com).

### Render Deployment Settings:
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vishnu4063/next-word-Prediction.git
   cd next-word-Prediction
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```powershell
   uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Open in browser:**
   - Web UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
