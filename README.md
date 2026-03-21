# 🛡️ CyberShield — Cyberbullying Detection App

AI-powered cyberbullying detection with **BERTweet** predictions and **LIME** word-level explanations.

---

## 📁 Project Structure

```
cyberbullying-app/
│
├── app.py                        ← Main Streamlit application
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── .streamlit/
│   └── config.toml               ← Streamlit theme & server config
│
├── static/
│   └── css/
│       └── style.css             ← Full custom CSS (dark cyberpunk theme)
│
└── saved_models/                 ← ⚠️ Copy from your Colab after training
    ├── bertweet_model/           ← BERTweet weights + tokenizer
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer_config.json
    │   └── vocab.json
    ├── tfidf_vectorizer.pkl      ← TF-IDF vectorizer (for LIME)
    ├── label_encoder.pkl         ← LabelEncoder
    └── best_model.pkl            ← LightGBM (for LIME explanations)
```

---

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/cyberbullying-app.git
git push -u origin main
```

### Step 2 — Upload saved_models
Since model files are large, use **Git LFS** or upload via Streamlit Secrets:
```bash
# Option A: Git LFS
git lfs install
git lfs track "saved_models/**"
git add .gitattributes saved_models/
git commit -m "Add models"
git push
```

### Step 3 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo → Branch: `main` → Main file: `app.py`
4. Click **Deploy**

✅ Your app is live at `https://YOUR_USERNAME-cyberbullying-app-app-XXXX.streamlit.app`

---

## 🤗 Deploy on Hugging Face Spaces (Free)

### Step 1 — Create a Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Name: `cyberbullying-detector`
4. SDK: **Streamlit**
5. Visibility: Public
6. Click **Create Space**

### Step 2 — Upload files
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/cyberbullying-detector
cd cyberbullying-detector

# Copy all project files
cp -r /path/to/cyberbullying-app/* .

# Push
git add .
git commit -m "Deploy CyberShield app"
git push
```

### Step 3 — Upload large model files via HF Hub
```python
from huggingface_hub import HfApi
api = HfApi()

# Upload entire saved_models folder
api.upload_folder(
    folder_path="saved_models",
    repo_id="YOUR_USERNAME/cyberbullying-detector",
    repo_type="space",
    path_in_repo="saved_models"
)
```

✅ Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/cyberbullying-detector`

---

## 💻 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Copy saved_models from your Colab:
# In Colab: from google.colab import files; files.download('saved_models.zip')
# Then unzip into this folder

# Run the app
streamlit run app.py
```

App opens at: http://localhost:8501

---

## 📤 Export saved_models from Google Colab

Add this cell at the end of your notebook:

```python
import shutil
shutil.make_archive('saved_models_export', 'zip', '.', 'saved_models')
from google.colab import files
files.download('saved_models_export.zip')
```

Then unzip into `cyberbullying-app/saved_models/`

---

## ⚙️ Demo Mode

If `saved_models/` is not present, the app runs in **Demo Mode**:
- Uses keyword-heuristic mock predictions
- Still shows full UI (LIME bars, charts, severity, preprocessing trace)
- Perfect for testing the UI before models are loaded

---

## 🎨 Features

| Feature | Description |
|---|---|
| BERTweet Prediction | 95%+ accurate Twitter-native model |
| LIME Explanation | Word-level contribution bars |
| Probability Chart | All 6 class probabilities visualised |
| Severity Score | 1–10 scale (Safe → Critical) |
| Category Hints | Auto-detects ethnicity/age/gender/religion signals |
| Preprocessing Trace | Shows exactly what the model sees |
| Example Buttons | 6 pre-loaded test cases |
| Demo Mode | Works without saved models |

---

## 🔑 Notes

- The app automatically detects GPU/CPU
- LIME uses LightGBM (fast) not BERTweet for explanations
- Severity scoring logic is identical to the notebook (Cell 14)
- Custom CSS is loaded from `static/css/style.css`
