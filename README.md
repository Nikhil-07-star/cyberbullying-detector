---
title: CyberShield Cyberbullying Detector
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: mit
pinned: false
---

# 🛡️ CyberShield — Cyberbullying Detection App

AI-powered cyberbullying detection with **BERTweet** predictions and **LIME** word-level explanations.

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚙️ Demo Mode

If `saved_models/` is not present, the app runs in Demo Mode using keyword-heuristic predictions. Full UI still works.

---

## 🎨 Features

| Feature | Description |
|---|---|
| BERTweet Prediction | 95%+ accurate Twitter-native model |
| LIME Explanation | Word-level contribution bars |
| Probability Chart | All 6 class probabilities visualised |
| Severity Score | 1–10 scale (Safe → Critical) |
| Preprocessing Trace | Shows exactly what the model sees |
| Demo Mode | Works without saved models |

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
