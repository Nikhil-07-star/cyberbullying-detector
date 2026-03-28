# saved_models/

Place your trained model files here after running the Colab notebook.

## Required files:

saved_models/
├── bertweet_model/           ← from: bert_model.save_pretrained('saved_models/bertweet_model')
│   ├── config.json
│   ├── model.safetensors     (or pytorch_model.bin)
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── tfidf_vectorizer.pkl      ← from: pickle.dump(tfidf, ...)
├── label_encoder.pkl         ← from: pickle.dump(le, ...)
└── best_model.pkl            ← from: pickle.dump(trained_models['LightGBM'], ...)

## How to download from Colab (add this as a final cell):

    import shutil
    shutil.make_archive('saved_models_export', 'zip', '.', 'saved_models')
    from google.colab import files
    files.download('saved_models_export.zip')

Then unzip here:
    unzip saved_models_export.zip -d .

Without these files the app runs in Demo Mode with keyword-based predictions.
