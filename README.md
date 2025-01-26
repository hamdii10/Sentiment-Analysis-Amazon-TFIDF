Here’s the README file customized for your sentiment analysis project using **Streamlit** and **TF-IDF with Decision Tree Classifier**:

---

# Amazon Reviews Sentiment Classifier

This repository contains a sentiment analysis project to classify Amazon reviews as positive or negative using a **Decision Tree Classifier**. The project demonstrates text preprocessing with TF-IDF, model training, evaluation, and deployment via **Streamlit** for user interaction.

## Features

- **Machine Learning Model**: A Decision Tree Classifier trained on TF-IDF-transformed text data.
- **Text Preprocessing**: Includes tokenization, TF-IDF vectorization, and handling of imbalanced datasets using SMOTE.
- **Interactive Web Application**: A Streamlit app for real-time sentiment analysis of user-provided reviews.
- **Comprehensive Notebook**: Step-by-step implementation of preprocessing, model training, and evaluation.

## Try the App

You can try the live version of this app here: [Amazon Reviews Sentiment Classifier](#)


## Project Structure

```plaintext
amazon-reviews-sentiment-classifier/
├── models/                                     # Pre-trained model files
│   ├── decision_tree_model.pkl                 # Decision Tree Classifier
│   └── tfidf_vectorizer.pkl                    # TF-IDF vectorizer
├── notebooks/                                  # Jupyter notebooks
│   └── sentiment-analysis-amazon-tfidf.ipynb   # Main notebook for training and evaluation
├── app.py                                      # Streamlit application script
├── README.md                                   # Project overview and instructions
├── LICENSE                                     # License file (if applicable)
└── requirements.txt                            # Python dependencies for the project
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd amazon-reviews-sentiment-classifier
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

### Run Locally

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the app**:
   - Enter a review in the text box.
   - Click "Predict Sentiment" to see the classification result (Positive or Negative).

### Use the Notebook

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/sentiment-analysis.ipynb
   ```

2. Follow the steps in the notebook to:
   - Preprocess the dataset.
   - Train the Decision Tree Classifier on TF-IDF-transformed data.
   - Evaluate the model's performance.

## Dataset

The project uses the **Amazon Alexa Reviews Dataset** to classify user reviews of Alexa products.

**Dataset Details**:
- Includes user feedback and binary sentiment labels: positive and negative.
- Preprocessed for cleaning, tokenization, and balancing with SMOTE.

## Model Details

The model uses:
- **TF-IDF Vectorization**: Converts text data into numerical features.
- **Decision Tree Classifier**: A simple and interpretable machine learning model.
- **SMOTE (Synthetic Minority Oversampling Technique)**: Balances the training dataset to handle class imbalance.

Saved model files:
- `models/decision_tree_model.pkl`: The trained Decision Tree Classifier.
- `models/tfidf_vectorizer.pkl`: The TF-IDF vectorizer for preprocessing.

## Contributing

Contributions are welcome! Feel free to fork the repository, enhance features, or optimize the model. Submit pull requests with your improvements.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact

For questions or suggestions, reach out to:

- **Email**: ahmed.hamdii.kamal@gmail.com

---
