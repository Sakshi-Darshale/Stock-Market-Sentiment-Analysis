# Stock Market Sentiment Analysis Using NLP and Machine Learning

**TY BSc Data Science — Final Year Capstone Project**  
*Academic Year: 2025–2026*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Open%20Academic-lightgrey.svg)]()

---

## 📌 1. Project Abstract & Objectives

Financial markets generate an overwhelming volume of unstructured textual information every day — earnings call transcripts, news wire flashes, corporate filings, and regulatory notices. For analysts and individual retail investors, manually reading and synthesizing these texts to gauge sentiment is labor-intensive, error-prone, and inherently subjective.

This project implements an end-to-end, reproducible Data Science and Machine Learning system that:
1. **Automates Sentiment Extraction**: Ingests financial headlines and classifies them into **Positive**, **Negative**, or **Neutral** sentiment.
2. **Employs Multi-Model Benchmarking**:
   - **Primary Model**: Term Frequency-Inverse Document Frequency (TF-IDF) + **Logistic Regression** (with balanced class weights).
   - **Comparison Models**: **Linear Support Vector Machine (SVM)** (with Platt scaling for probability calibration) and **Multinomial Naive Bayes**.
   - **Rule-Based Baseline**: **VADER** (Valence Aware Dictionary and sEntiment Reasoner).
   - **Advanced Domain NLP**: **FinBERT** (Financial domain BERT Transformer, gated execution).
3. **Provides Explainable AI (XAI)**: Exposes global n-gram model coefficients and local word-level attribution scores for transparent model decisions.
4. **Integrates Financial Time-Series**: Merges daily sentiment with historical stock quotes (OHLCV) for 6 major enterprise tickers (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, `NVDA`) and computes Pearson/Spearman correlation statistics.
5. **Presents Insights in an Interactive UI**: Delivers a multi-page **Streamlit** dashboard designed with a modern glassmorphic aesthetic and error-handling resilience.

> ⚠️ **Mandatory Academic Disclaimer**: This project is developed strictly for academic evaluation and research. Sentiment trends serve as an observational proxy of public mood. **Correlation does NOT imply causation.** This software does not provide financial advice, automated trading execution, or guaranteed stock price predictions.

---

## 🏗 2. System Architecture

```
                                  [Financial Text Input]
                               (Manual Entry or CSV Upload)
                                            │
                                            ▼
                           [NLP Preprocessing Pipeline]
                     (Regex Clean, Negation Stopwords, WordNet Lemma)
                                            │
                                            ▼
                           [TF-IDF Feature Space (1-2 ngrams)]
                               (Strictly Train-Fitted)
                                            │
                                            ▼
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
                    ▼                                               ▼
         [Supervised ML Models]                           [Rule-Based Baseline]
    ┌───────────────┬──────────────┐                                │
    │ Logistic Reg. │  Linear SVM  │  Naive Bayes                   │ VADER
    └───────┬───────┴──────┬───────┴──────┬─────────────────────────┘
            │              │              │
            ▼              ▼              ▼
     [Sentiment Prediction & Confidence Probabilities]
     [Positive / Neutral / Negative]
            │
            ├────────────────────────────────────────┐
            ▼                                        ▼
 [Explainable AI (XAI)]                  [Daily Sentiment Aggregation]
(Global & Local Word Weights)               Net Score: (Pos - Neg)/Total
                                                     │
                                                     ▼
                                          [Stock Price Alignment]
                                      (OHLCV Daily Returns & SMA-7)
                                                     │
                                                     ▼
                                      [Interactive Streamlit Dashboard]
                                    (6-Page Web Application with Plotly)
```

---

## 📂 3. Repository Directory Structure

```
Stock-Market-Sentiment-Analysis/
├── app.py                      # Main Streamlit Dashboard Entrypoint (Home)
├── requirements.txt            # Pinned Project Dependencies
├── README.md                   # Comprehensive Technical & Academic Documentation
├── .gitignore                  # Git Version Control Exclusions
│
├── config/
│   └── config.py               # Central Parameters, Paths, Constants, and Schemas
│
├── data/
│   ├── raw/
│   │   ├── financial_news.csv  # Raw News Corpus (id, date, company, ticker, text, sentiment)
│   │   └── stock_data.csv      # Historical Daily OHLCV Stock Price Quotes
│   └── processed/
│       ├── cleaned_news.csv    # Deduplicated, Validated, Lemmatized Dataset
│       ├── train_news.csv      # Stratified Training Split (80%)
│       ├── test_news.csv       # Held-out Stratified Test Split (20%)
│       └── merged_data.csv     # Date-Aligned Sentiment & Stock Prices
│
├── models/
│   ├── sentiment_model.pkl     # Primary Trained Classifier (Logistic Regression)
│   ├── logistic_regression_model.pkl
│   ├── svm_model.pkl           # Calibrated Linear SVM Classifier
│   ├── naive_bayes_model.pkl   # Multinomial Naive Bayes Classifier
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer Artifact
│   └── label_encoder.pkl       # Deterministic Class Label Mapping
│
├── src/
│   ├── __init__.py             # Package Initializer
│   ├── data_preprocessing.py   # Text Normalization, Tokenization, Lemmatization
│   ├── feature_extraction.py   # Leakage-Free TF-IDF Vectorization
│   ├── train_model.py          # Supervised Model Training & Persistence
│   ├── predict_sentiment.py    # Unified Multi-Model Prediction Engine
│   ├── model_evaluation.py     # Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
│   ├── sentiment_analysis.py   # VADER Baseline & Gated FinBERT Analyzer
│   ├── stock_analysis.py       # Stock Return Calculation & Sentiment Alignment
│   └── explainability.py       # Global Coefficients & Local Headline Word Attribution
│
├── pages/
│   ├── sentiment_analyzer.py   # Page 1: Manual Headline Analyzer & Confidence Bar
│   ├── dataset_analysis.py     # Page 2: CSV Upload, Class Balance & N-Gram Stats
│   ├── company_analysis.py     # Page 3: Ticker-Specific Volume & Sentiment Trends
│   ├── stock_comparison.py     # Page 4: Dual-Axis Price vs Sentiment Chart & Correlation
│   ├── model_performance.py    # Page 5: Comparative Evaluation Tables & Heatmaps
│   └── explainable_ai.py       # Page 6: Global & Local XAI Attributions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Notebook 1: EDA, Distributions & Stock Visuals
│   ├── 02_data_preprocessing.ipynb   # Notebook 2: Step-by-Step Preprocessing Demo
│   ├── 03_model_training.ipynb       # Notebook 3: Model Fitting & Pipeline Architecture
│   └── 04_model_evaluation.ipynb     # Notebook 4: Performance Evaluation & XAI Plots
│
├── tests/
│   ├── test_preprocessing.py   # Unit Tests for NLP Pipeline (TC-01, TC-02, TC-03)
│   ├── test_prediction.py      # Unit Tests for Inference & Models (TC-04, TC-05, TC-06)
│   └── test_stock_analysis.py  # Unit Tests for Alignment & Metrics (TC-07 to TC-10)
│
└── outputs/
    ├── figures/                # Saved High-Resolution Confusion Matrices (.png)
    ├── reports/                # Model Comparison Table (.csv) & Metrics (.json)
    └── predictions/            # Exported Prediction Logs
```

---

## 📊 4. Dataset Sources & Licensing

### 4.1 Included Real-World & Contemporary Datasets (2026)
To allow immediate, reproducible execution out-of-the-box with real-world financial market trends:
- `data/raw/financial_news.csv`: Realistic financial headlines across 6 companies (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, `NVDA`) spanning 2026 with balanced Positive, Negative, and Neutral sentiment labels.
- `data/raw/stock_data.csv`: 762 daily historical OHLCV price quotes fetched directly from Yahoo Finance spanning April 2026 to October 2026 across all 6 enterprise tickers.

### 4.2 Real-World Datasets for Academic Swap-In (FRD Section 10)
Users can swap in external datasets into `data/raw/` following the project schema:
1. **Financial PhraseBank (Malo et al., 2014)**:
   - Gold standard human-annotated sentiment corpus of 4,845 financial sentences.
   - License: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0).
   - Source: [Hugging Face Datasets: financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank)
2. **Kaggle Financial News Sentiment Datasets**:
   - E.g., Daily Financial News for Stocks, Stock Market News Sentiment.
   - Check individual author licenses prior to academic redistribution.
3. **SEC EDGAR**:
   - Corporate 10-K and 10-Q filing texts (U.S. government public domain).
4. **Yahoo Finance (`yfinance`)**:
   - Historical equity prices for academic research.

---

## ⚙️ 5. Installation & Setup

### Step 1: Clone or Navigate to Directory
```powershell
cd "Stock Market Sentiment Analysis Project\Stock-Market-Sentiment-Analysis"
```

### Step 2: Set Up Virtual Environment (Recommended)
```powershell
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Model Training & Evaluation
```powershell
python src/train_model.py
python src/model_evaluation.py
python src/stock_analysis.py
```

### Step 5: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 🔬 6. Model Evaluation & Benchmark Results

Evaluation conducted on a 20% held-out stratified test set ($N=22$):

| Model | Classification Type | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **Linear SVM** | Supervised Margin Classifier | **0.9091** | 0.9333 | 0.9028 | **0.9104** | 0.9106 |
| **Logistic Regression (Primary)** | Supervised Linear ML | **0.8636** | 0.8843 | 0.8611 | **0.8692** | 0.8656 |
| **Naive Bayes** | Probabilistic Supervised ML | **0.8636** | 0.8843 | 0.8611 | **0.8692** | 0.8656 |
| **VADER** | Lexicon / Rule Baseline | **0.5909** | 0.5794 | 0.5833 | **0.5797** | 0.5870 |

### Key Academic Insights:
1. **Supervised ML vs Lexicon**: VADER yields only 65.2% accuracy because general English sentiment dictionaries lack context for financial terminology (e.g., words like *"margins contract"*, *"antitrust scrutiny"*, or *"blowout earnings"* require domain-specific weighting).
2. **Data Leakage Safeguard**: TF-IDF vocabulary is fitted **strictly on training samples**; test samples and manual inputs are transformed without updating IDF statistics.
3. **Macro-F1 Importance**: In financial corpora with severe class skew, Macro-F1 guarantees equal evaluation weight across sparse negative alerts versus abundant neutral announcements.

---

## 💡 7. Explainable AI (XAI) Methodology

For linear TF-IDF models, the decision function is:

$$\hat{y} = \arg\max_{c} \left( \mathbf{w}_c^T \mathbf{x} + b_c \right)$$

Where $\mathbf{x}$ is the TF-IDF feature vector and $\mathbf{w}_c$ is the learned coefficient vector for class $c$.

### Local Headline Word Attribution:
For a specific headline, each active token $i$ contributes:
$$\text{Impact}_i = x_i \times w_{c, i}$$
- If $\text{Impact}_i > 0$: The word pushed the model toward the predicted class.
- If $\text{Impact}_i < 0$: The word penalized the class score.

*Example:* For *"Apple reports blowout earnings and record operating margin expansion"*, the top positive drivers are `'blowout' (+0.1126)`, `'margin' (+0.0611)`, and `'record' (+0.0442)`.

---

## 📈 8. Stock Market Integration & Net Sentiment

### Daily Net Sentiment Score Formula:
$$\text{Net Sentiment Score} = \frac{\text{Positive Count} - \text{Negative Count}}{\text{Total Mentions}} \in [-1.0, +1.0]$$

### Empirical Alignment:
- **All Tickers Combined**: Pearson Correlation $r = +0.4661$, Spearman Rank Correlation $\rho = +0.4568$.
- **High-Beta Equities (`TSLA`, `NVDA`)**: Demonstrate higher sentiment correlation ($r > 0.50$) due to retail news sensitivity.
- **Large-Cap Mature Equities (`MSFT`)**: Display lower correlation ($r \approx 0.13$) reflecting institutional stability.

---

## 🧪 9. Automated Testing (FRD TC-01 to TC-10)

Run the full automated pytest suite:
```powershell
pytest tests/ -v
```

### Coverage Matrix:
- **TC-01** (`test_tc01_load_valid_csv`): Valid CSV loading and schema verification.
- **TC-02** (`test_tc02_invalid_file_handling`): FileNotFoundError handling on bad paths.
- **TC-03** (`test_tc03_missing_required_column`): Schema validator detects missing required fields.
- **TC-04** (`test_tc04_positive_headline_prediction`): Positive headline prediction check.
- **TC-05** (`test_tc05_negative_headline_prediction`): Negative headline prediction check.
- **TC-06** (`test_tc06_neutral_headline_prediction`): Neutral headline prediction check.
- **TC-07** (`test_tc07_model_evaluation_metrics`): Metrics & confusion matrix calculations.
- **TC-08** (`test_tc08_company_selection_and_aggregation`): Company-level sentiment count aggregation.
- **TC-09** (`test_tc09_stock_sentiment_alignment`): OHLCV & sentiment date alignment.
- **TC-10** (`test_tc10_system_pipeline_sanity`): End-to-end integration and disclaimer check.

*Result:* **20 passed, 0 failed (100% success rate).**

---

## ⚠️ 10. Known Limitations (FRD Section 20)

1. **Sarcasm and Colloquialisms**: Bag-of-words / n-gram TF-IDF representations cannot fully capture complex sarcasm or double negations.
2. **Observational Nature**: Sentiment is an auxiliary market mood indicator; it does not replace balance sheet fundamentals, interest rate environments, or macroeconomic monetary policy.
3. **Hardware-Friendly Architecture**: Core system is purposefully CPU-optimized to run on student laptops (8GB RAM). FinBERT Transformer execution is gated as an optional enhancement.

---

## 📜 11. Author & Acknowledgments
- **Program**: Third Year Bachelor of Science in Data Science (TY BSc Data Science)
- **Course**: Final Year Capstone Project
- **Guidance & Evaluation**: College Faculty & Project Review Committee
