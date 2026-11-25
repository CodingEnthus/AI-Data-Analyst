# 📊 AI Data Analyst

An intelligent AI-powered Data Analyst built using **LangGraph**, **LangChain**, **Groq LLM**, **Pandas**, and **Plotly**. This application enables users to interact with datasets using natural language, perform statistical analysis, and generate interactive visualizations without writing code.

---

## 🚀 Features

- 📂 Upload CSV and Excel datasets
- 🤖 Natural language interaction with datasets
- 📈 Generate interactive visualizations
- 📊 Perform statistical analysis
- 🧠 Conversation memory using LangGraph
- 🔍 Fuzzy column name matching
- 📉 Automatic chart generation
- 💾 Interactive Plotly chart export
- ⚡ Powered by Groq LLM for fast responses

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| AI Framework | LangChain, LangGraph |
| Machine Learning | Scikit-Learn, Scipy, Joblib |
| LLM | Groq |
| Data Processing | Pandas, DuckDB |
| Visualization | Plotly |
| Environment | Python Virtual Environment |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```
AI_Data_Analyst/
│
├── agent.py
├── app.py
├── prompts.py
├── data_store.py
├── requirements.txt
│
├── tools/
│   ├── __init__.py
│   ├── dataset_tools.py
│   ├── statistics_tools.py
│   ├── duckdb_tools.py
│   ├── visualization_tools.py
│   ├── helper_tools.py
│   └── ml_tools.py
│
├── ml_engine/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── anomaly_detection.py
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── model_manager.py
│
├── models/
│   └── .gitkeep
│
├── tests/
│   ├── test_ml_engine.py
│   └── run_tests.py
│
├── outputs/
│   ├── charts/
│   └── reports/
│
└── data/
```

---

## 📊 Supported Statistical Operations

- Mean
- Median
- Mode
- Sum
- Minimum
- Maximum
- Standard Deviation
- Variance
- Count
- Unique Values

---

## 📈 Supported Visualizations

- Bar Chart
- Line Chart
- Scatter Plot
- Histogram
- Pie Chart
- Heatmap
- Box Plot
- Violin Plot

Interactive charts are automatically saved in the `outputs/charts/` directory.

---

## 🤖 Machine Learning Engine (Stage 1)

- **🚨 Anomaly Detection (Isolation Forest)**: Flags potential statistical outliers and calculates normalized severity scores ($[0, 1]$).
- **🎯 Supervised Classification (Random Forest)**: Binary and multiclass predictive modeling with stratified train/test split, zero data leakage, confusion matrix, and feature importances.
- **📈 Supervised Regression (Random Forest)**: Continuous prediction with MAE, MSE, RMSE, and $R^2$, paired with Actual vs. Predicted and Residual distribution plots.
- **🔍 Unsupervised Clustering (K-Means)**: Natural grouping on normalized features with cluster profiles, inertia, optional Elbow analysis, and 2D PCA cluster visualizations.
- **💾 Model Persistence & Schema Validation**: Serializes models with `joblib` into `models/`, validates inference schemas, and logs complete training metadata.

---

## 💬 Example Prompts

### Dataset Information

```
Load Titanic-Dataset.csv
```

```
Show dataset information
```

---

### Statistical Analysis

```
What is the average age?
```

```
Find the maximum fare.
```

```
How many unique passenger classes are there?
```

---

### SQL Analysis (DuckDB)

```
Using SQL, show the top 5 highest paying passengers who survived.
```

```
Run a SQL query to calculate average age and survival rate grouped by Passenger Class.
```

---

### Visualization

```
Create a histogram of Age.
```

```
Generate a scatter plot of Age vs Fare.
```

```
Create a pie chart of Passenger Class.
```

```
Generate a heatmap.
```

```
Create a box plot of Fare grouped by Passenger Class.
```

```
Generate a violin plot of Age by Sex.
```

---

### Machine Learning

```
Find unusual records and statistical outliers in this dataset.
```

```
Train a classification model to predict Survived and show its accuracy and feature importances.
```

```
Train a regression model to predict Fare and show the R2 score and actual vs predicted chart.
```

```
Perform clustering with 3 clusters on Age and Fare to find customer segments.
```

```
Evaluate model rf_clf_xxxxxxxx.
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/AI-Data-Analyst.git
cd AI-Data-Analyst
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
# Primary Fast Provider (Default)
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b

# Optional Local Offline Provider
# LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434
```

> **Note:** Never commit your `.env` file to GitHub.

---

## ▶️ Running the Project

### Option 1: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

### Option 2: Interactive Terminal / CLI Mode

```bash
python agent.py
```

---

## 📸 Screenshots

> Screenshots and demo GIFs will be added after the Streamlit frontend is completed.

---

## 🔮 Future Enhancements

- ✅ Streamlit Web Interface
- 📄 PDF Report Generation
- 📊 Dashboard Analytics
- 🤖 Multi-agent Workflow
- 📈 More Chart Types
- 📁 Support for Multiple Datasets
- ☁️ Cloud Deployment

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project, feel free to fork the repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Samrat Dutta**

If you found this project useful, consider giving it a ⭐ on GitHub!
