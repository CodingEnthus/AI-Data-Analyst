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
| LLM | Groq |
| Data Processing | Pandas |
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
│   ├── visualization_tools.py
│   └── helper_tools.py
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

Interactive charts are automatically saved in the `outputs/charts/` directory.

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

```
GROQ_API_KEY=your_api_key_here
```

> **Note:** Never commit your `.env` file to GitHub.

---

## ▶️ Running the Project

```bash
python app.py
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
