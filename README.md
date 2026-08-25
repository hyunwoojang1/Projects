# Hyunwoo Jang 👋
Data Science Major | Quantitative Finance & Manufacturing Analytics

Data Science major with hands-on experience in  
WRDS-based financial data analysis and quantitative modeling.  
Currently expanding into production, quality, and manufacturing analytics
for engineering-focused roles.

## About Me
I am a Data Science major with a strong interest in applying data-driven methods to both financial and industrial problems. My academic and project experience has primarily focused on quantitative finance, where I have worked extensively with WRDS datasets such as CRSP and Compustat to analyze stock returns, build predictive models, and evaluate portfolio performance using statistical and machine learning techniques. Through these projects, I developed a solid foundation in data engineering, feature construction, and model evaluation. Building on this background, I am now expanding my focus toward production, quality, and manufacturing analytics, with the goal of applying data science to real-world operational challenges such as process monitoring, KPI analysis, and defect pattern identification. I am particularly interested in roles where analytical insights can directly support engineering decisions and operational improvements.

## Education
Penn State University  
B.S. in Data Science  

### Relevant Coursework
- Data Science & Analytics (Python, SQL, Pandas)
- Machine Learning  (scikit-learn)
- Statistical Inference 
- Big Data Systems (Spark, Hadoop)
- Optimization & Linear Programming
- Database Systems (SQL, NO-SQL)

## Projects

### 🤖 AI Agent — Quant Investment Agent (Live)
> [`Quant Investment Agent/`](./Quant%20Investment%20Agent/)

Fully automated daily investment analysis system. Runs every morning at 08:00 KST via GitHub Actions and delivers a Kakao Talk report — no manual intervention required.

**Key Features**
- **QQQ/TQQQ Timing Signal** — ML ensemble (XGBoost + LightGBM + LSTM + Transformer) trained on 30 years of data with 5-fold walk-forward CV. Outputs 4-class actionable signal: TQQQ / QLD / QQQ / Cash
- **Analog Finder** — Finds the 10 most similar historical periods to today's market and shows TQQQ 1Y/3Y returns from those dates
- **Market Regime (HMM)** — Gaussian Hidden Markov Model classifies the current market into Bull / Bear / Transition / Crash regimes
- **Fear & Greed Index** — Custom 7-component model (validated against CNN index: MAE 9.2pt, correlation 0.78)
- **Sector Rotation** — 11 SPDR sector ETF momentum + breadth scoring
- **Macro Score** — Fed rate cycle, yield curve, credit spread composite

**Tech Stack:** Python · PyTorch · XGBoost · LightGBM · hmmlearn · yfinance · FRED API · Kakao API · GitHub Actions

---

### 📈 Quantitative Finance & Financial Data Projects

- WRDS-Based Stock Return Prediction
  - Financial data extraction using CRSP & Compustat
  - Technical indicators and macroeconomic features
  - Machine learning classification models and performance comparison

- **Factor-Based Portfolio Performance Analysis**
  - CAPM and multi-factor models
  - Risk-adjusted performance evaluation

---

### 🏭 Manufacturing / Production / Quality Analytics

- **Production Process KPI Analysis**
  - Analysis of operational KPIs
  - Identification of inefficiencies and abnormal patterns

- **Quality Defect Pattern Analysis**
  - Defect classification and frequency analysis
  - Root cause exploration using data


## Contact
- 📧 Email: hfj5102@psu.edu
- 💼 LinkedIn: https://www.linkedin.com/in/hyunwoo-jang-921320288/
