# 🌿 AI-Powered IoT Twin for Smart Sustainable Buildings

## 🌏 Overview
This project implements an **AI-powered Digital Twin** for smart buildings that integrates **IoT simulation**, **machine learning**, and **real-time analytics**.  
The goal is to forecast and optimize three key sustainability metrics essential for modern buildings:

1. ⚡ **Energy Consumption (kW)**  
2. 🫧 **CO₂ Concentration (ppm)**  
3. 🛋️ **Thermal Comfort Index (/100)**  

An interactive **Streamlit dashboard** provides live predictions, dynamic visualizations, and actionable recommendations.

---

## 🎯 Key Objectives
- Simulate real-world building conditions using IoT-style data  
- Forecast energy use, air quality, and comfort levels using ML models  
- Compute a unified **Smart Score** for sustainability tracking  
- Provide **data-driven recommendations** for improving comfort and efficiency  

---

## 🧠 Core Features

| Feature | Description |
|----------|--------------|
| 🔍 **AI Forecasting** | Predicts energy, CO₂, and comfort using RandomForest or Linear Regression models |
| 🌟 **Smart Score** | Weighted index that measures overall building sustainability |
| 📈 **Dynamic Visualization** | Real-time charts for predictions, correlations, and feature importance |
| 🧩 **Scenario Simulator** | “What-if” analysis for occupancy, temperature, and humidity changes |
| 🏢 **Multi-Building Support** | Aggregates metrics from multiple virtual buildings |
| 📥 **CSV Upload** | Load your own building dataset to replace the default demo data |
| 📄 **PDF Report Generator** | One-click export of sustainability metrics and recommendations |

---

## ⚙️ System Workflow

1. **Data Input**
   - Uses `smart.csv` (7-day demo dataset)
   - Supports CSV uploads with columns like:
     ```
     hour, temperature, humidity, occupancy, ventilation_on, appliance_load_kw, co2_ppm
     ```

2. **Machine Learning Models**
   - Trains models (RandomForest or LinearRegression)
   - Predicts energy, CO₂, and comfort index

3. **AI Twin Simulation**
   - Adjusts predictions based on slider inputs (temperature, humidity, occupancy)
   - Dynamically updates graphs and sustainability scores

4. **Recommendations**
   - Generates real-time tips to reduce energy usage and improve comfort
   - Calculates estimated carbon savings per day

---

## 📊 Example Metrics

| Metric | Description | Example |
|---------|-------------|----------|
| **Energy (kW)** | Predicted energy use | `2.46 kW` |
| **CO₂ (ppm)** | Indoor air concentration | `960 ppm` |
| **Comfort (/100)** | Thermal comfort score | `80.5` |
| **Smart Score** | Sustainability performance | `85.2` |
| **Carbon Savings** | Estimated daily reduction | `2.1 kg/day` |

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **UI Framework** | Streamlit |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn (RandomForest, LinearRegression) |
| **Visualization** | matplotlib |
| **PDF Export** | FPDF |
| **Language** | Python 3.9+ |

---

## 🏗️ Project Structure
📁 ai_iot_twin/
│
├── app.py # Main Streamlit application
├── smart.csv # Sample IoT dataset (7 days)
├── requirements.txt # Dependencies
├── README.md # Documentation
└── .streamlit/config.toml # Dashboard theme (dark mode)


---

## 💻 Run Locally

```bash
git clone https://github.com/<your-username>/smart-building-twin.git
cd smart-building-twin
pip install -r requirements.txt
streamlit run app.py
Then open http://localhost:8501 in your browser.

☁️ Deploy on Streamlit Cloud
Push your project to GitHub

Go to https://share.streamlit.io

Select your repository

Set Main file path: app.py

Click Deploy 🚀

🚀 Future Improvements
Integration with real IoT sensors (MQTT, REST API)

Add LSTM-based time-series forecasting

Cloud-based model retraining and analytics

Carbon footprint visualization dashboards

Energy-Comfort optimization via reinforcement learning

👤 Author
Arun KC
🌐 AI | IoT | Smart Systems | Sustainability

📜 License
This project is licensed under the MIT License.
You’re free to modify, use, or extend it for your own projects.