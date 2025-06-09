# 💻 Laptop Price Predictor

This project is a **Streamlit-based web application** that predicts the price of a laptop based on user-selected configurations. It uses a machine learning model trained on a dataset of laptops and their respective specifications and prices.

---

## 🚀 Features

* User-friendly web interface using **Streamlit**
* Dynamically filtered inputs based on brand selection
* Calculates **PPI (Pixels Per Inch)** from resolution and screen size
* Predicts price using a trained pipeline with preprocessing and a regression model

---

## 📁 Project Structure

```
.
├── app.py              # Streamlit web app
├── Project.ipynb       # Notebook for data cleaning, EDA, and model training
├── df.pkl              # Preprocessed DataFrame used in the app
├── pipe.pkl            # Trained machine learning pipeline (with preprocessing)
├── laptop_data.csv     # Raw dataset
```

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/laptop-price-predictor.git
cd laptop-price-predictor
```

### 2. Install dependencies

Make sure you have Python 3.8+ and install required packages:

```bash
pip install -r requirements.txt
```

> *Note: You may need to create your own `requirements.txt` using:*
>
> ```bash
> pip freeze > requirements.txt
> ```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧠 Model Overview

* **Algorithm**: Linear Regression (or any regressor chosen)
* **Preprocessing**: Feature engineering for PPI, encoding categorical variables
* **Target Variable**: `Price` (log-transformed during training)

---

## 📊 Dataset Overview

* Source: Kaggle or other public dataset repositories
* Columns include: Brand, Type, RAM, CPU, GPU, Storage (HDD/SSD), Resolution, Screen Size, etc.
* Target: Price of the laptop

---

## 🛠 Customization Tips

* To improve accuracy, try other regression models (e.g., Random Forest, XGBoost)
* Add filters like year, processor generation, etc.
* Deploy this app on **Streamlit Cloud** or **Render** for public access

---

## 📬 Contact

Created by \[Om Patel] - feel free to reach out via \[[patel.k.om74@gmail.com](mailto:patel.k.om74@gmail.com)] or GitHub Issues.

---
