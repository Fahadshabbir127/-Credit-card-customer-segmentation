# 💳 Credit Card Customer Segmentation

Customer segmentation on credit card behavioral data using **K-Means Clustering**, an unsupervised machine learning technique. This project identifies distinct customer groups based on spending, payment, and credit usage patterns to support data-driven marketing and risk management decisions.

🔗 **Live App:** [https://credit-card-customer-segmentation-3rdproject.streamlit.app/]
📂 **Notebook:** `Credit_Card_Dataset_for_Clustering.ipynb`

---

## 📌 Problem Statement

Banks and financial institutions manage thousands of credit card customers with very different spending and repayment behaviors. Treating all customers the same way leads to missed opportunities (premium customers not rewarded) and unmanaged risk (high-risk customers not flagged).

This project segments customers into meaningful groups using unsupervised learning, enabling targeted marketing, personalized offers, and proactive risk monitoring.

---

## 📊 Dataset

- **Source:** [Credit Card Dataset for Clustering (Kaggle)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
- **Size:** 8,950 customers, 18 behavioral features
- **Features include:** Balance, Purchases, Cash Advance, Credit Limit, Payments, Purchase Frequency, Tenure, and more (6 months of credit card usage data)

---

## 🛠️ Workflow

1. **Data Cleaning** — Missing values in `CREDIT_LIMIT` and `MINIMUM_PAYMENTS` filled using median; duplicates checked
2. **EDA** — Distribution analysis and correlation heatmap to understand feature relationships
3. **Feature Engineering** — Log transformation (`log1p`) applied to highly skewed features to improve clustering quality
4. **Feature Scaling** — StandardScaler applied so all features contribute equally to distance-based clustering
5. **Optimal Cluster Selection** — Elbow Method (WCSS) and Silhouette Score used to determine the ideal number of clusters (**k = 4**)
6. **Model Training** — K-Means clustering applied to segment customers into 4 groups
7. **Cluster Profiling** — Each cluster analyzed and labeled with a business-meaningful identity
8. **Visualization** — PCA used to visualize clusters in 2D space
9. **Deployment** — Interactive Streamlit app built for real-time segment prediction

---

## 🎯 Customer Segments Identified

| Cluster | Segment Name | Key Traits | Recommendation |
|---|---|---|---|
| 0 | **Low Activity / Dormant Customers** | Low balance, low purchases, minimal card usage | Re-engagement campaigns, cashback offers |
| 1 | **Cash Advance Revolvers (Risky)** | High balance, low purchases, highest cash advance usage, near-zero full payment | Structured repayment plans, risk monitoring |
| 2 | **Premium / High-Value Customers** | Highest credit limit, highest purchases & payments | Loyalty rewards, credit limit increases |
| 3 | **Budget-Conscious / Responsible Shoppers** | Moderate purchases, highest full-payment ratio | Reward responsible repayment behavior |

---

## 📈 Model Evaluation

- **Method used:** Elbow Method (WCSS) + Silhouette Score
- **Optimal clusters:** 4
- Clusters validated against real customer samples and a synthetic test case — predictions aligned correctly with expected segment behavior

---

## 🚀 Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Deployment:** Streamlit Cloud

---

## 💻 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Fahadshabbir127/-Credit-card-customer-segmentation.git
cd -Credit-card-customer-segmentation

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Repository Structure

```
credit-card-customer-segmentation/
├── README.md
├── Credit_Card_Dataset_for_Clustering.ipynb
├── app.py
├── kmeans_model.pkl
├── scaler.pkl
├── log_columns.pkl
└── requirements.txt
```

---

## 👤 Author

**Fahad Shabbir**
Part of a Data Science & Machine Learning portfolio covering classification, regression, and clustering.

- 🔗 [Telco Customer Churn Prediction](https://github.com/Fahadshabbir127/Telco-customer-churn-prediction) (Classification)
- 🔗 [House Price Prediction](https://github.com/Fahadshabbir127/House-price-prediction) (Regression)
- 🔗 Credit Card Customer Segmentation (Clustering) — this project
