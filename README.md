# Customer Segmentation and Churn Pattern Analytics in European Banking

An interactive Streamlit dashboard that shows who leaves a European retail bank and which customer segments carry the most risk. It uses a dataset of 10,000 customers across France, Germany and Spain.

## What it does

The sidebar filters customers by geography, gender, age, credit score, number of products and active status. Every chart and KPI updates with the filters.

The dashboard has five tabs:

1. Overview: churn rate by geography, number of products, member status and card ownership, plus a geography by age group heatmap.
2. Demographics: churn by age group, gender and tenure, and the age distribution of churned and retained customers.
3. Financial profile: churn by credit score band and zero balance, box plots of balance and salary, and each variable's correlation with churn.
4. Segmentation: KMeans clustering on features you choose, with 2 to 8 segments. A table ranks the segments by churn rate.
5. Data: the filtered table, with a CSV download.

## Key findings

The overall churn rate is 20.4% (2,037 of 10,000 customers). Churned customers held about €185.6M in balances.

- Germany churns at 32.4%, about double France (16.2%) and Spain (16.7%).
- Women churn more than men (25.1% vs 16.5%), and the gap shows up in every country.
- Age is the strongest single signal. Churn is 7.6% under 30, 30.8% at 40 to 49, and peaks at 56.0% at 50 to 59.
- Customers with two products are the most stable (7.6%). One product churns at 27.7%, three at 82.7% and four at 100%, though the last two groups are small (266 and 60 customers).
- Inactive members churn at 26.9%, against 14.3% for active members.
- Churned customers have higher average balances (€91k vs €73k) and are older on average (45 vs 37). Credit score, tenure, salary and card ownership show little relationship with churn.

## Suggested retention actions

- Focus on customers aged 40 to 60, especially in Germany.
- Investigate why customers with three or four products leave, since that pattern looks like a product or pricing issue.
- Run re-engagement campaigns for inactive members.
- Look at the German market separately, including female customers there (37.6% churn).

## How to run

1. Put `app.py` and `European_Bank.csv` in the same folder.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Start the dashboard:
   ```
   streamlit run app.py
   ```
4. Open http://localhost:8501 in your browser.

If the CSV is not in the folder, the app shows an upload box instead.

## Tools

Python, pandas, Plotly, scikit-learn and Streamlit.
