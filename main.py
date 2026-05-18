import pandas as pd
import xgboost as xgb
import numpy as np
import os

# 1. Setup Data & Model
df = pd.read_csv('retail_price.csv')
top_category = df['product_category_name'].value_counts().idxmax() # Correct
df_cat = df[df['product_category_name'] == top_category].copy()

X = df_cat[['unit_price']]
y = df_cat['qty']

# Train XGBoost
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
model.fit(X, y)

# 2. Bargaining Simulation Logic
def calculate_payoff(price, cost_price=50):
    """
    Seller Payoff = (Price - Cost) * Predicted Qty
    Buyer Payoff = (Max Willingness to Pay - Price) * Predicted Qty
    """
    predicted_qty = model.predict(np.array([[price]]))[0]
    predicted_qty = max(0, predicted_qty) # Ensure no negative demand
    
    seller_profit = (price - cost_price) * predicted_qty
    # Assuming max willingness to pay is 20% above mean price for the simulation
    buyer_surplus = (df_cat['unit_price'].max() - price) * predicted_qty
    
    return seller_profit, buyer_surplus

# 3. Find the "Hot Deal" (Nash Point)
prices_to_test = np.linspace(df_cat['unit_price'].min(), df_cat['unit_price'].max(), 50)
results = []

for p in prices_to_test:
    s_payoff, b_payoff = calculate_payoff(p)
    results.append({"price": p, "seller": s_payoff, "buyer": b_payoff, "total_utility": s_payoff + b_payoff})

sim_df = pd.DataFrame(results)
optimal_deal = sim_df.loc[sim_df['total_utility'].idxmax()]

print(f"--- Bargaining Simulation Results for {top_category} ---")
print(f"Optimal 'Hot Deal' Price: ${optimal_deal['price']:.2f}")
print(f"Projected Seller Profit: ${optimal_deal['seller']:.2f}")