import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_research_heatmap(model, df_cat, features):
    print("--- Generating Research-Grade Bargaining Heatmap ---")
    
    # 1. Prepare "Standard Market" context
    # We use medians for non-price features to represent a typical scenario
    context_values = df_cat[features].median()
    
    # 2. Define Price Range for the Axes
    # From 20% below to 20% above the category mean
    mean_p = df_cat['unit_price'].mean()
    price_steps = np.linspace(mean_p * 0.8, mean_p * 1.2, 25)
    
    # 3. Initialize the Matrix
    # Rows = Seller Ask, Cols = Buyer Bid
    matrix = np.zeros((len(price_steps), len(price_steps)))
    cost_price = mean_p * 0.6  # Assumption for Profit calculation
    
    for i, seller_price in enumerate(price_steps):
        for j, buyer_price in enumerate(price_steps):
            # Midpoint Negotiation Price
            agreed_price = (seller_price + buyer_price) / 2
            
            # Create the feature vector for prediction
            input_data = context_values.copy()
            input_data['unit_price'] = agreed_price
            # If total_price exists in features, update it too
            if 'total_price' in input_data:
                input_data['total_price'] = agreed_price 
            
            # Predict Quantity using the trained model
            # Convert to DataFrame to keep feature names consistent
            input_df = pd.DataFrame([input_data])
            qty = model.predict(input_df)[0]
            qty = max(0, qty)
            
            # Calculate Joint Utility (The Metric for the Heatmap)
            seller_profit = (agreed_price - cost_price) * qty
            buyer_surplus = (price_steps.max() - agreed_price) * qty
            matrix[i, j] = seller_profit + buyer_surplus

    # 4. Visualization
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        matrix, 
        xticklabels=[f"{p:.1f}" for p in price_steps], 
        yticklabels=[f"{p:.1f}" for p in price_steps],
        cmap="mako", # Professional research palette
        cbar_kws={'label': 'Joint Utility (Seller Profit + Buyer Surplus)'}
    )
    
    plt.title(f"Bargaining Equilibrium Zone: {df_cat['product_category_name'].iloc[0]}", fontsize=15)
    plt.xlabel("Buyer Bid Price ($)", fontsize=12)
    plt.ylabel("Seller Ask Price ($)", fontsize=12)
    
    # Add an annotation for the "Nash Point" (Maximum Utility)
    max_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    plt.plot(max_idx[1]+0.5, max_idx[0]+0.5, 'ro', markersize=10, label='Optimal Bargaining Point')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv('retail_price.csv')
    top_cat = df['product_category_name'].value_counts().idxmax()
    df_cat = df[df['product_category_name'] == top_cat].copy()
    
    # 2. Re-train the best model (e.g., XGBoost) for the heatmap
    research_features = ['unit_price', 'total_price', 'freight_price', 'comp_1', 'ps1', 'fp1', 'product_photos_qty']
    X = df_cat[research_features]
    y = df_cat['qty']
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
    model.fit(X, y)
    
    # 3. Generate the plot
    generate_research_heatmap(model, df_cat, research_features)