"""Script to create sample data for testing the price prediction system"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_samples=1000, output_path="data/raw/prices.csv"):
    """
    Generates sample price prediction data.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the generated data
    """
    np.random.seed(42)
    
    # Generate sample features
    data = {
        'area': np.random.normal(1500, 500, n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'bathrooms': np.random.choice([1, 2, 3, 4], n_samples),
        'age': np.random.exponential(10, n_samples),
        'location': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'condition': np.random.choice(['excellent', 'good', 'fair', 'poor'], n_samples),
    }
    
    # Generate target price based on features (with some noise)
    # Price = base + area*100 + bedrooms*50000 + bathrooms*30000 - age*2000 + location_factor + condition_factor + noise
    location_factors = {'A': 50000, 'B': 30000, 'C': 10000, 'D': 0}
    condition_factors = {'excellent': 20000, 'good': 10000, 'fair': 0, 'poor': -10000}
    
    base_price = 100000
    price = (
        base_price +
        data['area'] * 100 +
        data['bedrooms'] * 50000 +
        data['bathrooms'] * 30000 -
        data['age'] * 2000 +
        [location_factors[loc] for loc in data['location']] +
        [condition_factors[cond] for cond in data['condition']] +
        np.random.normal(0, 20000, n_samples)  # Noise
    )
    
    # Ensure prices are positive
    price = np.maximum(price, 50000)
    
    data['price'] = price
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {n_samples} samples")
    print(f"💾 Saved to {output_path}")
    print(f"\n📊 Data preview:")
    print(df.head())
    print(f"\n📈 Statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    generate_sample_data()

