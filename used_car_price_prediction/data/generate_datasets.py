import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_used_cars_data(num_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    brands_models = {
        "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma", "Prius"],
        "Honda": ["Civic", "Accord", "CR-V", "Pilot", "HR-V"],
        "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Edge"],
        "BMW": ["3 Series", "5 Series", "X3", "X5", "M3"],
        "Mercedes-Benz": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
        "Audi": ["A4", "A6", "Q5", "Q7", "S5"],
        "Chevrolet": ["Silverado", "Equinox", "Malibu", "Tahoe", "Camaro"],
        "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe"],
        "Nissan": ["Altima", "Sentra", "Rogue", "Pathfinder"],
        "Tesla": ["Model 3", "Model Y", "Model S"],
        "Lexus": ["RX 350", "ES 350", "NX 300", "IS 300"],
        "Subaru": ["Outback", "Forester", "Crosstrek", "Impreza"],
        "Jeep": ["Wrangler", "Grand Cherokee", "Cherokee"],
        "Porsche": ["911", "Cayenne", "Macan", "Panamera"]
    }

    brand_base_prices = {
        "Toyota": 28000, "Honda": 27000, "Ford": 32000, "BMW": 52000,
        "Mercedes-Benz": 56000, "Audi": 50000, "Chevrolet": 31000,
        "Hyundai": 24000, "Nissan": 25000, "Tesla": 48000, "Lexus": 49000,
        "Subaru": 29000, "Jeep": 36000, "Porsche": 88000
    }

    engine_options = {
        "Electric": ["Dual Electric Motor", "Single Electric Motor"],
        "Gasoline": ["1.5L I4 Turbo", "2.0L I4 Turbo", "2.5L I4", "3.0L V6 Turbo", "3.5L V6", "5.0L V8"],
        "Diesel": ["2.0L I4 Turbo Diesel", "3.0L V6 Turbo Diesel"],
        "Hybrid": ["2.0L I4 Hybrid", "2.5L I4 Hybrid"]
    }

    transmissions = ["Automatic", "Manual", "CVT", "Dual-Clutch"]

    records = []
    current_year = 2024

    for _ in range(num_samples):
        brand = random.choice(list(brands_models.keys()))
        model = random.choice(brands_models[brand])
        
        # Year distribution: 2012 to 2024 with more recent cars
        year = int(np.random.choice(np.arange(2012, 2025), p=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.12, 0.11, 0.08, 0.05]))
        age = current_year - year

        # Mileage: roughly 11,000 miles per year of age with random variation
        avg_annual_mileage = np.random.normal(11500, 3000)
        mileage = int(max(2500, age * avg_annual_mileage + np.random.normal(0, 4000)))

        # Fuel type
        if brand == "Tesla":
            fuel_type = "Electric"
        else:
            fuel_type = np.random.choice(["Gasoline", "Hybrid", "Diesel"], p=[0.80, 0.15, 0.05])

        engine = random.choice(engine_options[fuel_type])

        # Transmission
        if brand == "Tesla":
            transmission = "Automatic"
        elif fuel_type == "Hybrid":
            transmission = np.random.choice(["Automatic", "CVT"], p=[0.6, 0.4])
        else:
            transmission = np.random.choice(transmissions, p=[0.72, 0.10, 0.12, 0.06])

        # Accident history
        accident = np.random.choice(
            ["None reported", "At least 1 accident or damage reported"],
            p=[0.78, 0.22]
        )

        clean_title = "Yes" if accident == "None reported" or np.random.rand() > 0.35 else "No"

        # Price calculation based on depreciation and attributes
        base_price = brand_base_prices[brand]
        
        # Premium model bonus
        if any(prem in model for prem in ["M3", "S-Class", "Panamera", "911", "Model S", "F-150", "Grand Cherokee"]):
            base_price *= 1.35
        elif any(budget in model for budget in ["Corolla", "Civic", "Elantra", "Sentra"]):
            base_price *= 0.85

        # Depreciation: ~10% per year, compounded, with mileage penalty
        deprec_factor = (0.91 ** age) * (0.97 ** (mileage / 12000.0))
        
        # Accident penalty
        accident_mult = 0.82 if accident != "None reported" else 1.0
        title_mult = 0.88 if clean_title == "No" else 1.0

        # Fuel type and engine modifiers
        engine_mult = 1.15 if "V8" in engine or "Dual Electric" in engine else (1.08 if "V6" in engine else 1.0)

        # Random market noise
        market_noise = np.random.normal(1.0, 0.06)

        calculated_price = base_price * deprec_factor * accident_mult * title_mult * engine_mult * market_noise
        final_price = int(max(4500, round(calculated_price, -2)))  # Round to nearest hundred

        records.append({
            "brand": brand,
            "model": model,
            "model_year": year,
            "milage": mileage,
            "fuel_type": fuel_type,
            "engine": engine,
            "transmission": transmission,
            "accident": accident,
            "clean_title": clean_title,
            "price": final_price
        })

    df = pd.DataFrame(records)
    return df

def generate_sequential_maintenance_data(num_sequences=40, seq_len=20, random_seed=42):
    """Generates authentic sequential vehicle lifecycle and maintenance telemetry for HMM demonstrations."""
    np.random.seed(random_seed)
    random.seed(random_seed)

    hidden_states = ["Normal Operation", "Minor Component Wear", "Critical System Fault"]
    observations = ["Smooth Ride", "Engine Noise", "Warning Light", "Brake Squeal", "Performance Drop"]

    # Transition probability matrix:
    # From Normal -> Normal (85%), Minor Wear (12%), Critical (3%)
    # From Minor Wear -> Normal (repaired 15%), Minor Wear (70%), Critical (15%)
    # From Critical -> Normal (repaired 40%), Minor Wear (20%), Critical (40%)
    trans_matrix = np.array([
        [0.85, 0.12, 0.03],
        [0.15, 0.70, 0.15],
        [0.40, 0.20, 0.40]
    ])

    # Emission probability matrix:
    # Normal: Smooth Ride (75%), Engine Noise (10%), Warning Light (5%), Brake Squeal (5%), Performance Drop (5%)
    # Minor Wear: Smooth Ride (25%), Engine Noise (30%), Warning Light (15%), Brake Squeal (20%), Performance Drop (10%)
    # Critical: Smooth Ride (5%), Engine Noise (25%), Warning Light (35%), Brake Squeal (10%), Performance Drop (25%)
    emission_matrix = np.array([
        [0.75, 0.10, 0.05, 0.05, 0.05],
        [0.25, 0.30, 0.15, 0.20, 0.10],
        [0.05, 0.25, 0.35, 0.10, 0.25]
    ])

    records = []
    base_time = datetime(2023, 1, 1, 9, 0)

    for vid in range(1, num_sequences + 1):
        vehicle_id = f"VEH-{1000 + vid}"
        current_state_idx = 0  # Start Normal
        current_mileage = random.randint(15000, 45000)

        for step in range(seq_len):
            timestamp = base_time + timedelta(days=step * 14, hours=random.randint(1, 8))
            current_mileage += random.randint(400, 900)

            # Emit observation
            obs_idx = np.random.choice(len(observations), p=emission_matrix[current_state_idx])
            observed = observations[obs_idx]
            hidden_st = hidden_states[current_state_idx]

            action = "Routine Inspection"
            if observed in ["Warning Light", "Performance Drop"]:
                action = "Diagnostic Scan & Service"
            elif observed == "Brake Squeal":
                action = "Brake Service"
            elif observed == "Engine Noise":
                action = "Engine Inspection"

            records.append({
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                "vehicle_id": vehicle_id,
                "step_index": step + 1,
                "mileage": current_mileage,
                "observed_symptom": observed,
                "hidden_state": hidden_st,
                "recommended_action": action
            })

            # Transition to next state
            current_state_idx = np.random.choice(len(hidden_states), p=trans_matrix[current_state_idx])

    df_seq = pd.DataFrame(records)
    return df_seq

if __name__ == "__main__":
    df_cars = generate_used_cars_data(1500)
    df_cars.to_csv("/Users/mohammedsahilts/.gemini/antigravity/scratch/used_car_price_prediction/data/used_cars_sample.csv", index=False)
    print("Created used_cars_sample.csv with shape:", df_cars.shape)
    print("Columns:", list(df_cars.columns))
    print(df_cars.head(3))

    df_seq = generate_sequential_maintenance_data(40, 20)
    df_seq.to_csv("/Users/mohammedsahilts/.gemini/antigravity/scratch/used_car_price_prediction/data/sequential_maintenance_sample.csv", index=False)
    print("Created sequential_maintenance_sample.csv with shape:", df_seq.shape)
    print("Columns:", list(df_seq.columns))
