import sys
import pandas as pd
import pickle


DATA_PATH = "../data/autojarov_cars.csv"
MODEL_OUTPUT_PATH = "../model/model.pkl"

def select_option(options, title):
    print(f"\nSelect {title}:")
    for i, option in enumerate(options):
        print(f"{i + 1}. {option}")

    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")


def run_cli_prediction():
    print("\n=== Car Price Prediction CLI ===")

    # Load dataset (to get dropdown values)
    df = pd.read_csv(DATA_PATH)

    brands = sorted(df["brand"].dropna().unique())
    fuels = sorted(df["fuel"].dropna().unique())
    transmissions = sorted(df["transmission"].dropna().unique())
    # Load trained model
    with open(MODEL_OUTPUT_PATH, "rb") as f:
        model = pickle.load(f)

    # Selection menus
    brand = select_option(brands, "Brand")
    fuel = select_option(fuels, "Fuel Type")
    transmission = select_option(transmissions, "Transmission")

    # Numeric inputs
    km = float(input("\nEnter mileage (km): "))
    year = int(input("Enter year: "))

    # Create dataframe for prediction
    input_data = pd.DataFrame([{
        "brand": brand,
        "fuel": fuel,
        "transmission": transmission,
        "km": km,
        "year": year
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    print("\n=== Predicted Price ===")
    print(f"{prediction:,.0f} Kč")


if __name__ == "__main__":
    run_cli_prediction()