from utils import load_data, train_model, save_model, load_model
from sklearn.metrics import mean_squared_error
import pandas as pd


def main():
    # Load data
    X, y = load_data("data/housing.csv")

    # Train model
    model = train_model(X, y)

    # Predict
    predictions = model.predict(X) #Technicaly this is overfitting but for a first run it works

    # Evaluate
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

    # Save model
    save_model(model, "model/price_model.pkl")

    #User input from trained_model

    user_input = input("Please enter the sqft of the house you want to predict the price of: ")

    try:
        size = float(user_input)
        loaded_model = load_model("model/price_model.pkl")

        size_data = pd.DataFrame([[size]], columns=["size"])

        price = loaded_model.predict(size_data)[0]
        print(f"Predicted price: ${price:,.2f}")
    except ValueError:
        print("Please enter a valid number.")


if __name__ == "__main__":
    main()