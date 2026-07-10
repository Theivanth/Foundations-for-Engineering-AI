from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train_model(df):
    # Features
    x = df[["Temperature", "Humidity"]]
    y = df["AQI"]
    
    #Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create model
    model = LinearRegression()

    # Train the model
    model.fit(x_train, y_train)
    
    # Predict
    prediction = model.predict(x_test)

    # Evaluate the model
    error = mean_absolute_error(y_test, prediction)
    print(f"Mean Absolute Error: {error}")
    return model

