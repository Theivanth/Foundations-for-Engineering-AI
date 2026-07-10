import pandas as pd 
import numpy as np 

def train_model(csv ="machine_data.csv"):
    df = pd.read_csv(csv)
    # Clean column names and string values, then print
    df.columns = df.columns.str.strip()
    # Strip whitespace from object (string) columns to remove trailing/leading spaces in values
    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    print(df)

    # Label Encoding
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df["Status"] = encoder.fit_transform(df["Status"])
    print(df)

    # Define features and target
    x = df[["Temperature", "Vibration", "Pressure", "RPM"]]
    y = df["Status"]

    # Split the dataset 
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    # Train Model
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=3,random_state=42)
    model.fit(x_train,y_train)
    print('Training Complete')

    return model,encoder,x_test,y_test, df