from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def predict_stock_prices(data):
    X = data.drop(['Close'], axis=1)
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predicted_prices = model.predict(X_test)
    return predicted_prices

def evaluate_model(y_true
