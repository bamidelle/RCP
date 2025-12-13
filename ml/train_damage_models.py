from sklearn.ensemble import GradientBoostingRegressor
import joblib

model = GradientBoostingRegressor()
model.fit(X, y)

joblib.dump(model, "models/water_damage.pkl")
