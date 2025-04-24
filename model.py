import pickle
from sklearn.ensemble import RandomForestClassifier 
from src.models.load_model import load_model

# Example model training
model = RandomForestClassifier()
# Fit your model on data (example)
# model.fit(X_train, y_train)

# Save the model
with open('src/models/model.pkl', 'wb') as file:
    pickle.dump(model, file)
