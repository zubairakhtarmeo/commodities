from enhanced_predictor import EnhancedCryptoPricePredictor

predictor = EnhancedCryptoPricePredictor()

# load your saved model (adjust path if needed)

predictor.load_model_full("models")

current_price = 81000  # or your current BTC price

preds = predictor.predict(current_price)

print("\n=== RAW MODEL OUTPUT ===")

for k, v in preds.items():
    print(k, v)
