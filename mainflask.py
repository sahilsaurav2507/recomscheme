from flask import Flask, request, jsonify
import numpy as np
import pymongo
from tensorflow.keras.models import load_model

app = Flask(__name__)

## MongoDB Config....
client = pymongo.MongoClient("mongodb+srv://thakurharsh345:pBS49MPMBhjZY1Pb@cluster0.wgvqw.mongodb.net/SIH2024USER")
db = client['SIH2024USER'] 
location_collection = db['location']  
scheme_collection = db['scheme']

# Load Trained Model.....
model = load_model('schemerecom_model.keras')


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = data.get('user_id')
        location = data.get('location')

        # Fetch location data
        loc_data = location_collection.find_one({'District': location})
        if not loc_data:
            return jsonify({"error": "Location data not found"}), 404

        scheme_data = list(scheme_collection.find())
        if not scheme_data:
            return jsonify({"error": "Scheme data not found"}), 404

        location_features = np.array(list(loc_data.values())[1:], dtype=np.float32) 
        scheme_features = np.array([list(s.values())[1:] for s in scheme_data], dtype=np.float32)  
        mf_user_input = np.random.rand(5)  
        mf_item_input = np.random.rand(len(scheme_features), 5) 
        context_features = np.concatenate([location_features, scheme_features], axis=1)

        # Predict the recon=
        predictions = model.predict([
            np.tile(location_features, (len(scheme_features), 1)),
            scheme_features,
            np.tile(mf_user_input, (len(scheme_features), 1)),
            mf_item_input,
            np.tile(context_features, (len(scheme_features), 1))
        ])

        # Format recom
        scheme_names = [s['scheme_name'] for s in scheme_data]
        recommendations = sorted(zip(scheme_names, predictions.flatten()), key=lambda x: x[1], reverse=True)
        recommendations_json = [
            {"scheme": scheme, "predicted_rating": float(pred)} for scheme, pred in recommendations[:5]
        ]

        return jsonify({
            "user_id": user_id,
            "location": location,
            "recommendations": recommendations_json
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

