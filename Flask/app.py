from flask import Flask, render_template, request, jsonify
from .model import predict_image
from .utils import get_disease_info

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    img_bytes = file.read()
    prediction = predict_image(img_bytes)
    result_html = get_disease_info(prediction)

    return jsonify({
        "prediction": prediction,
        "result": result_html
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
