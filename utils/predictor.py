import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/breed_classifier_final.keras"
CLASS_NAMES_PATH = "model/class_names.json"
IMG_SIZE = (224, 224)

_MODEL_INSTANCE = None


def _clean_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _clean_config(v)
    elif isinstance(obj, list):
        for item in obj:
            _clean_config(item)


def load_classification_model(model_path: str = MODEL_PATH):
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    try:
        model = tf.keras.models.load_model(model_path)
    except (TypeError, ValueError):
        import io
        import zipfile
        import keras

        # If loading fails due to Keras config version incompatibility, sanitize config
        with zipfile.ZipFile(model_path, "r") as zin:
            cfg = json.loads(zin.read("config.json").decode("utf-8"))
            _clean_config(cfg)
            meta = json.loads(zin.read("metadata.json").decode("utf-8"))
            meta["keras_version"] = keras.__version__

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if item.filename == "config.json":
                        zout.writestr("config.json", json.dumps(cfg))
                    elif item.filename == "metadata.json":
                        zout.writestr("metadata.json", json.dumps(meta))
                    else:
                        zout.writestr(item.filename, zin.read(item.filename))
            buffer.seek(0)
            model = tf.keras.models.load_model(buffer)

    _MODEL_INSTANCE = model
    return _MODEL_INSTANCE


# Load class names
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


def predict_breed_details(image: Image.Image, top_k: int = 5):
    """
    Predict top-k dog breeds with confidence scores from an image.
    Note: The model has an internal Rescaling layer (scale=1/127.5, offset=-1),
    so raw [0, 255] float32 values are passed directly.
    """
    model = load_classification_model()

    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize(IMG_SIZE)

    image_array = np.array(image_resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    # Sort indices by descending confidence
    sorted_indices = np.argsort(predictions)[::-1]

    top_predictions = []
    for idx in sorted_indices:
        breed = class_names[idx]
        conf = float(predictions[idx]) * 100.0
        top_predictions.append({
            "breed": breed,
            "display_name": breed.replace("_", " "),
            "confidence": conf
        })

    top_breed = top_predictions[0]["breed"]
    top_confidence = top_predictions[0]["confidence"]

    return top_breed, top_confidence, top_predictions[:top_k], top_predictions


def predict_breed(image: Image.Image):
    """
    Predict the breed of a dog from an uploaded image (backwards-compatible helper).
    """
    top_breed, confidence, _, _ = predict_breed_details(image, top_k=1)
    return top_breed, confidence