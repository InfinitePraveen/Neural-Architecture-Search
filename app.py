import os
import numpy as np
from django import forms
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import HttpResponse
from django.template import engines
from django.urls import path
from PIL import Image
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_nas_model.keras")

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="nas-demo-secret",
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["127.0.0.1", "localhost"],
        TEMPLATES=[{"BACKEND":"django.template.backends.django.DjangoTemplates",
                    "DIRS":[os.path.join(BASE_DIR,"templates")]}],
    )

import django
django.setup()

LABELS = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
          "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

class PredictionForm(forms.Form):
    image = forms.ImageField(label="Upload a Fashion-MNIST style image")

def home(request):
    prediction = None
    error = None
    form = PredictionForm(request.POST or None, request.FILES or None)

    if request.method == "POST" and form.is_valid():
        if not os.path.exists(MODEL_PATH):
            error = "Run the notebook first to create models/best_nas_model.keras."
        else:
            try:
                image = Image.open(form.cleaned_data["image"]).convert("L").resize((28,28))
                array = np.asarray(image, dtype="float32") / 255.0
                model = keras.models.load_model(MODEL_PATH)
                probabilities = model.predict(array.reshape(1,28,28), verbose=0)[0]
                index = int(np.argmax(probabilities))
                prediction = f"{LABELS[index]} ({probabilities[index]*100:.1f}% confidence)"
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    template = engines["django"].get_template("index.html")
    return HttpResponse(template.render(
        {"form":form, "prediction":prediction, "error":error}, request))

urlpatterns = [path("", home, name="home")]

if __name__ == "__main__":
    execute_from_command_line([__file__, "runserver", "127.0.0.1:8000"])
