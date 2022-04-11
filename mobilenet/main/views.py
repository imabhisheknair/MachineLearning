from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import numpy as np
from classify.settings import BASE_DIR
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.mobilenet import MobileNet, preprocess_input
import os

from main.models import Document

# Create your views here.
def home(request):
    return render(request, 'index.html')

def find(request):
    files = request.FILES.get('img')
    img = files
    print(img)
    mobile = MobileNet()
    newdoc = Document(docfile = img)
    newdoc.save()
    preprocessed_img = prepare_img(img)
    predictions = mobile.predict(preprocessed_img)
    results = imagenet_utils.decode_predictions(predictions)
    prediction = str(results[0][0][1])
    if prediction.count('_') > 0:
        prediction = prediction.replace('_', ' ')
    confidence = str(results[0][0][2])
    confidence = confidence[2:4]+'%'
    # return HttpResponse('<br><br><br><center><h1> Its '+ str(results[0][0][1]) +'!!!<br><br><br> Confidence: '+ str(results[0][0][2]) +' </h1></center>')
    return JsonResponse({'message': 'success', 'pred': prediction, 'conf': confidence})


def prepare_img(file):
  img_path = os.path.join(BASE_DIR, 'media/documents/')
  img = image.load_img(img_path + str(file), target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array_expanded_dims = np.expand_dims(img_array, axis=0)
  os.remove(img_path + str(file))
  return preprocess_input(img_array_expanded_dims,)        