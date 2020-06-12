from flask import Flask, render_template, request
from pymongo import MongoClient
import datetime
import pprint
import json
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile 
import tensorflow as tf 
import zipfile
import pathlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display 
from itertools import groupby 

 
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 
app = Flask(__name__)
client = MongoClient("mongodb+srv://jeanM:iWjYfAp7COoLJS2Q@cesitest-xb3jc.gcp.mongodb.net/machine_learning?retryWrites=true&w=majority")

db = client['machine_learning']
collection = db['results']
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def load_model(model_name):
    base_url = "http://download.tensorflow.org/models/object_detection/"
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, origin= base_url+model_file, untar= True)
    model_dir = pathlib.Path(model_dir)/"saved_model"

    model=tf.saved_model.load(str(model_dir))
    model= model.signatures['serving_default']
    return model 

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#pour tester avec nos images 
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))



model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # Convertir en tensor. (tensor = lot de données)
  input_tensor = tf.convert_to_tensor(image)
  #le model attend un lot d'image (et la doc parle d'un axis a ajouté mais là je suis paumé)
  input_tensor = input_tensor[tf.newaxis,...]

  # Executer le modèle
  output_dict = model(input_tensor)

  # Les outputs doivent être des tensors et on prend l'index 0 
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                  for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # doit être des int
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  
  if 'detection_masks' in output_dict:
    #recardrer l'image par rapport au mask
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def identify_object(model, image_path):
  # ici on prepare le tableau qui resultera de l'image pour y placer les tags (si j'ai bien compris)
  image_np = np.array(Image.open(image_path))
  # detection.
  output_dict = run_inference_for_single_image(model, image_np)
  
  print(list(dict.fromkeys(output_dict['detection_classes'])))
  #print(output_dict['detection_classes'])
  
  # Resultats de la detection
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)
  #display(Image.fromarray(image_np))
  return list(dict.fromkeys(output_dict['detection_classes']))

@app.route('/', methods=['GET', 'POST'])
def index():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        imagefile = request.files.get('avatar', '')
        #TRAITEMENT DE MILOUX ANTOINE
        datademiloux = "grandclochard"
        # # # # # # # # # # # # # # # #
        #ENVOI DES DATA A MANGODB
        currentDateTime = datetime.datetime.now()
        result = {
            "result": datademiloux,
            "date": currentDateTime.strftime("%A")
        }
        collection.insert_one(result)
        return (datademiloux)
        
    except Exception as err:
        print(err)
        return ("Image non uploadé")

@app.route('/results',methods=['POST','GET'])
def result():
    collectionData = collection.find() #retourne un array de result
    for data in collectionData:
        print(data)
    return ("lol")




    
