from flask import Flask, render_template, request
from pymongo import MongoClient
import datetime
import pprint
app = Flask(__name__)
client = MongoClient("mongodb+srv://jeanM:iWjYfAp7COoLJS2Q@cesitest-xb3jc.gcp.mongodb.net/machine_learning?retryWrites=true&w=majority")

db = client['machine_learning']
collection = db['results']

@app.route('/', methods=['GET', 'POST'])
def index():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        imagefile = request.files.get('avatar', '')
        #TRAITEMENT DE MILOUX ANTOINE
        datademiloux = "humain"
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
    print(collectionData)
    return ('ok')




    
