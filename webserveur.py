from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        imagefile = request.files.get('avatar', '')
        return("correctement upload")
    except Exception as err:
        print(err)
        return ("Image non uploadé")    