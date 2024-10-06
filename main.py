from flask import Flask, jsonify, render_template
from importlib import import_module

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route('/handtrack')
def handtrack():
    module = import_module("hand_tracking")
    result = module.Handtrack()
    return render_template("index.html")

@app.route('/Volume')
def volume():
    module = import_module("volume_cont")
    result = module.Vol()
    return render_template("index.html")

@app.route("/Brightness")
def brightness():
    module = import_module("birghtness_cntl")
    result = module.bright()
    return render_template("index.html")

@app.route("/vmouse")
def virtual():
    module = import_module("VirtualMouse")
    result = module.vmouse()
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

