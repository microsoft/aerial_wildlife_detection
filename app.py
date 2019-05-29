import os, sys, time
from bottle import Bottle, static_file

# dirname = os.path.dirname(os.path.abspath(sys.argv[0]))
# Set up bottle server
app = Bottle()
    

@app.route('/')
def index():
    return static_file("index.html", root="")

@app.route('/<filename:re:.*\.css>')
def send_css(filename):
    return static_file(filename, root='./static/')
    # return bottle.static_file(filename, root=dirname+'/static')

@app.route('/<filename:re:^openseadragon\\.min\\.js$>')
def send_osd(filename):
    return static_file(filename, root='./static/openseadragon/')
#     # return bottle.static_file(filename, root=dirname+'/openseadragon')

app.run(host='0.0.0.0', port=8080)
