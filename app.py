import os, sys, time
from bottle import Bottle, static_file

# dirname = os.path.dirname(os.path.abspath(sys.argv[0]))
# Set up bottle server
app = Bottle()
    

@app.route('/')
def index():
    # print('INDEX.HTML')
    return static_file("index.html", root="")

@app.route('/<filename:re:.*\.css>')
def send_css(filename):
    # print('SOME CSS')
    # print(filename)
    return static_file(filename, root='./static/css/')
    # return bottle.static_file(filename, root=dirname+'/static')

@app.route('/<filename:re:openseadragon\.min\.js$>')
def send_osd(filename):
    print('OPENSEADRAGON.MIN.JS')
    print(filename)
    return static_file(filename, root='./static/openseadragon/')
#     # return bottle.static_file(filename, root=dirname+'/openseadragon')

@app.route('/<filename:re:image-picker\.min\.js$>')
def send_imgpkr(filename):
    print('IMAGE-PICKER.MIN.JS')
    print(filename)
    return static_file(filename, root='./static/js/')
#     # return bottle.static_file(filename, root=dirname+'/openseadragon')

@app.route('/<filename:re:image-picker-masonry\.js$>')
def send_imgpkrmasonry(filename):
    print('IMAGE-PICKER-MASONRY.JS')
    print(filename)
    return static_file(filename, root='./static/js/')

@app.route('/<filename:re:.*_IMG_.*\.JPG$>')
def send_img(filename):
    print('IMAGE FILE')
    print(filename)
    return static_file(filename, root='./static/img/')

app.run(host='0.0.0.0', port=8080)
