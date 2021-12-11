/**
 * Handles different formats of images.
 * 
 * 2021 Benjamin Kellenberger
 */




/**
 * Band and render configuration
 */
//TODO: load from server?
const DEFAULT_BAND_CONFIG = [
    'Red', 'Green', 'Blue'
]

const DEFAULT_RENDER_CONFIG = {
    'bands': {
        'indices': {
            'red': 0,
            'green': 1,
            'blue': 2
        }
    },
    "contrast": {
        "percentile": {
            "min": 0.0,
            "max": 100.0
        }
    },
    "grayscale": false,
    "white_on_black": false,
    "brightness": 0
}

const _update_render_config = (renderConfig, defaults) => {
    if(typeof(defaults) === 'object') {
        for(var key in defaults) {
            if(!renderConfig.hasOwnProperty(key)) {
                renderConfig[key] = defaults[key];
            } else {
                renderConfig[key] = _update_render_config(renderConfig[key], defaults[key]);
            }
        }
    }
    return renderConfig;
}

const get_band_config = (bandConfig) => {
    if(!Array.isArray(bandConfig) || bandConfig.length === 0) {
        bandConfig = DEFAULT_BAND_CONFIG;
    }
    return bandConfig;
}

const get_render_config = (bandConfig, renderConfig) => {
    bc_out = get_band_config(bandConfig);
    rc_out = _update_render_config(renderConfig, DEFAULT_RENDER_CONFIG);
    for(var l=0; l<rc_out['bands']['indices'].length; l++) {
        rc_out['bands']['indices'][l] = Math.min(rc_out['bands']['indices'][l], bc_out.length-1);
    }
    return rc_out;
}

const get_render_config_val = (renderConfig, tokens, fallback) => {
    if(typeof(tokens) === 'string') {
        tokens = [tokens];
    }
    let val = fallback;
    try {
        val = renderConfig[tokens[0]];
        if(val === undefined) val = fallback;
    } catch {
        val = fallback;
    }
    if(tokens.length <= 1) {
        return val;
    } else {
        return get_render_config_val(val, tokens.slice(1), fallback);
    }
}

/**
 * Performs image enhancements according to the options in "renderConfig":
 * {
 *   "contrast": {
 *     "percentile": {
 *       "min": 0.0,
 *       "max": 100.0
 *     }
 *   },
 *   "grayscale": false,
 *   "white_on_black": false,
 *   "brightness": 0
 * }
 * 
 * Works on interleaved arrays only.
 */
const image_enhancement = (arr, renderConfig) => {
    //TODO: contrast stretch
    let grayscale = get_render_config_val(renderConfig, 'grayscale', false);
    let whiteOnBlack = get_render_config_val(renderConfig, 'white_on_black', false);
    let brightness = get_render_config_val(renderConfig, 'brightness', 0);
    if(!(grayscale || whiteOnBlack || brightness)) return arr;

    //TODO: multithreaded?
    return new Promise((resolve) => {
        for(var p=0; p<arr.length; p+=4) {
            let cVals = arr.slice(p,p+3);
            if(grayscale) {
                let gray = mean(cVals);
                for(var v in cVals) {
                    cVals[v] = gray;
                }
            }
            //TODO: contrast stretch
            [0,1,2].map((idx) => {
                if(whiteOnBlack) {
                    arr[p+idx] = 255 - cVals[idx] + brightness;
                } else {
                    arr[p+idx] = cVals[idx] + brightness;
                }
            });
        }
        return resolve(arr);
    });
}

const calculate_edges = (arr, numBands, width, height, interleave) => {
    /**
     * Calculates a Sobel edge detection on an array. Returns an interleaved
     * array if "interleave" is true, else band sequential.
     */
    let promise = Promise.resolve(arr);
    // reshape array
    if(is_interleaved(arr)) {
        promise = biptobsq(arr, numBands);
    }

    promise = promise.then((arr_bsq) => {
        // convert typed to JS arrays
        arr_bsq = arr_bsq.map((band) => {
            return Array.from(band);
        });
        arr_bsq = math.reshape(arr_bsq, [numBands, width, height]);
        return sobel(arr_bsq);
    })
    .then((arr_edges) => {
        if(interleave) {
            return bsqtobip(arr_edges);
        } else {
            return arr_edges;
        }
    });
    return promise;
}


/**
 * Image Parsers
 */
class WebImageParser {
    /**
     * Handles Web-compliant images such as JPEGs, PNGs, etc.
     */
    constructor(source) {
        this.source = source;
        this.image = null;
        this.imageLoadingPromise = null;

        // determine source type
        this.source = source;
        if(typeof(this.source) === 'string') {
            this.sourceType = 'uri';
        } else if(typeof(this.source) === 'object') {
            if(this.source.hasOwnProperty('files')) {
                this.source = this.source.files[0];
            }
            this.sourceType = 'blob';
        } else {
            this.sourceType = undefined;
        }
    }

    load_image(forceReload) {
        /**
         * Loads the actual image contents, but no prepared array yet.
         */
        if(forceReload || this.imageLoadingPromise === null) {
            let self = this;
            if(this.sourceType === 'uri') {
                self.imageLoadingPromise = new Promise(resolve => {
                    self.image = new Image();
                    self.image.addEventListener('load', () => {
                        return resolve(self.image);
                    });
                    self.image.src = self.source;
                });
            } else if(this.sourceType === 'blob') {
                //TODO: untested
                self.imageLoadingPromise = new Promise(resolve => {
                    let reader  = new FileReader();
                    reader.onload = function(e) {
                        self.image = new Image();
                        self.image.addEventListener('load', () => {
                            return resolve(self.image);
                        });
                        self.image.src = e.target.result;
                    }
                    reader.readAsDataURL(self.source);
                });
            }
        }
        return this.imageLoadingPromise;
    }

    get_raw_image() {
        return this.image;
    }

    get_image_array(bands, bsq) {
        /**
         * Loads the image if necessary and then returns an interleaved array
         * with selected bands.
         */
        let self = this;
        if(self.image === null) {
            return self.load_image().then(() => {
                return self._image_to_array(self.image, bands, bsq);
            });
        } else {
            return self._image_to_array(self.image, bands, bsq);
        }
    }

    _image_to_array(image, bands, bsq) {
        /**
         * Draws the given image to a canvas and extracts image data at given
         * location (bands).
         * TODO: try with black-and-white images
         */
        return new Promise(resolve => {
            let canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            let context = canvas.getContext('2d');
            context.drawImage(image, 0, 0);
            let imageData = context.getImageData(0, 0, image.width, image.height);
            return resolve(imageData);
        }).then((imageData) => {
            if(4*imageData.width*imageData.height !== imageData.data.length) {      // 4 for RGBA
                // need to subset array for bands
                let numBands = imageData.data.length / (imageData.width*imageData.height);
                if(bsq) {
                    return biptobsq(imageData.data, imageData.data.length).then((arr_bsq) => {
                        return band_select(arr_bsq, bands, numBands);
                    });
                } else {
                    return band_select(imageData.data, bands, numBands);
                }
            } else {
                if(bsq) {
                    return biptobsq(imageData.data, imageData.data.length);
                } else {
                    return imageData.data;
                }
            }
        });
    }

    getWidth() {
        try {
            return this.image.width;
        } catch {
            return undefined;
        }
    }

    getHeight() {
        try {
            return this.image.height;
        } catch {
            return undefined;
        }
    }

    getNumBands() {
        // standard canvas images always come in RGBA configuration
        return 4;
    }
}
WebImageParser.prototype.get_supported_formats = function() {
    return {
        'extensions': [
            '.jpg',
            '.jpeg',
            '.png',
            '.gif',
            '.bmp',
            '.ico',
            '.jfif',
            '.pjpeg',
            '.pjp'
        ],
        'mime_types': [
            'image/jpeg',
            'image/bmp',
            'image/x-windows-bmp',
            'image/gif',
            'image/x-icon',
            'image/png'
        ]
    }
}

class TIFFImageParser extends WebImageParser {
    load_image(forceReload) {
        let self = this;
        if(!forceReload && self.imageLoadingPromise !== null) {
            return self.imageLoadingPromise;
        } else {
            self.size = [];
            if(self.sourceType === 'uri') {
                self.imageLoadingPromise = GeoTIFF.fromUrl(self.source);
            } else if(self.sourceType === 'blob') {
                self.imageLoadingPromise = GeoTIFF.fromBlob(self.source);
            }
            self.imageLoadingPromise = self.imageLoadingPromise.then((tiff) => {
                return tiff.getImage();
            }).then((imageSource) => {
                self.size.push(imageSource.getWidth());
                self.size.push(imageSource.getHeight());
                return imageSource.readRasters({interleave:false}).then((img) => {  //TODO: speed up with interleaving?
                    self.image = img;
                });    
            });
            return self.imageLoadingPromise;
        }
    }

    _image_to_array(image, bands, bsq) {
        /**
         * For the TIFF parser we don't need a virtual canvas but can extract
         * bands directly.
         */
        return new Promise(resolve => {
            if(bands.length < image.length) {
                // need to subset array for bands
                return band_select(image, bands).then((arr) => {
                    if(bsq) {
                        return resolve(arr);
                    } else {
                        return resolve(bsqtobip(arr));
                    }
                });
            } else {
                if(bsq) {
                    return resolve(image);
                } else {
                    return resolve(bsqtobip(image));
                }
            }
        });
    }

    getWidth() {
        return this.size[0];
    }

    getHeight() {
        return this.size[1];
    }

    getNumBands() {
        try {
            return this.image.length;
        } catch {
            return undefined;
        }
    }
}
TIFFImageParser.prototype.get_supported_formats = function() {
    return {
        'extensions': [
            '.tif',
            '.tiff',
            '.geotif',
            '.geotiff'
        ],
        'mime_types': [
            'image/tif',
            'image/tiff'
        ]
    }
}


// Inventory of image parsers
window.imageParsers = [
    WebImageParser,
    TIFFImageParser
]
// get renderers by format
window.imageParsersByFormat = {
    'mime': {},
    'extension': {}
}
for(var r in window.imageParsers) {
    let parser = window.imageParsers[r];
    let capabilities = parser.prototype.get_supported_formats();
    for(var e in capabilities['extensions']) {
        let ext = capabilities['extensions'][e];
        window.imageParsersByFormat['extension'][ext] = parser;
    }
    for(var t in capabilities['mime_types']) {
        let type = capabilities['mime_types'][t];
        window.imageParsersByFormat['mime'][type] = parser;
    }
}

const getParserByExtension = (ext) => {
    ext = ext.toLowerCase().trim();
    if(!ext.startsWith('.')) ext = '.' + ext;
    try {
        return window.imageParsersByFormat['extension'][ext];
    } catch {
        return WebImageParser;
    }
}

const getParserByMIMEtype = (type)  => {
    type = type.toLowerCase().trim();
    if(!type.startsWith('image/')) type = 'image/' + type;
    else if(!type.startsWith('image')) type = 'image' + type;
    try {
        return window.imageParsersByFormat['mime'][type];
    } catch {
        return WebImageParser;
    }
}



/**
 * Image Renderer. Responsible for finding the appropriate image parser,
 * depending on the format, as well as for rendering properties like band
 * selection, contrast stretch, etc.
 */
class ImageRenderer {
    constructor(viewport, bandConfig, renderConfig, source) {
        this.viewport = viewport;
        this.canvas = null;
        this.data = null;
        this.renderPromise = null;
        this.bandConfig = get_band_config(bandConfig);
        this.renderConfig = get_render_config(bandConfig, renderConfig);

        // determine source type and required image parser
        this.source = source;
        let parserClass = WebImageParser;
        if(typeof(this.source) === 'string') {
            this.sourceType = 'uri';
            let fileName = this.source.split('/').pop().split('#')[0].split('?')[0];
            fileName = fileName.substring(fileName.lastIndexOf('.'));
            parserClass = getParserByExtension(fileName)

        } else if(typeof(this.source) === 'object') {
            if(this.source.hasOwnProperty('files')) {
                this.source = this.source.files[0];
            }
            this.sourceType = 'blob';
            parserClass = getParserByMIMEtype(this.source.type);
        } else {
            this.sourceType = undefined;
        }
        this.parser = new parserClass(this.source);
    }

    load_image() {
        let self = this;
        return this.parser.load_image()
        .then(() => {
            return self._render_image(false);
        });
    }

    _render_image(force) {
        let self = this;
        if(force || this.renderPromise === null) {
            this.renderPromise = new Promise((resolve) => {
                // band selection
                let bands = [       //TODO: grayscale
                    self.renderConfig['bands']['indices']['red'],
                    self.renderConfig['bands']['indices']['green'],
                    self.renderConfig['bands']['indices']['blue']
                ]
                return resolve(self.parser.get_image_array(bands));
            })
            .then((arr) => {
                // image touch-up: grayscale conversion, white on black, etc.
                return image_enhancement(arr, self.renderConfig);
            })
            //TODO: test to display edge image
            // .then(() => {
            //     return self.get_edge_image();
            // })
            // .then((arr) => {
            //     return bsqtobip([math.reshape(arr, [-1])], 'float32')
            // })
            .then((arr) => {
                self.data = arr;
                let imageData = new ImageData(new Uint8ClampedArray(arr), self.getWidth(), self.getHeight());
                self.canvas = document.createElement('canvas');
                self.canvas.width = imageData.width;
                self.canvas.height = imageData.height;
                self.canvas.getContext('2d').putImageData(imageData, 0, 0);
            });
    
            /**
             * //TODO: implement functionality below (+ brightness)
             * .then((data) => {
                // quantile stretch
                let minVal = get_render_config_val(self.renderConfig, ['contrast', 'percentile', 'min'], 0.0) / 100.0;
                let maxVal = get_render_config_val(self.renderConfig, ['contrast', 'percentile', 'max'], 100.0) / 100.0;
                return quantileStretchImage(data, minVal, maxVal, self.brightness);
            })
            .then((arr_out) => {
                if(self.renderConfig['grayscale']) {
                    return to_grayscale(arr_out);
                } else {
                    return arr_out;
                }
            })
            .then((arr_out) => {
                if(self.renderConfig['white_on_black']) {
                    return white_on_black(arr_out);
                } else {
                    return arr_out;
                }
            })
            .then((arr_out) => {
                return bsqtobip(arr_out);
            })
             */
        }
        return this.renderPromise;
    }

    get_image(as_canvas) {
        if(as_canvas) {
            return this.canvas;
        } else {
            return this.data;
        }
    }

    get_edge_image(fromCanvas) {
        /**
         * Calculates the image's edges with a Sobel filter. If "fromCanvas" is
         * true, the current RGB image as visible on the annotation canvas will
         * be taken as reference. This can result in strong speedups and reduced
         * memory requirements, especially if the original image is of very high
         * resolution and/or has more than three bands. However, it might reduce
         * the quality of the edges. Otherwise the original image in full
         * resolution with all bands is used to calculate the edges.
         */
        if(this.edgeImage === undefined) {
            window.taskMonitor.addTask('edgeImage', 'finding edges');
            let numBands = this.getNumBands();
            let width = this.getWidth();
            let height = this.getHeight();
            let self = this;
            return new Promise((resolve) => {
                let promise = null;
                if(fromCanvas) {
                    // draw image onto resized canvas       //TODO: very hacky; also: dedicated function?
                    numBands = 3;   // RGB (we'll remove the alpha band below)
                    promise = new Promise((resolve) => {
                        let imgC = self.get_image(true);
                        let scaleRatio = Math.max(
                            self.viewport.canvas.width() / imgC.width,
                            self.viewport.canvas.height() / imgC.height
                        )
                        let dummyCanvas = document.createElement('canvas');
                        width = parseInt(scaleRatio * imgC.width);
                        height = parseInt(scaleRatio * imgC.height);
                        dummyCanvas.width = width;
                        dummyCanvas.height = height;
                        let ctx = dummyCanvas.getContext('2d');
                        ctx.drawImage(imgC, 0, 0, dummyCanvas.width, dummyCanvas.height);
                        let imageData = ctx.getImageData(0, 0, width, height);
                        return resolve(imageData.data);
                    })
                    .then((arr) => {
                        return band_select(arr, [0,1,2], 4);        // remove alpha band
                    });
                } else {
                    let allBands = Array.from(Array(numBands).keys());
                    promise = self.parser.get_image_array(allBands, true);
                }
                promise.then((arr) => {
                    return calculate_edges(arr, numBands, width, height, false)
                    .then((edges) => {
                        self.edgeImage = edges;
                        window.taskMonitor.removeTask('edgeImage');
                        return resolve(edges);
                    });
                });
            });
        } else {
            return Promise.resolve(this.edgeImage);
        }
    }

    async rerenderImage() {
        let self = this;
        this._render_image(true).then(() => {
            self.viewport.render();
        });
    }

    getNumBands() {
        return this.parser.getNumBands();
    }

    getWidth() {
        return this.parser.getWidth();
    }

    getHeight() {
        return this.parser.getHeight();
    }

    async updateRenderConfig(renderConfig) {
        /**
         * Receives an updated dict of capabilities and re-renders the image
         * accordingly.
         */
        this.renderConfig = renderConfig;
        this.rerenderImage();
    }
}
ImageRenderer.prototype.get_render_capabilities = function() {
    return {
        "bands": true,
        "grayscale": true,
        "contrast": {
            "percentile": true
        },
        "white_on_black": true,
        "brightness": true  //TODO
    }
}