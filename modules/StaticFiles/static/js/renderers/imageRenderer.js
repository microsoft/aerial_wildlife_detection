/**
 * Handles different formats of images.
 * 
 * 2021 Benjamin Kellenberger
 */



/**
 * TODO: math utils from here:
 * https://stackoverflow.com/questions/48719873/how-to-get-median-and-quartiles-percentiles-of-an-array-in-javascript-or-php
 */
// sort array ascending
const asc = arr => arr.sort((a, b) => a - b);
const sum = arr => arr.reduce((a, b) => a + b, 0);
const mean = arr => sum(arr) / arr.length;

const roundNumber = (value, multiplier) => {
    return Math.round(value * multiplier) / multiplier;
}

// sample standard deviation
const std = (arr) => {
    const mu = mean(arr);
    const diffArr = arr.map(a => (a - mu) ** 2);
    return Math.sqrt(sum(diffArr) / (arr.length - 1));
};

const quantile = (arr_in, q) => {
    const arr = arr_in.slice();
    const sorted = asc(arr);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
        return sorted[base];
    }
};

const linspace = (a,b,n) => {
    if(typeof n === "undefined") n = Math.max(Math.round(b-a)+1,1);
    if(n<2) { return n===1?[a]:[]; }
    var i,ret = Array(n);
    n--;
    for(i=n;i>=0;i--) { ret[i] = (i*b+(n-i)*a)/n; }
    return ret;
}

const quantiles = (arr_in, quants) => {
    const arr = arr_in.slice();
    const sorted = asc(arr);
    const vals_out = [];
    for(var q in quants) {
        const pos = (sorted.length - 1) * quants[q];
        const base = Math.floor(pos);
        const rest = pos - base;
        if (sorted[base + 1] !== undefined) {
            vals_out.push(sorted[base] + rest * (sorted[base + 1] - sorted[base]));
        } else {
            vals_out.push(sorted[base]);
        }
    }
    return vals_out;
}

const normalizeImage = (arr) => {
    /**
     *  Performs 0-1 rescaling of image.
     */
    let promises = arr.map(band => {
        return new Promise((resolve) => {
            let quants = quantiles(band, [0.0,1.0]);
            let quantDiff = quants[1] - quants[0];
            let band_norm = new Float32Array(band).map(function(v) { 
                return (v - quants[0]) / quantDiff;
            });
            return resolve(band_norm);
        });
    });
    return Promise.all(promises);
}

const quantileStretchImage = (arr, low, high, brightness) => {
    let promises = arr.map(band => {
        return new Promise((resolve) => {
            // calc. quantiles
            let quantvals = quantiles(band, [low, high]);
            let quantdiff = parseFloat(quantvals[1] - quantvals[0]);
            let band_stretch = band.map(pixel => {
                return 255 * (pixel - quantvals[0]) / quantdiff + brightness;
            });
            return resolve(band_stretch);
        });
    });
    return Promise.all(promises);
}

// source: https://stackoverflow.com/questions/66037026/interleave-mutliple-arrays-in-javascript
const biptobsq = arr => Array.from({
    length: Math.max(...arr.map(o => o.length)), // find the maximum length
  },
  (_, i) => arr.map(r => r[i] ?? null) // create a new row from all items in same column or substitute with null
).flat() // flatten the results

const bsqtobip = (arr) => {
    //TODO: multithreaded?
    return new Promise((resolve) => {
        let numVals = 4*arr[0].length;      // 4 for RGBA
        let preparedArray = new Uint8ClampedArray(new Array(numVals));
        [0, 1, 2, -1].map(bIdx => {
            if(bIdx>=0) {
                let band = arr[bIdx];
                for(var s=0; s<numVals; s++) {
                    preparedArray[(s*4)+bIdx] = band[s];
                }
            } else {
                // alpha
                for(var s=0; s<numVals; s++) {
                    preparedArray[(s*4)+3] = 255;
                }
            }
        });
        return resolve(preparedArray);
    });
}

/**
 * Performs band selection on an array. Can handle both interleaved and band
 * sequential arrays. For interleaved arrays "arr_num_bands" (i.e., number of
 * bands in the original input "arr") must be specified.
 */
const band_select = (arr, bands, arr_num_bands) => {
    let arr_out = [];
    let promises = [];
    if(arr[0].length) {
        // array of arrays: band sequential
        promises = bands.map((band) => {
            arr_out.push(arr[band]);
        });
    } else {
        // interleaved
        let numPix = arr.length / arr_num_bands;
        arr_out = new Array(bands.length*numPix);
        let bandIndices = linspace(0,bands.length)
        promises = bandIndices.map((bIdx) => {
            for(var p=bands[bIdx]; p<arr.length; p+=arr_num_bands) {
                arr_out[(p*arr_num_bands)+bIdx] = arr[p]; 
            }
        });
    }
    return Promise.all(promises).then(() => {
        return arr_out;
    })
}

const to_grayscale = (arr) => {
    /**
     * Works on both interleaved and band sequential arrays with four bands
     * (RGBA). Ignores the alpha band.
     */
    //TODO: multithreaded?
    if(arr[0].length) {
        // array of arrays: band sequential
        return new Promise((resolve) => {
            for(var p=0; p<arr[0].length; p++) {
                let g = (arr[0][p] + arr[1][p] + arr[2][p]) / 3.0;
                arr[0][p] = g;
                arr[1][p] = g;
                arr[2][p] = g;
            }
            return resolve(arr);
        });
    } else {
        // interleaved
        return new Promise((resolve) => {
            for(var p=0; p<arr.length; p+=4) {
                let g = (arr[p] + arr[p+1] + arr[p+2]) / 3.0;
                arr[p] = g;
                arr[p+1] = g;
                arr[p+2] = g;
            }
            return resolve(arr);
        });
    }
}

const white_on_black = (arr) => {
    /**
     * Works on BSQ arrays; expects 255 as maximum value
     */
    return new Promise((resolve) => {
        arr.map((band) => {
            for(var b=0; b<band.length; b++) {
                band[b] = 255 - band[b];
            }
        });
        return resolve(arr);
    });
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


/**
 * Render configuration
 */
//TODO: load from server?
const DEFAULT_RENDER_CONFIG = {
    'bands': {
        'labels': [
            "Red", "Green", "Blue"
        ],
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

const get_render_config = (renderConfig) => {
    rc_out = _update_render_config(renderConfig, DEFAULT_RENDER_CONFIG);
    for(var l=0; l<rc_out['bands']['indices'].length; l++) {
        rc_out['bands']['indices'][l] = Math.min(rc_out['bands']['indices'][l], rc_out['bands']['labels'].length-1);
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
 * Image Parsers
 */
class WebImageParser {
    /**
     * Handles Web-compliant images such as JPEGs, PNGs, etc.
     */
    constructor(source) {
        this.source = source;

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
        let self = this;
        if(!forceReload && self.imageLoadingPromise !== undefined) {
            return self.imageLoadingPromise;
        } else {
            if(this.sourceType === 'uri') {
                self.imageLoadingPromise = new Promise(resolve => {
                    self.image = new Image();
                    self.image.addEventListener('load', () => {
                        resolve(self.image);
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
            return self.imageLoadingPromise;
        }
    }

    get_image_array(bands) {
        /**
         * Loads the image if necessary and then returns an interleaved array
         * with selected bands.
         */
        let self = this;
        if(self.image === undefined) {
            return self.load_image().then(() => {
                return self._image_to_array(self.image, bands);
            });
        } else {
            return self._image_to_array(self.image, bands);
        }
    }

    _image_to_array(image, bands) {
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
                return band_select(imageData.data, bands, numBands);
            } else {
                return imageData.data;
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
        if(!forceReload && self.imageLoadingPromise !== undefined) {
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

    _image_to_array(image, bands) {
        /**
         * For the TIFF parser we don't need a virtual canvas but can extract
         * bands directly.
         */
        return new Promise(resolve => {
            if(bands.length < image.length) {
                // need to subset array for bands
                return band_select(image, bands).then((arr) => {
                    return resolve(bsqtobip(arr));
                });   // band sequential
            } else {
                return resolve(bsqtobip(image));
            }
        });
    }

    getWidth() {
        return this.size[0];
    }

    getHeight() {
        return this.size[1];
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
    constructor(viewport, renderConfig, source) {
        this.viewport = viewport;
        this.renderConfig = get_render_config(renderConfig);

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
        return this.parser.load_image();
    }

    render_image(force) {
        let self = this;
        if(force || self.canvas === undefined) {
            return self.load_image().then((img) => {
                // band selection
                let bands = [       //TODO: grayscale
                    self.renderConfig['bands']['indices']['red'],
                    self.renderConfig['bands']['indices']['green'],
                    self.renderConfig['bands']['indices']['blue']
                ]
                return self.parser.get_image_array(bands);
            })
            .then((arr) => {
                // image touch-up: grayscale conversion, white on black, etc.
                return image_enhancement(arr, self.renderConfig);
            })
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
    }

    get_image(as_canvas) {
        let self = this;
        if(this.canvas === undefined) {
            return this.render_image().then(() => {
                if(as_canvas) {
                    return self.canvas;
                } else {
                    return self.data;
                }
            })
        } else {
            if(as_canvas) {
                return Promise.resolve(self.canvas);
            } else {
                return Promise.resolve(self.data);
            }
        }
    }

    async rerenderImage() {
        let self = this;
        this.render_image(true).then(() => {
            self.viewport.render();
        });
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




// //TODO: OLD


// /**
//  * IMAGE RENDERERS
//  */
// class AbstractImageRenderer {
//     constructor(viewport, renderConfig, source) {
//         this.viewport = viewport;
//         this.renderConfig = get_render_config(renderConfig);
        
//         // determine source type
//         this.source = source;
//         if(typeof(this.source) === 'string') {
//             this.sourceType = 'uri';
//         } else if(typeof(this.source) === 'object') {
//             if(this.source.hasOwnProperty('files')) {
//                 this.source = this.source.files[0];
//             }
//             this.sourceType = 'blob';
//         } else {
//             this.sourceType = undefined;
//         }
//     }

//     load_image() {
//         throw Error('Not implemented for abstract base class.')
//     }

//     get_image() {
//         throw Error('Not implemented for abstract base class.')
//     }

//     getWidth() {
//         throw Error('Not implemented for abstract base class.')
//     }

//     getHeight() {
//         throw Error('Not implemented for abstract base class.')
//     }

//     getMetadata() {
//         /**
//          * Returns a dict of details like band names, etc.
//          */
//         throw Error('Not implemented for abstract base class.')
//     }

//     renderImage() {
//         throw Error('Not implemented for abstract base class.')
//     }

//     async rerenderImage() {
//         let self = this;
//         this.renderImage().then(() => {
//             self.viewport.render();
//         });
//     }

//     async updateRenderConfig(renderConfig) {
//         /**
//          * Receives an updated dict of capabilities and re-renders the image
//          * accordingly.
//          */
//         this.renderConfig = renderConfig;
//         this.rerenderImage();
//     }
// }
// AbstractImageRenderer.prototype.get_render_capabilities = function() {
//     /**
//      * Returns a dict with the following possible values:
//      * {
//      *      "extensions": [".jpg", ".jpeg", ...],       // supported file extensions
//      *      "mime_types": ["image/jpeg", ...],          // supported MIME types
//      *      "bands": true,       // whether or not bands can be reorganised
//      *      "contrast": {
//      *          "percentile": true  // whether or not percentiles can be used for contrast stretch
//      *      }
//      * }
//      */
//     return {
//         "extensions": [],
//         "mime_types": [],
//         "bands": false,
//         "grayscale": false,
//         "contrast": {
//             "percentile": false
//         }
//     }
// }



// class WebImageRenderer extends AbstractImageRenderer {
//     /**
//      * Default Image Renderer for Web-compliant images.
//      * Does not allow any customization in viewing (contrast stretch, etc.).
//      */
//     constructor(viewport, renderConfig, source) {
//         super(viewport, renderConfig, source);
//         this.image = undefined;
//     }

//     load_image() {
//         let self = this;
//         if(this.sourceType === 'uri') {
//             return new Promise(resolve => {
//                 self.image = new Image();
//                 self.image.addEventListener('load', () => {
//                     resolve(self.image);
//                 });
//                 self.image.src = self.source;
//             });
//         } else if(this.sourceType === 'blob') {
//             //TODO: untested
//             let reader  = new FileReader();
//             reader.onload = function(e) {
//                 self.image = new Image();
//                 self.image.addEventListener('load', () => {
//                     resolve(self.image);
//                 });
//                 self.image.src = e.target.result;
//             }
//             reader.readAsDataURL(self.source);
//         }
//     }

//     _get_image_array(image) {
//         return new Promise(resolve => {
//             let canvas = document.createElement('canvas');
//             canvas.width = image.width;
//             canvas.height = image.height;
//             let context = canvas.getContext('2d');
//             context.drawImage(image, 0, 0);
//             return resolve(context.getImageData(0, 0, image.width, image.height).data);
//         });
//     }

//     get_image() {
//         let self = this;
//         if(this.image === undefined) {
//             return this.load_image().then((resolve) => {
//                 if(as_img) return resolve(self.image);
//                 else return self._get_image_array(self.image);
//             })
//         } else {
//             if(as_img) return resolve(self.image);
//             else return self._get_image_array(self.image);
//         }
//     }

//     renderImage() {
//         return this.get_image();
//     }

//     getWidth() {
//         return this.image.width;
//     }

//     getHeight() {
//         return this.image.height;
//     }

//     getMetadata() {
//         return {
//             'band_names': [
//                 'Red', 'Green', 'Blue'      //TODO: account for grayscale images
//             ]
//         }
//     }

//     get_render_capabilities() {
//         let capabilities = super.get_render_capabilities();
//         capabilities['extensions'] = [
//             '.jpg',
//             '.jpeg',
//             '.png',
//             '.gif',
//             '.bmp',
//             '.ico',
//             '.jfif',
//             '.pjpeg',
//             '.pjp'
//         ];
//         capabilities['mime_types'] = [
//             'image/jpeg',
//             'image/bmp',
//             'image/x-windows-bmp',
//             'image/gif',
//             'image/x-icon',
//             'image/png'
//         ];
//         return capabilities;
//     }
// }


// class TIFFrenderer extends AbstractImageRenderer {

//     constructor(viewport, renderConfig, source) {
//         super(viewport, renderConfig, source);
//         /**
//          * TODO: fetch properties:
//          * - band configuration
//          * - percentile cutoffs
//          */
//         this.img_norm = undefined;
//         this.fakeCanvas = undefined;
//         this.renderingPromise = undefined;

//         //TODO
//         this.percStretch = 0.02;
//         this.brightness = 0.0;
//     }

//     get_render_capabilities() {
//         return {
//             "extensions": [
//                 ".tif",
//                 ".tiff"
//             ],
//             "mime_types": [
//                 "image/tif",
//                 "image/tiff"
//             ],
//             "bands": true,
//             "grayscale": true,
//             "white_on_black": true,
//             "contrast": {
//                 "percentile": true
//             }
//         }
//     }

//     load_image() {
//         this.size = [];
//         let self = this;
//         let promise = undefined;
//         if(this.sourceType === 'uri') {
//             promise = GeoTIFF.fromUrl(this.source);
//         } else if(this.sourceType === 'blob') {
//             promise = GeoTIFF.fromBlob(this.source);
//         }
//         promise = promise.then((tiff) => {
//             return tiff.getImage();
//         }).then((image) => {
//             self.image = image;
//             self.size.push(image.getWidth());
//             self.size.push(image.getHeight());
//             return image.readRasters({interleave:false});    //TODO: speed up with interleaving?
//         }).then((rasters) => {
//             // get metadata (band names)
//             let meta = self.image.getGDALMetadata();
//             self.bandNames = [];
//             if(meta === null || meta === undefined) {
//                 //TODO: extract band names somehow
//                 for(var b=0; b<rasters.length; b++) {
//                     self.bandNames.push('Band ' + (b+1));
//                 }
//             } else {
//                 // extract from GDAL metadata       //TODO: not sure if always works
//                 for(var b=0; b<rasters.length; b++) {
//                     try {
//                         self.bandNames.push(meta['Band_' + (b+1).toString()]);
//                     } catch {
//                         self.bandNames.push('Band ' + (b+1).toString());
//                     }
//                 }
//             }
//             return normalizeImage(rasters);
//         }).then((img_norm) => {
//             self.img_norm = img_norm;
//             return Promise.resolve(self.img_norm);
//         });
//         return promise;
//     }

//     renderImage() {
//         /**
//          * To be called whenever rendering properties (band order, percentile
//          * stretch, etc.) change.
//          */

//         if(this.renderingPromise !== undefined) {
//             return this.renderingPromise;
//         }

//         if(this.img_norm === undefined) {
//             this.renderingPromise = this.load_image();
//         } else {
//             this.renderingPromise = Promise.resolve(this.img_norm);
//         }

//         let self = this;
//         this.renderingPromise = this.renderingPromise.then((data) => {
//             // band selection
//             let data_bandSel = [
//                 data[Math.min(data.length-1, self.renderConfig['bands']['indices']['red'])],
//                 data[Math.min(data.length-1, self.renderConfig['bands']['indices']['green'])],
//                 data[Math.min(data.length-1, self.renderConfig['bands']['indices']['blue'])]
//             ];
//             return data_bandSel;
//         })
//         .then((data) => {
//             // quantile stretch
//             let minVal = get_render_config_val(self.renderConfig, ['contrast', 'percentile', 'min'], 0.0) / 100.0;
//             let maxVal = get_render_config_val(self.renderConfig, ['contrast', 'percentile', 'max'], 100.0) / 100.0;
//             return quantileStretchImage(data, minVal, maxVal, self.brightness);
//         })
//         .then((arr_out) => {
//             if(self.renderConfig['grayscale']) {
//                 return to_grayscale(arr_out);
//             } else {
//                 return arr_out;
//             }
//         })
//         .then((arr_out) => {
//             if(self.renderConfig['white_on_black']) {
//                 return white_on_black(arr_out);
//             } else {
//                 return arr_out;
//             }
//         })
//         .then((arr_out) => {
//             return bsqtobip(arr_out);
//         })
//         .then((arr_bip) => {
//             let imageData = new ImageData(new Uint8ClampedArray(arr_bip), self.size[0], self.size[1]);
//             if(self.fakeCanvas === undefined)
//                 self.fakeCanvas = document.createElement('canvas');
//             self.fakeCanvas.width = self.size[0];
//             self.fakeCanvas.height = self.size[1];
//             self.fakeCanvas.getContext('2d').putImageData(imageData, 0, 0);
//             self.renderingPromise = undefined;
//             return self.fakeCanvas;
//         });
//         return this.renderingPromise;
//     }

//     get_image() {
//         let promise = Promise.resolve(this.fakeCanvas);
//         if(this.fakeCanvas === undefined) {
//             promise = this.renderImage();
//         }
//         return promise;
//     }

//     getWidth() {
//         return this.size[0];
//     }

//     getHeight() {
//         return this.size[1];
//     }

//     getMetadata() {
//         return {
//             'band_names': this.bandNames //TODO: format
//         }
//     }
// }


// // inventory of renderers
// window.renderers = [
//     WebImageRenderer,
//     TIFFrenderer
// ]
// // get renderers by format
// window.renderersByFormat = {
//     'mime': {},
//     'extension': {}
// }
// for(var r in window.renderers) {
//     let renderer = window.renderers[r];
//     let capabilities = renderer.prototype.get_render_capabilities();
//     for(var e in capabilities['extensions']) {
//         let ext = capabilities['extensions'][e];
//         window.renderersByFormat['extension'][ext] = renderer;
//     }
//     for(var t in capabilities['mime_types']) {
//         let type = capabilities['mime_types'][t];
//         window.renderersByFormat['mime'][type] = renderer;
//     }
// }

// const getRendererByExtension = (ext) => {
//     ext = ext.toLowerCase().trim();
//     if(!ext.startsWith('.')) ext = '.' + ext;
//     try {
//         return window.renderersByFormat['extension'][ext];
//     } catch {
//         return WebImageRenderer;
//     }
// }

// const getRendererByMIMEtype = (type)  => {
//     type = type.toLowerCase().trim();
//     if(!type.startsWith('image/')) type = 'image/' + type;
//     else if(!type.startsWith('image')) type = 'image' + type;
//     try {
//         return window.renderersByFormat['mime'][type];
//     } catch {
//         return WebImageRenderer;
//     }
// }