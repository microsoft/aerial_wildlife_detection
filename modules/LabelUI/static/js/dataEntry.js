/*
    Definition of a data entry, as shown on a grid on the screen.

    2019 Benjamin Kellenberger
 */

 class AbstractDataEntry {
     /*
        Abstract base class for data entries.
     */
    constructor(entryID, properties) {
        this.entryID = entryID;
        this.fileName = properties['fileName'];

        this._createImage();
        this._parseLabels(properties);
    }

    _parseLabels(properties) {
        /*
            Iterates through properties object's entries "predictions" and "annotations"
            and creates new primitive instances each.
        */
        this.predictions = {};
        this.annotations = {};

        if(properties.hasOwnProperty('predictions')) {
            for(var key in properties['predictions']) {
                //TODO: make failsafe?
                this.predictions[key] = window.parseAnnotation(key, properties['predictions'][key], 'prediction');
            }
        }
        if(properties.hasOwnProperty('annotations')) {
            for(var key in properties['annotations']) {
                //TODO: make failsafe?
                this.annotations[key] = window.parseAnnotation(key, properties['annotations'][key], 'annotation');
            }
        }
    }

    _createImage() {
        this.image = new Image();
        this.image.width = window.defaultImage_w;
        this.image.height = window.defaultImage_h;
    }

    _setup_markup() {
        this.markup = $('<div class="entry"></div>');

        // define canvas
        this.canvas = $('<canvas id="'+this.canvasID+'" width="'+window.defaultImage_w+'" height="'+window.defaultImage_h+'"></canvas>');
        this.canvas.css('cursor', 'crosshair');
        this.markup.append(this.canvas);
        this.ctx = this.canvas[0].getContext('2d');
    }

    loadImage() {
        this.image.src = this.getImageURI();
    }

    getImageURI() {
        return window.dataServerURI + this.fileName + '?' + Date.now();
    }

    getProperties(minimal) {
        var props = { 'id': this.entryID };
        props['annotations'] = {};
        for(var key in this.annotations) {
            props['annotations'][key] = this.annotations[key].getProperties(minimal);
        }
        if(!minimal) {
            props['fileName'] = this.fileName;
            props['predictions'] = {};
            for(var key in this.predictions) {
                props['predictions'][key] = this.predictions[key].getProperties(minimal);
            }
        }
        return props;
    }


    /* canvas and drawing functionalities */
    _getCanvasCoordinates(event) {
        var posX = (event.pageX - this.canvas.offset().left);
        var posY = (event.pageY - this.canvas.offset().top);
        return [posX, posY];
    }

    _getScaledCanvasCoordinates(event) {
        var coords = this._getCanvasCoordinates(event);
        return this._convertCoordinates(coords, 'scaled');
    }

    _getCanvasScaleFactors() {
        var scaleX = window.defaultImage_w / this.canvas.width();
        var scaleY = window.defaultImage_h / this.canvas.height();
        return [scaleX, scaleY];
    }

    _convertCoordinates(coords, outputFormat) {
        var scales = this._getCanvasScaleFactors();
        if(outputFormat == 'scaled') {
            return [scales[0]*coords[0], scales[1]*coords[1]];
        } else {
            return [coords[0]/scales[0], coords[1]/scales[1]];
        }
    }

    _convertToCanvasScale(coords) {
        return this._convertCoordinates(coords, 'scaled');
    }

    _drawHoverText() {
        //TODO: switch text position to left if too close to right side (top if too close to bottom)
        if(this.hoverPos != null) {
            var dimensions = this.ctx.measureText(this.hoverText);
            dimensions.height = window.styles.hoverText.box.height;
            var offsetH = window.styles.hoverText.offsetH;
            this.ctx.fillStyle = window.styles.hoverText.box.fill;
            this.ctx.fillRect(offsetH+this.hoverPos[0]-2, this.hoverPos[1]-(dimensions.height/2+2), dimensions.width+4, dimensions.height+4);
            this.ctx.strokeStyle = window.styles.hoverText.box.stroke.color;
            this.ctx.lineWidth = window.styles.hoverText.box.stroke.lineWidth;
            this.ctx.strokeRect(offsetH+this.hoverPos[0]-2, this.hoverPos[1]-(dimensions.height/2+2), dimensions.width+4, dimensions.height+4);
            this.ctx.fillStyle = window.styles.hoverText.text.color;
            this.ctx.font = window.styles.hoverText.text.font;
            this.ctx.fillText(this.hoverText, offsetH+this.hoverPos[0], this.hoverPos[1]);
        }
    }


    render() {
        var canvasSize = this._convertToCanvasScale([this.canvas.width(), this.canvas.height()]);
        this.ctx.fillStyle = window.styles.background;
        this.ctx.fillRect(0, 0, canvasSize[0], canvasSize[1]);
        var self = this;
        var rescale = function(coords) {
            return self._convertToCanvasScale(coords);
        }
        for(var key in this.predictions) {
            this.predictions[key].draw(this.ctx, this.canvas, rescale);
        }
        for(var key in this.annotations) {
            this.annotations[key].draw(this.ctx, this.canvas, rescale);
        }

        // show hover message
        this._drawHoverText();
    }
 }




 class ClassificationEntry extends AbstractDataEntry {
     /*
        Implementation for image classification.
        Expected keys in 'properties' input:
        - entryID: identifier for the data entry
        - fileName: name of the image file to retrieve from data server
        - predictedLabel: optional, array of label ID strings for model predictions
        - predictedConfidence: optional, float for model prediction confidence

        If predictedLabel is provided, a thin border tinted according to the
        arg max (i.e., the predicted label) is drawn around the image.

        As soon as the user clicks into the image, a thick border is drawn,
        colored w.r.t. the user-selected class. A second click removes the user
        label again.
     */
    constructor(entryID, properties) {
        super(entryID, properties);

        // setup label instances
        this.labelInstance = null;  // will be set by the user
        this.noLabel = false;       // will be set to true if user explicitly de-clicks the image (to avoid carry-over of existing annotations or predictions)
        this.defaultLabelInstance = this._getDefaultLabelInstance();

        this._setup_markup();
    }

    _getDefaultLabelInstance() {
        for(var key in this.annotations) {
            if(this.annotations[key].label != null) {
                return this.annotations[key];
            }
        }

        // no annotation found, fallback for predictions
        for(var key in this.predictions) {
            if(this.predictions[key].label != null) {
                return this.predictions[key];
            }
        }
        return null;
    }

    _setup_markup() {
        var self = this;
        super._setup_markup();
        $(this.canvas).css('cursor', 'pointer');

        // click handler
        this.markup.click(function(event) {
            self.toggleUserLabel(event.altKey);
        });
        
        // image
        this.image.onload = function() {
            self.render();
        };
        super.loadImage();
    }

    toggleUserLabel(removeLabel) {
        if(removeLabel) {
            this.labelInstance = null;
            this.noLabel = true;
            this.defaultLabelInstance = null;

        } else {
            // set label instance to current label class
            this.labelInstance = new LabelAnnotation(this.entryID+'_anno', {'label':window.labelClassHandler.getActiveClassID()}, 'userAnnotation');
        }
        this.render();
    }

    render() {
        var canvasSize = this._convertToCanvasScale([this.canvas.width(), this.canvas.height()]);
        this.ctx.fillStyle = window.styles.background;
        this.ctx.fillRect(0, 0, canvasSize[0], canvasSize[1]);
        this.ctx.drawImage(this.image, 0, 0, canvasSize[0], canvasSize[1]);
        var self = this;
        var rescale = function(coords) {
            return self._convertToCanvasScale(coords);
        }
        if(this.labelInstance != null) {
            this.labelInstance.draw(this.ctx, this.canvas, rescale);
        } else if(this.defaultLabelInstance != null) {
            this.defaultLabelInstance.draw(this.ctx, this.canvas, rescale);
        }
    }
 }




class PointAnnotationEntry extends AbstractDataEntry {
    /*
        Implementation for point annotations (note: just image coordinates,
        no bounding boxes).

        Expected keys in 'properties' input:
        - entryID: identifier for the data entry
        - fileName: name of the image file to retrieve from data server
        - pointAnnotations: optional, dict of point entries. Key = identifier
          of the annotation entry. Value:
            - annotationID: identifier of the annotation entry
            - x: float, x coordinate of the point in absolute pixel values
            - y: float, y coordinate of the point in absolute pixel values
            - predictedLabel: optional, label class identifier for the point as predicted by a model
            - predictedConfidence: optional, float value as model confidence of the point
            - userLabel: optional, label class identifier as provided by a user

        If predictedPoints is provided, crosshairs, tinted according to the color of the 
        corresponding entry at index in 'predictedLabel', will be drawn for each point.
     */
    constructor(entryID, properties) {
        super(entryID, properties);
        this.annotations = {};
        this.canvasID = entryID + '_canvas';
        this._setup_markup();
    }

    _setup_markup() {
        var self = this;
        this.markup = super._setup_markup();

        // interaction handlers
        this.canvas.mousemove(function(event) {
            self._canvas_mousein(event);
        });
        this.canvas.keydown(function(event) {
            //TODO: make hover text change when delete key is held down
            self._canvas_mousein(event);
        });
        this.canvas.mouseleave(function(event) {
            self._canvas_mouseout(event);
        });
        this.canvas.click(function(event) {
            self._canvas_click(event);
        });

        // image
        this.image.onload = function() {
            self.render();
        };
        super.loadImage();
    }


    /* canvas interaction functionalities */
    _getClosestPoint(coordinates) {
        /*
            Returns the point whose coordinates form the closest Euclidean
            distance to the provided coordinates, if within a default
            tolerance threshold. Otherwise returns null.
        */
        var tolerance = 20;      //TODO: make user-adjustable; adjust for canvas scale?

        var minDist = 1e9;
        var argMin = null;

        for(var key in this.predictions) {
            if(this.predictions[key] instanceof PointAnnotation) {
                var dist = this.predictions[key].euclideanDistance(coordinates[0], coordinates[1]);
                if((dist < minDist) && (dist <= tolerance)) {
                    minDist = dist;
                    argMin = this.predictions[key];
                }
            }
        }

        for(var key in this.annotations) {
            if(this.annotations[key] instanceof PointAnnotation) {
                var dist = this.annotations[key].euclideanDistance(coordinates[0], coordinates[1]);
                if((dist < minDist) && (dist <= tolerance)) {
                    minDist = dist;
                    argMin = this.annotations[key];
                }
            }
        }
        return argMin;
    }

    _canvas_mousein(event) {
        var coords = this._getCanvasCoordinates(event);
        var coords_scaled = this._convertCoordinates(coords, 'scaled');
        this.hoverText = null;
        this.hoverPos = null;

        // check if another point is close-by and show message
        var closest = this._getClosestPoint(coords);
        if(closest != null) {
            // point found
            if(event.altKey) {
                this.hoverText = 'remove point';
                this.hoverPos = coords_scaled;
            } else if(closest['label'] != window.labelClassHandler.getActiveClassID()) {
                this.hoverText = 'change to "' + window.labelClassHandler.getActiveClassName() + '"';
                this.hoverPos = coords_scaled;
            }
        }
        this.render();
    }

    _canvas_mouseout(event) {
        // clear hover text
        this.hoverText = null;
        this.hoverPos = null;
        this.render();
    }

    _canvas_click(event) {
        var coords = this._getCanvasCoordinates(event);

        // check if another point is close-by
        var closest = this._getClosestPoint(coords);
        if(closest != null) {
            // point found; alter or delete it
            if(event.altKey) {
                // remove point
                delete this.annotations[closest.annotationID];     //TODO: undo history?
            } else if(closest['label'] != window.labelClassHandler.getActiveClassID()) {
                // change label of closest point
                closest['label'] = window.labelClassHandler.getActiveClassID();
            }
        } else {
            // no point in proximity; add new
            var key = this['entryID'] + '_' + coords[0] + '_' + coords[1];
            var props = {
                'x': coords[0],
                'y': coords[1],
                'label': window.labelClassHandler.getActiveClassID()
            };
            this.annotations[key] = new PointAnnotation(key, props, 'annotation');
        }

        this.render();
    }
}