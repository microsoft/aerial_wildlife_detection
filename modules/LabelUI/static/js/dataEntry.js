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
    }

    _setup_markup() {
        var markup = $('<div class="entry"></div>');
        return markup;
    }

    _createImage() {
        this.image = new Image();
        this.image.width = window.defaultImage_w;
        this.image.height = window.defaultImage_h;
    }

    loadImage() {
        this.image.src = this.getImageURI();
    }

    getImageURI() {
        return window.dataServerURI + this.fileName + '?' + Date.now();
    }

    getProperties() {
        return {
            'fileName': this.fileName
        };
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
        this.predictedLabel = properties['predictedLabel'];
        this.predictedConfidence = properties['predictedConfidence'];
        this.userLabel = properties['userLabel'];

        this._setup_markup();
    }

    getProperties() {
        var props = super.getProperties();
        props['userLabel'] = this.userLabel;

        return props;
    }

    _setup_markup() {
        var self = this;
        super.loadImage();
        this.markup = super._setup_markup();

        // click handler
        this.markup.click(function() {
            self.toggleUserLabel()
        });

        // image
        this.markup.append(this.image);

        this._set_border_style();
    }

    _set_border_style() {
        // specify border decoration
        var style = 'none';
        if(this.userLabel!=null) {
            style = '6px solid ' + window.classes[this.userLabel]['color'];
        } else if(this.predictedLabel!=null) {
            style = '2px solid ' + window.classes[this.predictedLabel]['color'];
        }
        this.markup.css('border', style);
    }

    toggleUserLabel() {
        if(this.userLabel!=null) {
            this.userLabel = null;
        } else {
            // get ID from currently active label class
            var activeClass = window.labelClassHandler.getActiveClass();
            this.userLabel = activeClass['classID'];
            // this.userLabel = Object.keys()[Math.floor(Math.random() * window.classes.length)];
        }
        this._set_border_style();
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
        this._parsePoints(properties);
        this._setup_markup();
    }

    _parsePoints(properties) {
        if(properties.hasOwnProperty('pointAnnotations')) {
            for(var key in properties['pointAnnotations']) {
                var pointAnno = new PointAnnotation(key, properties['pointAnnotations'][key]);
                this.annotations[key] = pointAnno;
            }
        }
    }

    _setup_markup() {
        var self = this;
        this.markup = super._setup_markup();

        // define canvas
        this.canvas = $('<canvas id="'+this.canvasID+'" width="'+window.defaultImage_w+'" height="'+window.defaultImage_h+'"></canvas>');
        this.canvas.css('cursor', 'crosshair');
        this.markup.append(this.canvas);
        this.ctx = this.canvas[0].getContext('2d');

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

       for(var key in this.annotations) {
           var dist = this.annotations[key].euclideanDistance(coordinates[0], coordinates[1]);
           if((dist < minDist) && (dist <= tolerance)) {
               minDist = dist;
               argMin = this.annotations[key];
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
            } else if(closest['userLabel'] != window.labelClassHandler.getActiveClassID() && closest['predictedLabel'] != window.labelClassHandler.getActiveClassID()) {
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
            } else if(closest['userLabel'] != window.labelClassHandler.getActiveClassID()) {
                // change label of closest point
                closest['userLabel'] = window.labelClassHandler.getActiveClassID();
            }
        } else {
            // no point in proximity; add new
            var key = this['entryID'] + '_' + coords[0] + '_' + coords[1];
            var props = {
                'x': coords[0],
                'y': coords[1],
                'userLabel': window.labelClassHandler.getActiveClassID()
            };
            this.annotations[key] = new PointAnnotation(key, props);
        }

        this.render();
    }


    render() {
        this.ctx.drawImage(this.image, 0, 0);
        for(var key in this.annotations) {
            this.annotations[key].draw(this.ctx, this.canvas);
        }

        // show hover message
        super._drawHoverText();
    }

    getProperties() {
        var props = super.getProperties();

        // append points
        var pointProps = {};
        for(var key in this.annotations) {
            pointProps[key] = this.annotations[key].getProperties();
        }
        props['points'] = pointProps;
        return props;
    }
}