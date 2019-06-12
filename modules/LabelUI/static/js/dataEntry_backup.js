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
        this.canvasID = entryID + '_canvas';
        this.fileName = properties['fileName'];

        this._createImage();
        this._parseLabels(properties);
    }

    getAnnotationType() {
        throw Error('Not implemented.');
    }

    getAnnotationControls() {
        throw Error('Not implemented.');
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
                //TODO: make more failsafe?
                var annotation = window.parseAnnotation(key, properties['annotations'][key], 'annotation');

                // Only add annotation if it is of the correct type.
                // Should be handled by the server configuration, but this way
                // we can make it double-failsafe.
                if(annotation.getAnnotationType() == this.getAnnotationType()) {
                    this.annotations[key] = annotation;
                }
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
        if(this.hoverPos != null && this.hoverText != null) {
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
        this.ctx.drawImage(this.image, 0, 0, canvasSize[0], canvasSize[1]);
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

    getAnnotationType() {
        return 'label';
    }

    getAnnotationControls() {
        return null;
    }

    _getDefaultLabelInstance() {
        for(var key in this.annotations) {
            // Only add annotation if it is of the correct type.
            // TODO: implement e.g. mode of labels instead of first?
            if(this.annotations[key].getAnnotationType() == this.getAnnotationType()) {
                return this.annotations[key];
            }
        }

        // no annotation found, fallback for predictions
        for(var key in this.predictions) {
            if(this.predictions[key].getAnnotationType() == this.getAnnotationType()) {
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
        var self = this;
        var rescale = function(coords) {
            return self._convertToCanvasScale(coords);
        }
        super.render();
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
     */
    constructor(entryID, properties) {
        super(entryID, properties);
        this.annotations = {};
        this._setup_markup();
    }

    getAnnotationType() {
        return 'point';
    }

    getAnnotationControls() {
        return null;
    }

    _setup_markup() {
        var self = this;
        super._setup_markup();

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
        var minDist = 1e9;
        var argMin = null;

        for(var key in this.predictions) {
            if(this.predictions[key] instanceof PointAnnotation) {
                var dist = this.predictions[key].euclideanDistance(coordinates);
                if((dist < minDist) && (dist <= window.annotationProximityTolerance)) {
                    minDist = dist;
                    argMin = this.predictions[key];
                }
            }
        }

        for(var key in this.annotations) {
            if(this.annotations[key] instanceof PointAnnotation) {
                var dist = this.annotations[key].euclideanDistance(coordinates);
                if((dist < minDist) && (dist <= window.annotationProximityTolerance)) {
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
                //TODO: undo history?
                if(closest.annotationID in this.annotations) {
                    delete this.annotations[closest.annotationID];
                } else if(closest.annotationID in this.predictions) {
                    delete this.predictions[closest.annotationID];
                }
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




class BoundingBoxAnnotationEntry extends AbstractDataEntry {
    /*
        Implementation for bounding box annotations.
     */
    constructor(entryID, properties) {
        super(entryID, properties);
        this.annotations = {};
        this._setup_markup();

        //TODO
        window.interfaceControls = {};
        window.interfaceControls.addAnnotation = true;
    }

    getAnnotationType() {
        return 'boundingBox';
    }

    getAnnotationControls() {
        //TODO
        return $('<div class="annotationControls">' +
                '<button onclick=>Add box</button>' +
                '</div>');
    }

    _setup_markup() {
        var self = this;
        super._setup_markup();
        this.canvas.css('cursor', 'pointer');

        // interaction handlers
        this.canvas.mousemove(function(event) {
            self._canvas_mousemove(event);
        });
        this.canvas.mousedown(function(event) {
            self._canvas_mousedown(event);
        });
        this.canvas.mouseup(function(event) {
            self._canvas_mouseup(event);
        });
        // this.canvas.click(function(event) {
        //     self._canvas_click(event);
        // });

        // image
        this.image.onload = function() {
            self.render();
        };
        super.loadImage();
    }


    _getClosestBBox(coordinates, forceCorner) {
        /*
            Returns the annotated bounding box that is within a tolerance
            of the provided coordinates _and_ whose center point is closest
            to the coordinates.
        */
        var minDist = 1e9;
        var argMin = null; 
        for(var key in this.annotations) {
            if(this.annotations[key].isInDistance(coordinates, window.annotationProximityTolerance, forceCorner)) {
                var thisMinDist = this.annotations[key].euclideanDistance(coordinates);
                if(thisMinDist < minDist) {
                    minDist = thisMinDist;
                    argMin = this.annotations[key];
                }
            }
        }
        return argMin;
    }


    /* canvas interaction functionalities */
    _addAnnotation() {
        /*
            Takes the monitored start- and end-coordinates that have been
            drawn and creates a new bounding box instance from them.
        */
        this.mousedownCoords = this._convertCoordinates(this.mousedownCoords, 'fullRes');
        this.mouseupCoords = this._convertCoordinates(this.mouseupCoords, 'fullRes');

        var minX = Math.min(this.mousedownCoords[0], this.mouseupCoords[0]);
        var minY = Math.min(this.mousedownCoords[1], this.mouseupCoords[1]);
        var maxX = Math.max(this.mousedownCoords[0], this.mouseupCoords[0]);
        var maxY = Math.max(this.mousedownCoords[1], this.mouseupCoords[1]);
        var wh = [maxX - minX, maxY - minY];
        var xy = [minX + wh[0]/2, minY + wh[1]/2];

        // create instance from coordinates
        var key = this['entryID'] + '_' + xy[0] + '_' + xy[1];
        var props = {
            'x': xy[0],
            'y': xy[1],
            'w': wh[0],
            'h': wh[1],
            'label': window.labelClassHandler.getActiveClassID()
        };
        this.annotations[key] = new BoundingBoxAnnotation(key, props, 'annotation');

        // flush cache
        this.mousedownCoords = null;
        this.mouseupCoords = null;

        // this.render();
    }

    _setActiveAnnotation(event) {
        var coords_full = this._getCanvasCoordinates(event);
        var closest = this._getClosestBBox(coords_full, false);
        this.activeAnnotation = closest;
    }

    _deleteActiveAnnotation() {
        delete this.annotations[this.activeAnnotation.annotationID];
        this.activeAnnotation = null;
        this.activeAnnotationHandle = null;
        this.canvas.css('cursor', 'pointer');
        this.hoverText = null;
    }

    _updateActiveAnnotation(event) {
        /*
            Listens to mouse drag movements and modifies the annotation's
            coordinates accordingly.
        */
        if(this.activeAnnotation == null || this.activeAnnotationHandle == null) return;

        // calc. new coordinates
        var currentExtent = this.activeAnnotation.getExtent();
        var mousePosScaled = this._getCanvasCoordinates(event);
        if(this.activeAnnotationHandle.includes('w')) {
            this.activeAnnotation.w = currentExtent[2] - mousePosScaled[0];
            this.activeAnnotation.x = mousePosScaled[0] + this.activeAnnotation.w/2;
        }
        if(this.activeAnnotationHandle.includes('e')) {
            this.activeAnnotation.w = mousePosScaled[0] - currentExtent[0];
            this.activeAnnotation.x = mousePosScaled[0] - this.activeAnnotation.w/2;
        }
        if(this.activeAnnotationHandle.includes('n')) {
            this.activeAnnotation.h = currentExtent[3] - mousePosScaled[1];
            this.activeAnnotation.y = mousePosScaled[1] + this.activeAnnotation.h/2;
        }
        if(this.activeAnnotationHandle.includes('s')) {
            this.activeAnnotation.h = mousePosScaled[1] - currentExtent[1];
            this.activeAnnotation.y = mousePosScaled[1] - this.activeAnnotation.h/2;
        }
        if(this.activeAnnotationHandle.includes('c')) {
            var prevMousePosScaled = this._convertCoordinates(this.mousePos);
            this.activeAnnotation.x += mousePosScaled[0] - prevMousePosScaled[0];
            this.activeAnnotation.y += mousePosScaled[1] - prevMousePosScaled[1];
        }
    }

    _canvas_mousedown(event) {
        this.mouseDrag = true;
        var coords = this._getScaledCanvasCoordinates(event);
        this.mousedownCoords = coords;

        // check functionality
        if(window.interfaceControls.addAnnotation) {
            // start creating a new bounding box
            

        } else {
            // find closest bbox (if there) and set active
            if(this.activeAnnotation == null)
                this._setActiveAnnotation(event);
        }
        this.render();
    }

    _canvas_mousemove(event) {
        var coords = this._getScaledCanvasCoordinates(event);

        // check functionality
        if(window.interfaceControls.addAnnotation) {
            // update destination coordinates
            this.mouseupCoords = coords;
        }
        if(this.mouseDrag) {
            this._updateActiveAnnotation(event);
        } else {
            if(this.activeAnnotation != null) {
                // bounding box highlighted; find handles
                this.activeAnnotationHandle = this.activeAnnotation.getClosestHandle(this._convertCoordinates(this.mousePos), window.annotationProximityTolerance);

                // show tooltip
                if(this.activeAnnotationHandle == null) {
                    this.hoverPos = null;
                    this.hoverText = null;
                } else if(this.activeAnnotationHandle == 'c') {
                    this.hoverPos = coords;
                    if(event.altKey) {
                        this.hoverText = 'Delete'
                        this.canvas.css('cursor', 'pointer');
                    } else {
                        this.hoverText = 'Move (drag)';
                        if(this.activeAnnotation.label != window.labelClassHandler.getActiveClassID()) {
                            this.hoverText += ' or change to "' + window.labelClassHandler.getActiveClassName() + '" (click)'
                        }
                        this.canvas.css('cursor', 'move');
                    }
                } else {
                    this.hoverPos = coords;
                    this.hoverText = 'Resize';
                }
            }
        }
        this.mousePos = coords;
        this.hoverPos = coords;
        this.render();
    }

    _canvas_mouseup(event) {
        this.mouseDrag = false;

        // check functionality
        if(window.interfaceControls.addAnnotation) {
            // bounding box completed; add to annotations
            window.interfaceControls.addAnnotation = false;
            // this.mousePos = null;
            this._addAnnotation();
        }
        if(this.activeAnnotationHandle == null) {
            this._setActiveAnnotation(event);

        } else {
            // adjust active annotation
            if(event.altKey) {
                // delete it
                this._deleteActiveAnnotation();

            } else if(this.activeAnnotationHandle == 'c') {
                // change label
                this.activeAnnotation.label = window.labelClassHandler.getActiveClassID();
            }
        }
        
        // clear temporary properties
        // this.activeAnnotationHandle = null;
        this.mousedownCoords = null;

        this.render();
    }

    _drawAdjustMarker(coords) {
        var adjustBoxSize = 8;
        this.ctx.fillRect(coords[0] - adjustBoxSize/2,
            coords[1] - adjustBoxSize/2,
            adjustBoxSize, adjustBoxSize);
        this.ctx.strokeRect(coords[0] - adjustBoxSize/2,
            coords[1] - adjustBoxSize/2,
            adjustBoxSize, adjustBoxSize);
    }

    render() {
        super.render();
        if(window.interfaceControls.addAnnotation) {
            // show dashed lines for easier orientation
            if(this.mousePos != null) {
                this.ctx.strokeStyle = '#000000';
                this.ctx.lineWidth = 1;
                this.ctx.setLineDash([4, 4]);
                this.ctx.beginPath();
                this.ctx.moveTo(this.mousePos[0], 0);
                this.ctx.lineTo(this.mousePos[0], this.canvas.height());
                this.ctx.moveTo(0, this.mousePos[1]);
                this.ctx.lineTo(this.canvas.width(), this.mousePos[1]);
                this.ctx.stroke();
                this.ctx.closePath();
                this.ctx.setLineDash([]);
            }

            // show currently drawn bbox
            if(this.mousedownCoords != null && this.mouseupCoords != null) {
                this.ctx.strokeStyle = window.labelClassHandler.getActiveColor();
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(this.mousedownCoords[0], this.mousedownCoords[1],
                        this.mouseupCoords[0]-this.mousedownCoords[0], this.mouseupCoords[1]-this.mousedownCoords[1]);
            }

        } else {
            // check if there's an active box
            if(this.activeAnnotation != null) {
                // highlight box
                var adjustedXY = super._convertCoordinates([this.activeAnnotation.x, this.activeAnnotation.y], 'scaled');
                var adjustedWH = super._convertCoordinates([this.activeAnnotation.w, this.activeAnnotation.h], 'scaled');
                
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.strokeStyle = '#000000';
                this.ctx.lineWidth = 1;

                // draw markers
                this._drawAdjustMarker([adjustedXY[0] - adjustedWH[0]/2, adjustedXY[1] - adjustedWH[1]/2]);
                this._drawAdjustMarker([adjustedXY[0] - adjustedWH[0]/2, adjustedXY[1]]);
                this._drawAdjustMarker([adjustedXY[0] - adjustedWH[0]/2, adjustedXY[1] + adjustedWH[1]/2]);
                this._drawAdjustMarker([adjustedXY[0], adjustedXY[1] - adjustedWH[1]/2]);
                this._drawAdjustMarker([adjustedXY[0], adjustedXY[1] + adjustedWH[1]/2]);
                this._drawAdjustMarker([adjustedXY[0] + adjustedWH[0]/2, adjustedXY[1]]);
                this._drawAdjustMarker([adjustedXY[0] + adjustedWH[0]/2, adjustedXY[1] - adjustedWH[1]/2]);
                this._drawAdjustMarker([adjustedXY[0] + adjustedWH[0]/2, adjustedXY[1] + adjustedWH[1]/2]);

                // adjust cursor
                if(this.activeAnnotationHandle == null) {
                    this.canvas.css('cursor', 'crosshair');
                } else if(this.activeAnnotationHandle == 'c') {
                    this.canvas.css('cursor', 'move');
                } else {
                    this.canvas.css('cursor', this.activeAnnotationHandle+'-resize');
                }
            }
        }
    }
}