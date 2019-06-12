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

        this._setup_viewport();
        this._setup_markup();
        this._createImageEntry();
        this._parseLabels(properties);
    }

    _setup_viewport() {
        var self = this;
        if(window.dataType == 'images') {
            // create canvas
            this.canvas = $('<canvas id="'+this.canvasID+'" width="'+window.defaultImage_w+'" height="'+window.defaultImage_h+'"></canvas>');
            this.canvas.ready(function() {
                self.viewport.resetViewport();
            });
            this.canvas.css('cursor', 'crosshair');
            this.viewport = new ImageViewport(this.canvas);

        } else {
            // maps
            throw Error('Maps not yet implemented.');
        }
    }

    _addElement(element) {
        var key = element['annotationID'];
        if(element['type'] == 'annotation') {
            this.annotations[key] = element;
        } else if(element['type'] == 'prediction') {
            this.predictions[key] = element;
        }
        this.viewport.addRenderElement(element.getRenderElement());
    }

    _updateElement(element) {
        var key = element['annotationID'];
        if(element['type'] == 'annotation') {
            this.annotations[key] = element;
        } else if(element['type'] == 'prediction') {
            this.predictions[key] = element;
        }
        this.viewport.updateRenderElement(
            this.viewport.indexOfRenderElement(element.getRenderElement()),
            element.getRenderElement()
        );
    }

    _removeElement(element) {
        this.viewport.removeRenderElement(element.getRenderElement());
        if(element['type'] == 'annotation') {
            delete this.annotations[element['annotationID']];
        } else if(element['type'] == 'prediction') {
            delete this.predictions[element['annotationID']];
        }
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
                var prediction = window.parseAnnotation(key, properties['predictions'][key], 'prediction');
                this._addElement(prediction);
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
                    this._addElement(annotation);
                }
            }
        }
    }

    _createImageEntry() {
        this.imageEntry = new ImageElement(this.viewport, this.getImageURI(), window.defaultImage_w, window.defaultImage_h);
        this.viewport.addRenderElement(this.imageEntry);
    }

    _setup_markup() {
        this.markup = $('<div class="entry"></div>');
        this.markup.append(this.canvas);
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

    // /* canvas and drawing functionalities */
    // _getCanvasCoordinates(event) {
    //     var posX = (event.pageX - this.canvas.offset().left);
    //     var posY = (event.pageY - this.canvas.offset().top);
    //     return [posX, posY];
    // }

    // _getScaledCanvasCoordinates(event) {
    //     var coords = this._getCanvasCoordinates(event);
    //     return this._convertCoordinates(coords, 'scaled');
    // }

    // _getCanvasScaleFactors() {
    //     var scaleX = window.defaultImage_w / this.canvas.width();
    //     var scaleY = window.defaultImage_h / this.canvas.height();
    //     return [scaleX, scaleY];
    // }

    // _convertCoordinates(coords, outputFormat) {
    //     var scales = this._getCanvasScaleFactors();
    //     if(outputFormat == 'scaled') {
    //         return [scales[0]*coords[0], scales[1]*coords[1]];
    //     } else {
    //         return [coords[0]/scales[0], coords[1]/scales[1]];
    //     }
    // }

    // _convertToCanvasScale(coords) {
    //     return this._convertCoordinates(coords, 'scaled');
    // }

    // _drawHoverText() {
    //     //TODO: switch text position to left if too close to right side (top if too close to bottom)
    //     if(this.hoverPos != null && this.hoverText != null) {
    //         var dimensions = this.ctx.measureText(this.hoverText);
    //         dimensions.height = window.styles.hoverText.box.height;
    //         var offsetH = window.styles.hoverText.offsetH;
    //         this.ctx.fillStyle = window.styles.hoverText.box.fill;
    //         this.ctx.fillRect(offsetH+this.hoverPos[0]-2, this.hoverPos[1]-(dimensions.height/2+2), dimensions.width+4, dimensions.height+4);
    //         this.ctx.strokeStyle = window.styles.hoverText.box.stroke.color;
    //         this.ctx.lineWidth = window.styles.hoverText.box.stroke.lineWidth;
    //         this.ctx.strokeRect(offsetH+this.hoverPos[0]-2, this.hoverPos[1]-(dimensions.height/2+2), dimensions.width+4, dimensions.height+4);
    //         this.ctx.fillStyle = window.styles.hoverText.text.color;
    //         this.ctx.font = window.styles.hoverText.text.font;
    //         this.ctx.fillText(this.hoverText, offsetH+this.hoverPos[0], this.hoverPos[1]);
    //     }
    // }


    // render() {
    //     var canvasSize = this._convertToCanvasScale([this.canvas.width(), this.canvas.height()]);
    //     this.ctx.fillStyle = window.styles.background;
    //     this.ctx.fillRect(0, 0, canvasSize[0], canvasSize[1]);
    //     this.ctx.drawImage(this.image, 0, 0, canvasSize[0], canvasSize[1]);
    //     var self = this;
    //     var rescale = function(coords) {
    //         return self._convertToCanvasScale(coords);
    //     }
    //     for(var key in this.predictions) {
    //         this.predictions[key].draw(this.ctx, this.canvas, rescale);
    //     }
    //     for(var key in this.annotations) {
    //         this.annotations[key].draw(this.ctx, this.canvas, rescale);
    //     }

    //     // show hover message
    //     this._drawHoverText();
    // }

    render() {
        this.viewport.render();
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

        this.hoverTextElement = new HoverTextElement(null, null, 1);
        this.viewport.addRenderElement(this.hoverTextElement);

        // click handler
        this.markup.click(function(event) {
            self.toggleUserLabel(event.altKey);
        });

        // tooltip for label change
        this.markup.mousemove(function(event) {
            var pos = self.viewport.getCanvasCoordinates(event, false);
            self.hoverTextElement.position = pos;
            if(event.altKey) {
                this.hoverTextElement.setProperty('text', 'mark as unlabeled');
            }
            if(self.labelInstance == null) {
                this.hoverTextElement.setProperty('text', 'set label to "' + window.labelClassHandler.getActiveClassName() + '"');
            } else if(self.labelInstance.label != window.labelClassHandler.getActiveClassID()) {
                this.hoverTextElement.setProperty('text', 'change label to "' + window.labelClassHandler.getActiveClassName() + '"');
            } else {
                this.hoverTextElement.setProperty('text', null);
            }
            self.render();
        });
        this.markup.mouseout(function(event) {
            this.hoverTextElement.setProperty('text', null);
            self.render();
        });
    }

    toggleUserLabel(removeLabel) {
        if(removeLabel) {
            this._removeElement(this.labelInstance);
            // this.viewport.removeRenderElement(this.labelInstance.getRenderElement());
            // this.labelInstance = null;
            this.noLabel = true;
            this.defaultLabelInstance = null;

        } else {
            // set label instance to current label class
            var key = this.entryID + '_anno';
            this.labelInstance = new LabelAnnotation(key, {'label':window.labelClassHandler.getActiveClassID()}, 'userAnnotation');
            this._addElement(this.labelInstance);
            // this.viewport.addRenderElement(this.labelInstance.getRenderElement());
        }
        this.render();
    }
 }




class PointAnnotationEntry extends AbstractDataEntry {
    /*
        Implementation for point annotations (note: just image coordinates,
        no bounding boxes).
     */
    constructor(entryID, properties) {
        super(entryID, properties);
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

        this.hoverTextElement = new HoverTextElement(null, null, 1);
        this.viewport.addRenderElement(this.hoverTextElement);

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
        this.canvas.mouseup(function(event) {
            self._canvas_click(event);
        });
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
        var coords = this.viewport.getCanvasCoordinates(event, false);
        var coords_scaled = this.viewport.getCanvasCoordinates(event, true);
        this.hoverTextElement.setProperty('text', null);
        this.hoverTextElement.position = coords_scaled;

        // check if another point is close-by and show message
        var closest = this._getClosestPoint(coords);
        if(closest != null) {
            // point found
            if(event.altKey) {
                this.hoverTextElement.setProperty('text', 'remove point');
            } else if(closest['label'] != window.labelClassHandler.getActiveClassID()) {
                this.hoverTextElement.setProperty('text', 'change to "' + window.labelClassHandler.getActiveClassName() + '"');
            }
        }
        this.render();
    }

    _canvas_mouseout(event) {
        // clear hover text
        this.hoverTextElement.setProperty('text', null);
        this.render();
    }

    _canvas_click(event) {
        var coords = this.viewport.getCanvasCoordinates(event, false);

        // check if another point is close-by
        var closest = this._getClosestPoint(coords);
        if(closest != null) {
            // point found; alter or delete it
            if(event.altKey) {
                // remove point
                //TODO: undo history?
                this._removeElement(closest);
            } else if(closest['label'] != window.labelClassHandler.getActiveClassID()) {
                // change label of closest point
                closest.setProperty('label', window.labelClassHandler.getActiveClassID());
            }
            // this._updateElement(closest);
        } else if(!event.altKey) {
            // no point in proximity; add new
            var key = this['entryID'] + '_' + coords[0] + '_' + coords[1];
            var props = {
                'x': coords[0],
                'y': coords[1],
                'label': window.labelClassHandler.getActiveClassID()
            };
            var annotation = new PointAnnotation(key, props, 'annotation');
            this._addElement(annotation);
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
        this._setup_markup();
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

        this.hoverTextElement = new HoverTextElement(null, null, 1);
        this.viewport.addRenderElement(this.hoverTextElement);

        // interaction handlers
        this.canvas.mousemove(function(event) {
            self._canvas_mousemove(event);
        });
        this.canvas.mousedown(function(event) {
            self._canvas_mousedown(event);
        });
        this.canvas.mouseleave(function(event) {
            self._canvas_mouseout(event);
        });
        this.canvas.mouseup(function(event) {
            self._canvas_mouseup(event);
        });
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

    _canvas_mouseout(event) {
        // clear hover text
        this.hoverTextElement.setProperty('text', null);
        this.render();
    }

    _createAnnotation(event) {
        var coords = this.viewport.getCanvasCoordinates(event, false);
        var key = this['entryID'] + '_' + new Date().getMilliseconds();
        var props = {
            'x': coords[0],
            'y': coords[1],
            'width': 0,
            'height': 0,
            'label': window.labelClassHandler.getActiveClassID()
        };
        this.activeAnnotationHandle = 'se';
        this.activeAnnotation = new BoundingBoxAnnotation(key, props, 'annotation');
        this._addElement(this.activeAnnotation);
    }

    _setActiveAnnotation(event) {
        var coords_full = this.viewport.getCanvasCoordinates(event, false);
        var closest = this._getClosestBBox(coords_full, false);
        this.activeAnnotation = closest;

        // resize handles
        if(closest == null) {
            this.viewport.removeRenderElement(this.resizeHandles);
            this.resizeHandles == null;
        } else {    //if(this.resizeHandles == null) {
            this.resizeHandles = this.activeAnnotation.getResizeHandles(false, 1);
            this.viewport.addRenderElement(this.resizeHandles);
            this.canvas.css('cursor', 'pointer');
        }
    }

    _deleteActiveAnnotation() {
        this._removeElement(this.activeAnnotation);
        this.viewport.removeRenderElement(this.resizeHandles);
        this.activeAnnotation = null;
        this.activeAnnotationHandle = null;
        this.resizeHandles = null;
        this.canvas.css('cursor', 'pointer');
        this.hoverTextElement.setProperty('text', null);
    }

    _updateActiveAnnotation(event) {
        /*
            Listens to mouse drag movements and modifies the annotation's
            coordinates accordingly.
            Also takes care of the resize handles.
        */
        if(this.activeAnnotation == null || this.activeAnnotationHandle == null) return;

        // calc. new coordinates
        var currentExtent = this.activeAnnotation.getExtent();
        var mousePos = this.viewport.getCanvasCoordinates(event, false);
        if(this.activeAnnotationHandle.includes('w')) {
            var width = currentExtent[2] - mousePos[0];
            var x = mousePos[0] + width/2;
            this.activeAnnotation.setProperty('width', width);
            this.activeAnnotation.setProperty('x', x);
        }
        if(this.activeAnnotationHandle.includes('e')) {
            var width = mousePos[0] - currentExtent[0];
            var x = mousePos[0] - width/2;
            this.activeAnnotation.setProperty('width', width);
            this.activeAnnotation.setProperty('x', x);
        }
        if(this.activeAnnotationHandle.includes('n')) {
            var height = currentExtent[3] - mousePos[1];
            var y = mousePos[1] + height/2;
            this.activeAnnotation.setProperty('height', height);
            this.activeAnnotation.setProperty('y', y);
        }
        if(this.activeAnnotationHandle.includes('s')) {
            var height = mousePos[1] - currentExtent[1];
            var y = mousePos[1] - height/2;
            this.activeAnnotation.setProperty('height', height);
            this.activeAnnotation.setProperty('y', y);
        }
        if(this.activeAnnotationHandle.includes('c')) {
            var prevMousePosScaled = this.mousePos;
            this.activeAnnotation.setProperty('x', this.activeAnnotation.x + mousePos[0] - prevMousePosScaled[0]);
            this.activeAnnotation.setProperty('y', this.activeAnnotation.y + mousePos[1] - prevMousePosScaled[1]);
        }
    }

    _drawCrosshairLines(coords, visible) {
        if(this.crosshairLines == null && visible) {
            // create
            var vertLine = new LineElement(coords[0], 0, coords[0], this.canvas.height(),
                                window.styles.crosshairLines.strokeColor,
                                window.styles.crosshairLines.lineWidth,
                                window.styles.crosshairLines.lineDash,
                                1);
            var horzLine = new LineElement(0, coords[1], this.canvas.width(), coords[1],
                                window.styles.crosshairLines.strokeColor,
                                window.styles.crosshairLines.lineWidth,
                                window.styles.crosshairLines.lineDash,
                                1);
            this.crosshairLines = new ElementGroup([vertLine, horzLine], 1);
            this.viewport.addRenderElement(this.crosshairLines);

        } else {
            if(visible) {
                // update
                this.crosshairLines.elements[0].setProperty('startX', coords[0]);
                this.crosshairLines.elements[0].setProperty('endX', coords[0]);
                this.crosshairLines.elements[1].setProperty('startY', coords[1]);
                this.crosshairLines.elements[1].setProperty('endY', coords[1]);
            } else {
                // remove
                this.viewport.removeRenderElement(this.crosshairLines);
                this.crosshairLines = null;
            }
        }
    }

    _canvas_mousedown(event) {
        this.mouseDrag = true;
        var coords = this.viewport.getCanvasCoordinates(event, true);
        this.mousedownCoords = coords;

        // check functionality
        if(window.interfaceControls.addAnnotation) {
            // start creating a new bounding box
            this._createAnnotation(event);
            this._setActiveAnnotation(event);
        } else {
            // find closest bbox (if there) and set active
            if(this.activeAnnotation == null)
                this._setActiveAnnotation(event);
        }
        this.render();
    }

    _canvas_mousemove(event) {
        var coords = this.viewport.getCanvasCoordinates(event, false);

        // update crosshair lines
        this._drawCrosshairLines(coords, window.interfaceControls.addAnnotation);

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
                this.activeAnnotationHandle = this.activeAnnotation.getClosestHandle(this.mousePos, window.annotationProximityTolerance);

                // show tooltip
                if(this.activeAnnotationHandle == null) {
                    this.hoverTextElement.setProperty('text', null);
                } else if(this.activeAnnotationHandle == 'c') {
                    if(event.altKey) {
                        this.hoverTextElement.setProperty('text', 'delete box');
                        this.canvas.css('cursor', 'pointer');
                    } else {
                        var text = 'Move (drag)';
                        if(this.activeAnnotation.label != window.labelClassHandler.getActiveClassID()) {
                            text += ' or change to "' + window.labelClassHandler.getActiveClassName() + '" (click)';
                        }
                        this.hoverTextElement.setProperty('text', text);
                        this.canvas.css('cursor', 'move');
                    }
                } else {
                    this.hoverTextElement.setProperty('position', coords);
                    this.hoverTextElement.setProperty('text', 'resize');
                }
            }
        }
        this.mousePos = coords;
        this.hoverTextElement.setProperty('position', coords);
        this.render();
    }

    _canvas_mouseup(event) {
        this.mouseDrag = false;

        // check functionality
        if(window.interfaceControls.addAnnotation) {
            // new annotation completed
            window.interfaceControls.addAnnotation = false;
        }
        this._setActiveAnnotation(event);
        if(this.activeAnnotationHandle == null) {
            this._setActiveAnnotation(event);

        } else {
            // adjust active annotation
            if(event.altKey) {
                // delete it
                this._deleteActiveAnnotation();

            } else if(this.activeAnnotationHandle == 'c') {
                // change label
                this.activeAnnotation.setProperty('label', window.labelClassHandler.getActiveClassID());
            }
        }
        
        // clear temporary properties
        // this.activeAnnotationHandle = null;
        this.mousedownCoords = null;

        this.render();
    }

    // _drawAdjustMarker(coords) {
    //     var adjustBoxSize = 8;
    //     this.ctx.fillRect(coords[0] - adjustBoxSize/2,
    //         coords[1] - adjustBoxSize/2,
    //         adjustBoxSize, adjustBoxSize);
    //     this.ctx.strokeRect(coords[0] - adjustBoxSize/2,
    //         coords[1] - adjustBoxSize/2,
    //         adjustBoxSize, adjustBoxSize);
    // }

    render() {
        super.render();
        if(window.interfaceControls.addAnnotation) {
            // // show dashed lines for easier orientation
            // if(this.mousePos != null) {
            //     this.ctx.strokeStyle = '#000000';
            //     this.ctx.lineWidth = 1;
            //     this.ctx.setLineDash([4, 4]);
            //     this.ctx.beginPath();
            //     this.ctx.moveTo(this.mousePos[0], 0);
            //     this.ctx.lineTo(this.mousePos[0], this.canvas.height());
            //     this.ctx.moveTo(0, this.mousePos[1]);
            //     this.ctx.lineTo(this.canvas.width(), this.mousePos[1]);
            //     this.ctx.stroke();
            //     this.ctx.closePath();
            //     this.ctx.setLineDash([]);
            // }

            // // show currently drawn bbox
            // if(this.mousedownCoords != null && this.mouseupCoords != null) {
            //     this.ctx.strokeStyle = window.labelClassHandler.getActiveColor();
            //     this.ctx.lineWidth = 2;
            //     this.ctx.strokeRect(this.mousedownCoords[0], this.mousedownCoords[1],
            //             this.mouseupCoords[0]-this.mousedownCoords[0], this.mouseupCoords[1]-this.mousedownCoords[1]);
            // }

        } else {
            // check if there's an active box
            if(this.activeAnnotation != null) {

                // // highlight box
                // var adjustedXY = this.viewport.scaleToCanvas([this.activeAnnotation.x, this.activeAnnotation.y]);
                // var adjustedWH = this.viewport.scaleToCanvas([this.activeAnnotation.w, this.activeAnnotation.h]);
                
                // this.ctx.fillStyle = '#FFFFFF';
                // this.ctx.strokeStyle = '#000000';
                // this.ctx.lineWidth = 1;

                // // draw markers
                // this._drawAdjustMarker([adjustedXY[0] - adjustedWH[0]/2, adjustedXY[1] - adjustedWH[1]/2]);
                // this._drawAdjustMarker([adjustedXY[0] - adjustedWH[0]/2, adjustedXY[1]]);
                // this._drawAdjustMarker([adjustedXY[0] - adjustedWH[0]/2, adjustedXY[1] + adjustedWH[1]/2]);
                // this._drawAdjustMarker([adjustedXY[0], adjustedXY[1] - adjustedWH[1]/2]);
                // this._drawAdjustMarker([adjustedXY[0], adjustedXY[1] + adjustedWH[1]/2]);
                // this._drawAdjustMarker([adjustedXY[0] + adjustedWH[0]/2, adjustedXY[1]]);
                // this._drawAdjustMarker([adjustedXY[0] + adjustedWH[0]/2, adjustedXY[1] - adjustedWH[1]/2]);
                // this._drawAdjustMarker([adjustedXY[0] + adjustedWH[0]/2, adjustedXY[1] + adjustedWH[1]/2]);

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