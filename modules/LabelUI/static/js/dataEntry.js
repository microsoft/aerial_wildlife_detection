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
        this.imageEntry = new ImageElement(this.entryID + '_image', this.viewport, this.getImageURI(), window.defaultImage_w, window.defaultImage_h);
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

        this.hoverTextElement = new HoverTextElement(this.entryID + '_hoverText', null, null, 5);
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
                self.hoverTextElement.setProperty('text', 'mark as unlabeled');
            }
            if(self.labelInstance == null) {
                self.hoverTextElement.setProperty('text', 'set label to "' + window.labelClassHandler.getActiveClassName() + '"');
            } else if(self.labelInstance.label != window.labelClassHandler.getActiveClassID()) {
                self.hoverTextElement.setProperty('text', 'change label to "' + window.labelClassHandler.getActiveClassName() + '"');
            } else {
                self.hoverTextElement.setProperty('text', null);
            }
            self.render();
        });
        this.markup.mouseout(function(event) {
            self.hoverTextElement.setProperty('text', null);
            self.render();
        });
    }

    toggleUserLabel(removeLabel) {
        if(removeLabel) {
            this._removeElement(this.labelInstance);
            this.noLabel = true;
            this.defaultLabelInstance = null;

        } else {
            // Set label instance to current label class.
            // Remove label if it's the same.
            var key = this.entryID + '_anno';
            if(this.labelInstance == null) {
                this.labelInstance = new Annotation(key, {'label':window.labelClassHandler.getActiveClassID()}, 'userAnnotation');
                this._addElement(this.labelInstance);
                this.noLabel = false;
            } else if(this.labelInstance.label == window.labelClassHandler.getActiveClassID()) {
                this._removeElement(this.labelInstance);
                this.labelInstance = null;
                this.noLabel = true;
                this.defaultLabelInstance = null;
            } else {
                this.labelInstance.setProperty('label', window.labelClassHandler.getActiveClassID());
            }
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

        this.hoverTextElement = new HoverTextElement(this.entryID + '_hoverText', null, null, 5);
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

        this.hoverTextElement = new HoverTextElement(this.entryID + '_hoverText', null, null, 5);
        this.viewport.addRenderElement(this.hoverTextElement);

        // interaction handlers
        this.viewport.addCallback(this.entryID, 'mousedown', function(event) {
            self._canvas_mousedown(event);
        });
        this.viewport.addCallback(this.entryID, 'mousemove', function(event) {
            self._canvas_mousemove(event);
        });
        this.viewport.addCallback(this.entryID, 'mouseup', function(event) {
            self._canvas_mouseup(event);
        });
        // this.viewport.addCallback(this.entryID, 'mouseleave', function(event) {
        //     self._canvas_mouseleave(event);
        // });
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
        // this.activeAnnotationHandle = 'se';
        var anno = new Annotation(key, props, 'annotation');
        this._addElement(anno);
        anno.getRenderElement().registerAsCallback(this.viewport);
        anno.getRenderElement().setActive(true, this.viewport);
        // manually fire mousedown event on annotation
        anno.getRenderElement()._mousedown_event(event, this.viewport);
    }

    _deleteActiveAnnotations() {
        for(var key in this.annotations) {
            if(this.annotations[key].isActive()) {
                this.annotations[key].getRenderElement().deregisterAsCallback(this.viewport);
                this._removeElement(this.annotations[key]);
            }
        }
    }

    _drawCrosshairLines(coords, visible) {
        if(this.crosshairLines == null && visible) {
            // create
            var vertLine = new LineElement(this.entryID + '_crosshairX', coords[0], 0, coords[0], this.canvas.height(),
                                window.styles.crosshairLines.strokeColor,
                                window.styles.crosshairLines.lineWidth,
                                window.styles.crosshairLines.lineDash,
                                1);
            var horzLine = new LineElement(this.entryID + '_crosshairY', 0, coords[1], this.canvas.width(), coords[1],
                                window.styles.crosshairLines.strokeColor,
                                window.styles.crosshairLines.lineWidth,
                                window.styles.crosshairLines.lineDash,
                                1);
            this.crosshairLines = new ElementGroup(this.entryID + '_crosshairLines', [vertLine, horzLine], 1);
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
        if(window.interfaceControls.action == window.interfaceControls.actions.ADD_ANNOTATION) {
            // set all currently active boxes inactive
            for(var key in this.annotations) {
                this.annotations[key].setActive(false, this.viewport);
            }

            // start creating a new bounding box
            this._createAnnotation(event);
        }
        this.render();
    }

    _canvas_mousemove(event) {
        var coords = this.viewport.getCanvasCoordinates(event, false);

        // update crosshair lines
        this._drawCrosshairLines(coords, window.interfaceControls.action==window.interfaceControls.actions.ADD_ANNOTATION);

        // update hover text
        var hoverText = null;
        switch(window.interfaceControls.action) {
            case window.interfaceControls.actions.ADD_ANNOTATION:
                hoverText = 'add new \"' + window.labelClassHandler.getActiveClassName() + '"';
                break;
            case window.interfaceControls.actions.REMOVE_ANNOTATIONS:
                var numActive = 0;
                for(var key in this.annotations) {
                    if(this.annotations[key].isActive()) numActive++;
                }
                if(numActive>0) {
                    hoverText = 'Remove ' + numActive + ' annotation';
                    if(numActive>1) hoverText += 's';
                }
                break;
            case window.interfaceControls.actions.DO_NOTHING:
                //TODO: buggy
                for(var key in this.annotations) {
                    if(this.annotations[key].isActive() && 
                        this.annotations[key].label != window.labelClassHandler.getActiveClassID()) {
                        hoverText = 'Change label to "' + window.labelClassHandler.getActiveClassName() + '"';
                        break;
                    }
                }
                break;
        }
        this.hoverTextElement.setProperty('position', coords);
        this.hoverTextElement.setProperty('text', hoverText);

        this.render();
    }

    _canvas_mouseup(event) {
        // this.mouseDrag = false;

        // check functionality
        if(window.interfaceControls.action == window.interfaceControls.actions.ADD_ANNOTATION) {
            // new annotation completed
            //TODO: may fire before other rectangle's events, making them unwantedly active while finishing new rect
            window.interfaceControls.action = window.interfaceControls.actions.DO_NOTHING;

        } else if(window.interfaceControls.action == window.interfaceControls.actions.REMOVE_ANNOTATIONS) {
            this._deleteActiveAnnotations();
            window.interfaceControls.action = window.interfaceControls.actions.DO_NOTHING;

        } else {
            // update annotations to current label
            for(var key in this.annotations) {
                if(this.annotations[key].isActive()) {
                    this.annotations[key].setProperty('label', window.labelClassHandler.getActiveClassID());
                }
            }
        }

        this.render();
    }
}