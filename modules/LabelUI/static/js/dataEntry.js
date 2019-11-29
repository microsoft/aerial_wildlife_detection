/*
    Definition of a data entry, as shown on a grid on the screen.

    2019 Benjamin Kellenberger
 */

class AbstractDataEntry {
    /*
       Abstract base class for data entries.
    */
    constructor(entryID, properties, disableInteractions) {
        this.entryID = entryID;
        this.canvasID = entryID + '_canvas';
        this.fileName = properties['fileName'];
        this.disableInteractions = disableInteractions;

        // for interaction handlers
        this.mouseDown = false;
        this.mouseDrag = false;

        var self = this;
        this.imageEntry = null;
        this._setup_viewport();
        this._setup_markup();
        this.loadingPromise = this._loadImage(this.getImageURI()).then(image => {
            self._createImageEntry(image);
            self._parseLabels(properties);
            self.startTime = new Date();
            self.render();
            return true;
        })
        .catch(error => {
            self._createImageEntry(null);
            self.render();
            return false;
        });
    }

    _click(event) {
        /*
            Click listener to disable active annotations in other
            data entries (unless shift key held down)
        */
        if(event.shiftKey) return;
        window.activeEntryID = this.entryID;
        window.dataHandler.refreshActiveAnnotations();      //TODO: ugly hack...
    }

    _toggleGoldenQuestion() {
        /*
                Posts to the server to flip the bool about the entry
                being a golden question (if user is admin).
        */
        if(!window.isAdmin) return;

        var self = this;
        self.isGoldenQuestion = !self.isGoldenQuestion;
        var goldenQuestions = {};
        goldenQuestions[self.entryID] = self.isGoldenQuestion;

        $.ajax({
            url: 'setGoldenQuestions',
            method: 'POST',
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            data: JSON.stringify({
                goldenQuestions: goldenQuestions
            }),
            success: function(data) {
                if(data.hasOwnProperty('status') && data['status'] === 0) {
                    // change successful; set flag accordingly
                    if(self.isGoldenQuestion) {
                        self.flag.attr('src', 'static/img/controls/flag_active.svg');
                    } else {
                        self.flag.attr('src', 'static/img/controls/flag.svg');
                    }
                }
            },
            error: function(data) {
                console.log('ERROR')
                console.log(data);
            }
        })
    }

   _setup_viewport() {
       var self = this;
       if(window.dataType == 'images') {
           // create canvas
           this.canvas = $('<canvas id="'+this.canvasID+'" width="'+window.defaultImage_w+'" height="'+window.defaultImage_h+'"></canvas>');
           this.canvas.ready(function() {
               self.viewport.resetViewport();
           });
           this.canvas.css('cursor', window.uiControlHandler.getDefaultCursor());

           this.viewport = new ImageViewport(this.canvas);

       } else {
           // maps
           throw Error('Maps not yet implemented.');
       }
   }

   _addElement(element) {
       if(!element.isValid()) return;
       var key = element['annotationID'];
       if(element['type'] == 'annotation') {
           this.annotations[key] = element;
       } else if(element['type'] == 'prediction' && window.showPredictions) {
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

   _parseLabels(properties) {
       /*
           Iterates through properties object's entries "predictions" and "annotations"
           and creates new primitive instances each.
           Might automatically convert predictions and carry them over to the annotations
           if applicable and specified in the project settings.
       */
       var geometryType_pred = String(window.predictionType);
       var geometryType_anno = String(window.annotationType);

       this.predictions = {};
       this.annotations = {};
       var hasAnnotations = (properties.hasOwnProperty('annotations') && Object.keys(properties['annotations']).length > 0);
       var hasPredictions = (properties.hasOwnProperty('predictions') && Object.keys(properties['predictions']).length > 0);
       var carryOverPredictions = window.carryOverPredictions && hasPredictions && !hasAnnotations && (!properties.hasOwnProperty('viewcount') || properties['viewcount'] == 0);

       if(window.showPredictions || window.carryOverPredictions && hasPredictions) {
           if(window.showPredictions && !hasAnnotations) {
               // add predictions as static, immutable objects (only if entry has not yet been screened by user)
               for(var key in properties['predictions']) {
                   var prediction = new Annotation(key, properties['predictions'][key], geometryType_pred, 'prediction');
                   if(prediction.confidence >= window.showPredictions_minConf) {
                       this._addElement(prediction);
                   }
               }
           }

           if(carryOverPredictions) {
               /*
                   No annotations present for entry (i.e., entry not yet screened by user):
                   show an initial guess provided by the predictions. Convert predictions
                   if required. Also set annotation to "changed", since we assume the user
                   to have visually screened the provided converted predictions, even if they
                   are unchanged (i.e., deemed correct by the user).
               */
               if(window.annotationType === 'labels') {
                   // need image-wide labels
                   if(window.predictionType === 'points' || window.predictionType === 'boundingBoxes') {
                       // check carry-over rule
                       if(window.carryOverRule === 'maxConfidence') {
                           // select arg max
                           var maxConf = -1;
                           var argMax = null;
                           for(var key in properties['predictions']) {
                               var predConf = properties['predictions'][key]['confidence'];
                               if(predConf >= window.carryOverPredictions_minConf && predConf > maxConf) {
                                   maxConf = predConf;
                                   argMax = key;
                               }
                           }
                           if(argMax != null) {
                               // construct new classification entry
                               var id = properties['predictions'][key]['id'];
                               var label = properties['predictions'][key]['label'];
                               var anno = new Annotation(window.getRandomID(), {'id':id, 'label':label, 'confidence':maxConf}, geometryType_anno, 'annotation', true);
                               anno.setProperty('changed', true);
                               this._addElement(anno);
                           }
                       } else if(window.carryOverRule === 'mode') {
                           var counts = {};
                           for(var key in properties['predictions']) {
                               var prediction = new Annotation(window.getRandomID(), properties['predictions'][key], geometryType_anno, 'prediction');
                               if(!(counts.hasOwnProperty(prediction.label))) {
                                   counts[label] = 0;
                               }
                               counts[label] += 1;
                           }
                           // find mode
                           var count = -1;
                           var argMax = null;
                           for(var key in counts) {
                               if(counts[key] > count) {
                                   count = counts[key];
                                   argMax = key;
                               }
                           }
                           // add new label annotation
                           if(argMax != null) {
                               var anno = new Annotation(window.getRandomID(), {'label':argMax}, geometryType_anno, 'annotation', true);
                               anno.setProperty('changed', true);
                               this._addElement(anno);
                           }
                       }
                   }
               } else if(window.annotationType === 'points' && window.predictionType === 'boundingBoxes') {
                   // remove width and height
                   for(var key in properties['predictions']) {
                       var props = properties['predictions'][key];
                       if(props['confidence'] >= window.carryOverPredictions_minConf) {
                           delete props['width'];
                           delete props['height'];
                           var anno = new Annotation(window.getRandomID(), props, geometryType_anno, 'annotation', true);
                           this._addElement(anno);
                       }
                   }
               } else if(window.annotationType === 'boundingBoxes' && window.predictionType === 'points') {
                   // add default width and height
                   for(var key in properties['predictions']) {
                       var props = properties['predictions'][key];
                       if(props['confidence'] >= window.carryOverPredictions_minConf) {
                           props['width'] = window.defaultBoxSize_w;
                           props['height'] = window.defaultBoxSize_h;
                           var anno = new Annotation(window.getRandomID(), props, geometryType_anno, 'annotation', true);
                           anno.setProperty('changed', true);
                           this._addElement(anno);
                       }
                   }
               } else if(window.annotationType === window.predictionType) {
                   // no conversion required
                   for(var key in properties['predictions']) {
                       var props = properties['predictions'][key];
                       if(props['confidence'] >= window.carryOverPredictions_minConf) {
                           var anno = new Annotation(window.getRandomID(), props, geometryType_anno, 'annotation', true);
                           anno.setProperty('changed', true);
                           this._addElement(anno);
                       }
                   }
               }

           }
       }

       if(hasAnnotations) {
           for(var key in properties['annotations']) {
               //TODO: make more failsafe?
               var annotation = new Annotation(key, properties['annotations'][key], geometryType_anno, 'annotation');
               // Only add annotation if it is of the correct type.
               // if(annotation.getAnnotationType() == this.getAnnotationType()) {     //TODO: disabled for debugging purposes
               this._addElement(annotation);
               // }
           }
       }
   }

    _loadImage(imageURI) {
        return new Promise(resolve => {
            const image = new Image();
            image.addEventListener('load', () => {
                resolve(image);
            });
            image.src = imageURI;
        });
    }

    _createImageEntry(image) {
        this.imageEntry = new ImageElement(this.entryID + '_image', image, this.viewport);
        this.viewport.addRenderElement(this.imageEntry);
    }

   _setup_markup() {
        this.markup = $('<div class="entry"></div>');
        this.markup.append(this.canvas);
        var self = this;
        if(!this.disableInteractions)
            this.markup.on('click', (self._click).bind(self));

        // flag for golden questions (if admin)
        if(window.isAdmin) {
            this.flag = $('<img class="golden-question-flag" title="toggle golden question" />');
            if(self.isGoldenQuestion) {
                this.flag.attr('src', 'static/img/controls/flag_active.svg');
            } else {
                this.flag.attr('src', 'static/img/controls/flag.svg');
            }
            this.flag.click(function() {
                // toggle golden question on server
                self._toggleGoldenQuestion();
            });
            this.markup.append(this.flag);
        }
   }

   getImageURI() {
       return window.dataServerURI + this.fileName;    // + '?' + Date.now();
   }

   getProperties(minimal, onlyUserAnnotations) {
       var timeCreated = this.getTimeCreated();
       if(timeCreated != null) timeCreated = timeCreated.toISOString();
       var props = {
           'id': this.entryID,
           'timeCreated': timeCreated,
           'timeRequired': this.getTimeRequired()
       };
       props['annotations'] = [];
       for(var key in this.annotations) {
           if(!onlyUserAnnotations || this.annotations[key].getChanged())
               var annoProps = this.annotations[key].getProperties(minimal);
               
               // append time created and time required
               annoProps['timeCreated'] = this.getTimeCreated();
               var timeRequired = Math.max(0, this.annotations[key].getTimeChanged() - this.getTimeCreated());
               annoProps['timeRequired'] = timeRequired;
               props['annotations'].push(annoProps);
       }
       if(!minimal) {
           props['fileName'] = this.fileName;
           props['predictions'] = {};
           if(!onlyUserAnnotations) {
               for(var key in this.predictions) {
                   props['predictions'][key] = this.predictions[key].getProperties(minimal);
               }
           }
       }
       return props;
   }

   getTimeCreated() {
       // returns the timestamp of the image having successfully loaded
       return (this.imageEntry === null? null : this.imageEntry.getTimeCreated());
   }

   getTimeRequired() {
       // returns the difference between the last annotation's last changed
       // timestamp and the time created
       if(this.getTimeCreated() == null) return null;

       var lastModified = -1;
       for(var key in this.annotations) {
           var nextTimeChanged = this.annotations[key].getTimeChanged();
           if(nextTimeChanged == null) continue;
           if(nextTimeChanged > lastModified) {
               lastModified = nextTimeChanged;
           }
       }

       return Math.max(0, lastModified - this.getTimeCreated());
   }

   setLabel(label) {
       for(var key in this.annotations) {
           this.annotations[key].setProperty('label', label);
       }
       this.render();

       window.dataHandler.updatePresentClasses();
   }

   setPredictionsVisible(visible) {
       for(var key in this.predictions) {
           this.predictions[key].setVisible(visible);
       }
       this.render();
   }

   setAnnotationsVisible(visible) {
       for(var key in this.annotations) {
           this.annotations[key].setVisible(visible);
       }
       this.render();
   }

   setMinimapVisible(visible) {
       this.viewport.setMinimapVisible(visible);
   }

   setAnnotationsInactive() {
       for(var key in this.annotations) {
           this.annotations[key].setActive(false, this.viewport);
       }
       this.render();
   }

   removeActiveAnnotations() {
       for(var key in this.annotations) {
           if(this.annotations[key].isActive()) {
               this.annotations[key].setActive(false, this.viewport);
               this._removeElement(this.annotations[key]);
           }
       }
       this.render();

       window.dataHandler.updatePresentClasses();
   }

   removeAllAnnotations() {
       for(var key in this.annotations) {
           this.annotations[key].setActive(false, this.viewport);
           this._removeElement(this.annotations[key]);
       }
       this.render();

       window.dataHandler.updatePresentClasses();
   }

   toggleActiveAnnotationsUnsure() {
       var active = false;
       for(var key in this.annotations) {
           if(this.annotations[key].isActive()) {
               this.annotations[key].setProperty('unsure', !this.annotations[key].getProperty('unsure'));
               active = true;
           }
       }
       this.render();
       return active;
   }

   getActiveClassIDs() {
       /*
           Returns distinct label class IDs of all annotations and predictions
           present in the entry.
       */
       var classIDs = {};
       for(var key in this.annotations) {
           var label = this.annotations[key].label;
           if(label != null) classIDs[label] = 1;
       }
       for(var key in this.predictions) {
           var label = this.predictions[key].label;
           if(label != null) classIDs[label] = 1;
       }
       return classIDs;
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

       if(this.labelInstance == null) {
           // add a default, blank instance if nothing has been predicted or annotated yet
           var label = (window.enableEmptyClass ? null : window.labelClassHandler.getActiveClassID());
           this._addElement(new Annotation(window.getRandomID(), {'label':label}, 'labels', 'annotation'));
       }

       this._setup_markup();
   }

   getAnnotationType() {
       return 'label';
   }

   _addElement(element) {
       // allow only one label for classification entry
       var key = element['annotationID'];
       if(element['type'] == 'annotation') {
           if(Object.keys(this.annotations).length > 0) {
               // replace current annotation
               var currentKey = Object.keys(this.annotations)[0];
               this.viewport.removeRenderElement(this.annotations[currentKey]);
               delete this.annotations[currentKey];
           }

           // add new annotation from existing
           var unsure = element['geometry']['unsure'];
           var anno = new Annotation(key, {'label':element['label'], 'unsure':unsure}, 'labels', element['type']);
           this.annotations[key] = anno;
           this.viewport.addRenderElement(anno.getRenderElement());
           this.labelInstance = anno;

           // flip text color of BorderStrokeElement if needed
           var htFill = this.labelInstance.geometry.getProperty('fillColor');
           if(htFill != null && window.getBrightness(htFill) >= 92) {
               this.labelInstance.geometry.setProperty('textColor', '#000000');
           } else {
               this.labelInstance.geometry.setProperty('textColor', '#FFFFFF');
           }
           
       } else if(element['type'] == 'prediction' && window.showPredictions) {
           this.predictions[key] = element;
           this.viewport.addRenderElement(element.getRenderElement());
       }

       window.dataHandler.updatePresentClasses();
   }

   _setup_markup() {
       var self = this;
       super._setup_markup();
       $(this.canvas).css('cursor', window.uiControlHandler.getDefaultCursor());

       var htStyle = {
           fillColor: window.styles.hoverText.box.fill,
           textColor: window.styles.hoverText.text.color,
           strokeColor: window.styles.hoverText.box.stroke.color,
           lineWidth: window.styles.hoverText.box.stroke.lineWidth
       };
       this.hoverTextElement = new HoverTextElement(this.entryID + '_hoverText', null, null, 'validArea',
           htStyle,
           5);
       this.viewport.addRenderElement(this.hoverTextElement);

       if(!this.disableInteractions) {
           // click handler
           this.markup.mouseup(function(event) {
               if(window.uiBlocked) return;
               else if(window.uiControlHandler.getAction() === ACTIONS.DO_NOTHING) {
                   if(window.unsureButtonActive) {
                       self.labelInstance.setProperty('unsure', !self.labelInstance.getProperty('unsure'));
                       window.unsureButtonActive = false;
                       self.render();
                   } else {
                       self.toggleUserLabel(event.altKey);
                   }
               }

               window.dataHandler.updatePresentClasses();
           });

           // tooltip for label change
           this.markup.mousemove(function(event) {
               if(window.uiBlocked) return;
               var pos = self.viewport.getRelativeCoordinates(event, 'validArea');

               // offset tooltip position if loupe is active
               if(window.uiControlHandler.showLoupe) {
                   pos[0] += 0.2;  //TODO: does not account for zooming in
               }

               self.hoverTextElement.position = pos;
               if(window.uiControlHandler.getAction() in [ACTIONS.DO_NOTHING,
                   ACTIONS.ADD_ANNOTATION,
                   ACTIONS.REMOVE_ANNOTATIONS]) {
                   if(event.altKey) {
                       self.hoverTextElement.setProperty('text', 'mark as unlabeled');
                       self.hoverTextElement.setProperty('fillColor', window.styles.hoverText.box.fill);
                   } else if(window.unsureButtonActive) {
                       self.hoverTextElement.setProperty('text', 'toggle unsure');
                       self.hoverTextElement.setProperty('fillColor', window.styles.hoverText.box.fill);
                   } else if(self.labelInstance == null) {
                       self.hoverTextElement.setProperty('text', 'set label to "' + window.labelClassHandler.getActiveClassName() + '"');
                       self.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getActiveColor());
                   } else if(self.labelInstance.label != window.labelClassHandler.getActiveClassID()) {
                       self.hoverTextElement.setProperty('text', 'change label to "' + window.labelClassHandler.getActiveClassName() + '"');
                       self.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getActiveColor());
                   } else {
                       self.hoverTextElement.setProperty('text', null);
                   }
               } else {
                   self.hoverTextElement.setProperty('text', null);
               }

               // flip text color if needed
               var htFill = self.hoverTextElement.getProperty('fillColor');
               if(htFill != null && window.getBrightness(htFill) >= 92) {
                   self.hoverTextElement.setProperty('textColor', '#000000');
               } else {
                   self.hoverTextElement.setProperty('textColor', '#FFFFFF');
               }

               // set active (for e.g. "unsure" functionality)
               self.labelInstance.setActive(true);

               self.render();
           });
           this.markup.mouseout(function(event) {
               if(window.uiBlocked) return;
               self.hoverTextElement.setProperty('text', null);
               self.labelInstance.setActive(false);
               self.render();
           });
       }
   }

   setLabel(label) {
       if(this.labelInstance == null) {
           // add new annotation
           var anno = new Annotation(window.getRandomID(), {'label':label}, 'labels', 'annotation');
           this._addElement(anno);

       } else {
           this.labelInstance.setProperty('label', label);
       }

       // flip text color of BorderStrokeElement if needed
       var htFill = this.labelInstance.geometry.getProperty('fillColor');
       if(htFill != null && window.getBrightness(htFill) >= 92) {
           this.labelInstance.geometry.setProperty('textColor', '#000000');
       } else {
           this.labelInstance.geometry.setProperty('textColor', '#FFFFFF');
       }

       this.render();

       window.dataHandler.updatePresentClasses();
   }

   toggleUserLabel(forceRemove) {
       /*
           Toggles the classification label as follows:
           - if the annotation has no label, it is set to the user-specified class
           - if it has the same label as specified, the label is removed if the background
             class is enabled
           - if it has a different label than specified, the label is changed
           - if forceRemove is true, the label is removed (if background class is enabled)
       */
       if(forceRemove && window.enableEmptyClass) {
           this.setLabel(null);

       } else {
           var activeLabel = window.labelClassHandler.getActiveClassID();
           if(this.labelInstance == null) {
               // add new annotation
               var anno = new Annotation(window.getRandomID(), {'label':activeLabel}, 'labels', 'annotation');
               this._addElement(anno);

           } else {
               if(this.labelInstance.label == activeLabel && window.enableEmptyClass) {
                   // same label; disable
                   this.setLabel(null);

               } else {
                   // different label; update
                   this.setLabel(activeLabel);
               }
           }
       }
       this.render();

       window.dataHandler.updatePresentClasses();
   }

   removeAllAnnotations() {
       this.labelInstance.setProperty('label', null);
       this.render();

       window.dataHandler.updatePresentClasses();
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

   setLabel(label) {
       for(var key in this.annotations) {
           if(label == null) {
               this._removeElement(this.annotations[key]);
           } else {
               this.annotations[key].setProperty('label', label);
           }
       }
       this.render();
       window.dataHandler.updatePresentClasses();
   }

   _setup_markup() {
       var self = this;
       super._setup_markup();
       this.canvas.css('cursor', window.uiControlHandler.getDefaultCursor());

       var htStyle = {
           fillColor: window.styles.hoverText.box.fill,
           textColor: window.styles.hoverText.text.color,
           strokeColor: window.styles.hoverText.box.stroke.color,
           lineWidth: window.styles.hoverText.box.stroke.lineWidth,
           lineDash: []
       };
       this.hoverTextElement = new HoverTextElement(this.entryID + '_hoverText', null, [0, 0.99], 'canvas',
           htStyle,
           5);
       this.viewport.addRenderElement(this.hoverTextElement);

       // interaction handlers
       if(!this.disableInteractions) {
           this.viewport.addCallback(this.entryID, 'mousedown', (self._canvas_mousedown).bind(self));
           this.viewport.addCallback(this.entryID, 'mousemove', (self._canvas_mousemove).bind(self));
           this.viewport.addCallback(this.entryID, 'mouseup', (self._canvas_mouseup).bind(self));
           this.viewport.addCallback(this.entryID, 'mouseleave', (self._canvas_mouseleave).bind(self));
       }
   }

   _toggleActive(event) {
       /*
           Sets points active or inactive as follows:
           - if a point is close enough to the event's coordinates:
               - and is not active: it is turned active
               - and is active: nothing happens
               Any other point that is active _and_ is close enough to the point stays active.
               Other points that are active and are not near the coordinates get deactivated,
               unless the shift key is held down.
           - if no point is near the coordinates, every single one is deactivated.
       */
       var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
       var tolerance = this.viewport.transformCoordinates([0,0,window.annotationProximityTolerance,0], 'canvas', true)[2];
       var minDist = 1e9;
       var argMin = null;
       for(var key in this.annotations) {
           var point = this.annotations[key].getRenderElement();
           var dist = point.euclideanDistance(coords);
           if(dist <= tolerance) {
               if(dist < minDist) {
                   minDist = dist;
                   argMin = key;
               }
               if(!(event.shiftKey && this.annotations[key].isActive())) {
                   // deactivate
                   if(this.annotations[key].isVisible()) {
                       this.annotations[key].setActive(false, this.viewport);
                   }
               }
           }
       }

       // handle closest
       if(argMin == null) {
           // deactivate all
           for(var key in this.annotations) {
               if(this.annotations[key].isVisible()) {
                   this.annotations[key].setActive(false, this.viewport);
               }
           }
       } else if(this.annotations[argMin].isVisible()) {
           // set closest one active
           this.annotations[argMin].setActive(true, this.viewport);
       }
   }

   _canvas_mouseout(event) {
       // clear hover text
       if(window.uiBlocked) return;
       this.hoverTextElement.setProperty('text', null);
       this.render();
   }

   _createAnnotation(event) {
       var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
       var props = {
           'x': coords[0],
           'y': coords[1],
           'label': window.labelClassHandler.getActiveClassID()
       };
       var anno = new Annotation(window.getRandomID(), props, 'points', 'annotation');
       this._addElement(anno);
       anno.getRenderElement().registerAsCallback(this.viewport);
       anno.getRenderElement().setActive(true, this.viewport);
       // manually fire mousedown event on annotation
       anno.getRenderElement()._mousedown_event(event, this.viewport);
   }

   _deleteActiveAnnotations(event) {
       var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
       var minDist = 1e9;
       var argMin = null;
       for(var key in this.annotations) {
           var point = this.annotations[key].getRenderElement();
           var dist = point.euclideanDistance(coords);
           var tolerance = this.viewport.transformCoordinates([0,0,window.annotationProximityTolerance,0], 'canvas', true)[2];
           if(dist <= tolerance) {
               if(this.annotations[key].isActive()) {
                   this.annotations[key].getRenderElement().deregisterAsCallback(this.viewport);
                   this._removeElement(this.annotations[key]);
                   return;
               }
               
               if(dist < minDist) {
                   minDist = dist;
                   argMin = key;
               }
           }
       }

       if(argMin != null) {
           // no active annotation found, but clicked within another one; delete it
           this._removeElement(this.annotations[argMin]);
       }
       window.dataHandler.updatePresentClasses();
   }

   _canvas_mousedown(event) {
       if(window.uiBlocked) return;
       this.mouseDown = true;

       // check functionality
       if(window.uiControlHandler.getAction() === ACTIONS.ADD_ANNOTATION) {
           // set all currently active points inactive
           for(var key in this.annotations) {
               if(this.annotations[key].isVisible()) {
                   this.annotations[key].setActive(false, this.viewport);
               }
           }

           // start creating a new point
           this._createAnnotation(event);
       }
       this.render();
   }

   _canvas_mousemove(event) {
       if(window.uiBlocked) return;
       if(this.mouseDown) this.mouseDrag = true;

       var coords = this.viewport.getRelativeCoordinates(event, 'canvas');

       // update hover text
       var hoverText = null;
       switch(window.uiControlHandler.getAction()) {
           case ACTIONS.ADD_ANNOTATION:
               hoverText = 'add new \"' + window.labelClassHandler.getActiveClassName() + '"';
               this.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getActiveColor());
               break;
           case ACTIONS.REMOVE_ANNOTATIONS:
               var numActive = 0;
               for(var key in this.annotations) {
                   if(this.annotations[key].isActive()) numActive++;
               }
               if(numActive>0) {
                   hoverText = 'Remove ' + numActive + ' annotation';
                   if(numActive>1) hoverText += 's';
                   this.hoverTextElement.setProperty('fillColor', window.styles.hoverText.box.fill);
               }
               break;
           case ACTIONS.DO_NOTHING:
               // find point near coordinates
               var anno = null;
               var tolerance = this.viewport.transformCoordinates([0,0,window.annotationProximityTolerance,0], 'canvas', true)[2];
               for(var key in this.annotations) {
                   if(this.annotations[key].geometry.euclideanDistance(coords) <= tolerance) {
                       anno = this.annotations[key];
                   }
               }
               // set text
               if(anno === undefined || anno === null) {
                   hoverText = null;
                   break;
               }
               if(anno.isActive() && anno.label != window.labelClassHandler.getActiveClassID() && !this.mouseDrag) {
                   hoverText = 'move or change label to "' + window.labelClassHandler.getActiveClassName() + '"';
                   this.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getActiveColor());
               } else {
                   // show current class name of point
                   hoverText = window.labelClassHandler.getName(anno.label);
                   this.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getColor(anno.label));
               }
               break;
       }
       // this.hoverTextElement.setProperty('position', coords);

       // flip text color if needed
       var htFill = this.hoverTextElement.getProperty('fillColor');
       if(htFill != null && window.getBrightness(htFill) >= 92) {
           this.hoverTextElement.setProperty('textColor', '#000000');
       } else {
           this.hoverTextElement.setProperty('textColor', '#FFFFFF');
       }

       this.hoverTextElement.setProperty('text', hoverText);

       this.render();
   }

   _canvas_mouseup(event) {
       if(window.uiBlocked) return;

       // check functionality
       if(window.uiControlHandler.getAction() == ACTIONS.ADD_ANNOTATION) {
           // new annotation completed
           //TODO: may fire before other rectangle's events, making them unwantedly active while finishing new rect
           // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);

       } else if(window.uiControlHandler.getAction() == ACTIONS.REMOVE_ANNOTATIONS) {
           this._deleteActiveAnnotations(event);
           // window.uiControlHandler.setAction(ACTIONS.DO_NOTHING);

       } else if(window.uiControlHandler.getAction() == ACTIONS.DO_NOTHING) {
           // update annotations to current label (if active and no mouse dragging [moving] was performed)
           if(!this.mouseDrag) {
               var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
               var tolerance = this.viewport.transformCoordinates([0,0,window.annotationProximityTolerance,0], 'canvas', true)[2];
               for(var key in this.annotations) {
                   if(this.annotations[key].isActive()) {
                       if(event.shiftKey || this.annotations[key].getRenderElement().euclideanDistance(coords) <= tolerance) {
                           this.annotations[key].setProperty('label', window.labelClassHandler.getActiveClassID());
                       }
                   }
               }

               // activate or deactivate
               this._toggleActive(event);
           }
       }

       this.render();

       window.dataHandler.updatePresentClasses();

       this.mouseDown = false;
       this.mouseDrag = false;
   }

   _canvas_mouseleave(event) {
       if(window.uiBlocked) return;
       this.hoverTextElement.setProperty('text', null);
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

   setLabel(label) {
       for(var key in this.annotations) {
           if(label == null) {
               this._removeElement(this.annotations[key]);
           } else {
               this.annotations[key].setProperty('label', label);
           }
       }
       this.render();
       window.dataHandler.updatePresentClasses();
   }

   _setup_markup() {
       var self = this;
       super._setup_markup();
       this.canvas.css('cursor', window.uiControlHandler.getDefaultCursor());

       var htStyle = {
           fillColor: window.styles.hoverText.box.fill,
           textColor: window.styles.hoverText.text.color,
           strokeColor: window.styles.hoverText.box.stroke.color,
           lineWidth: window.styles.hoverText.box.stroke.lineWidth,
           lineDash: []
       };
       this.hoverTextElement = new HoverTextElement(this.entryID + '_hoverText', null, [0, 0.99], 'canvas',
           htStyle,
           5);
       this.viewport.addRenderElement(this.hoverTextElement);

       // interaction handlers
       if(!this.disableInteractions) {
           this.viewport.addCallback(this.entryID, 'mousedown', function(event) {
               self._canvas_mousedown(event);
           });
           this.viewport.addCallback(this.entryID, 'mousemove', function(event) {
               self._canvas_mousemove(event);
           });
           this.viewport.addCallback(this.entryID, 'mouseup', function(event) {
               self._canvas_mouseup(event);
           });
           this.viewport.addCallback(this.entryID, 'mouseleave', function(event) {
               self._canvas_mouseleave(event);
           });
       }
   }

   _toggleActive(event) {
       /*
           Sets boxes active or inactive as follows:
           - if a box encompasses the event's coordinates:
               - and is not active: it is turned active
               - and is active: nothing happens
               Any other box that is active _and_ contains the point stays active.
               Other boxes that are active and do not contain the point get deactivated,
               unless the shift key is held down.
           - if no box contains the coordinates, every single one is deactivated.
       */
       var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
       var minDist = 1e9;
       var argMin = null;
       for(var key in this.annotations) {
           var bbox = this.annotations[key].getRenderElement();
           if(bbox.containsPoint(coords)) {
               var dist = bbox.euclideanDistance(coords);
               if(dist < minDist) {
                   minDist = dist;
                   argMin = key;
               }
               if(!(event.shiftKey && this.annotations[key].isActive())) {
                   // deactivate
                   if(this.annotations[key].isVisible()) {
                       this.annotations[key].setActive(false, this.viewport);
                   }
               }
           } else if(!event.shiftKey) {
               // bbox outside and no shift key; deactivate
               if(this.annotations[key].isVisible()) {
                   this.annotations[key].setActive(false, this.viewport);
               }
           }
       }

       // handle closest
       if(argMin == null) {
           // deactivate all
           for(var key in this.annotations) {
               if(this.annotations[key].isVisible()) {
                   this.annotations[key].setActive(false, this.viewport);
               }
           }
       } else if(this.annotations[argMin].isVisible()) {
           // set closest one active
           this.annotations[argMin].setActive(true, this.viewport);
       }
   }

   _canvas_mouseout(event) {
       // clear hover text
       if(window.uiBlocked) return;
       this.hoverTextElement.setProperty('text', null);
       this.render();
   }

   _createAnnotation(event) {
       var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
       var props = {
           'x': coords[0],
           'y': coords[1],
           'width': 0,
           'height': 0,
           'label': window.labelClassHandler.getActiveClassID()
       };
       var anno = new Annotation(window.getRandomID(), props, 'boundingBoxes', 'annotation');
       this._addElement(anno);
       anno.getRenderElement().registerAsCallback(this.viewport);
       anno.getRenderElement().setActive(true, this.viewport);
       // manually fire mousedown event on annotation
       anno.getRenderElement()._mousedown_event(event, this.viewport);
   }

   _deleteActiveAnnotations(event) {
       var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
       var minDist = 1e9;
       var argMin = null;
       for(var key in this.annotations) {
           var bbox = this.annotations[key].getRenderElement();
           if(bbox.containsPoint(coords)) {
               if(this.annotations[key].isActive()) {
                   this.annotations[key].getRenderElement().deregisterAsCallback(this.viewport);
                   this._removeElement(this.annotations[key]);
                   return;
               }
               var dist = bbox.euclideanDistance(coords);
               if(dist < minDist) {
                   minDist = dist;
                   argMin = key;
               }
           }
       }

       if(argMin != null) {
           // no active annotation found, but clicked within another one; delete it
           this._removeElement(this.annotations[argMin]);
       }
       window.dataHandler.updatePresentClasses();
   }

   _drawCrosshairLines(coords, visible) {
       if(window.uiBlocked) return;
       if(this.crosshairLines == null && visible) {
           // create
           var vertLine = new LineElement(this.entryID + '_crosshairX', coords[0], 0, coords[0], window.defaultImage_h,
                               window.styles.crosshairLines,
                               false,
                               1);
           var horzLine = new LineElement(this.entryID + '_crosshairY', 0, coords[1], window.defaultImage_w, coords[1],
                               window.styles.crosshairLines,
                               false,
                               1);
           this.crosshairLines = new ElementGroup(this.entryID + '_crosshairLines', [vertLine, horzLine], 1);
           this.viewport.addRenderElement(this.crosshairLines);
           this.canvas.css('cursor', 'crosshair');

       } else {
           if(visible) {
               // update
               this.crosshairLines.elements[0].setProperty('startX', coords[0]);
               this.crosshairLines.elements[0].setProperty('endX', coords[0]);
               this.crosshairLines.elements[1].setProperty('startY', coords[1]);
               this.crosshairLines.elements[1].setProperty('endY', coords[1]);
               this.canvas.css('cursor', 'crosshair');
           } else {
               // remove
               this.viewport.removeRenderElement(this.crosshairLines);
               this.crosshairLines = null;
               this.canvas.css('cursor', window.uiControlHandler.getDefaultCursor());
           }
       }
   }

   _canvas_mousedown(event) {
       if(window.uiBlocked) return;
       this.mouseDown = true;

       // check functionality
       if(window.uiControlHandler.getAction() == ACTIONS.ADD_ANNOTATION) {
           // set all currently active boxes inactive
           for(var key in this.annotations) {
               if(this.annotations[key].isVisible()) {
                   this.annotations[key].setActive(false, this.viewport);
               }
           }

           // start creating a new bounding box
           this._createAnnotation(event);
       }
       this.render();
   }

   _canvas_mousemove(event) {
       if(window.uiBlocked) return;
       if(this.mouseDown) this.mouseDrag = true;

       var coords = this.viewport.getRelativeCoordinates(event, 'validArea');

       // update crosshair lines
       this._drawCrosshairLines(coords, window.uiControlHandler.getAction()==ACTIONS.ADD_ANNOTATION);

       // update hover text
       var hoverText = null;
       switch(window.uiControlHandler.getAction()) {
           case ACTIONS.ADD_ANNOTATION:
               hoverText = 'add new \"' + window.labelClassHandler.getActiveClassName() + '"';
               this.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getActiveColor());
               break;
           case ACTIONS.REMOVE_ANNOTATIONS:
               var numActive = 0;
               for(var key in this.annotations) {
                   if(this.annotations[key].isActive()) numActive++;
               }
               if(numActive>0) {
                   hoverText = 'Remove ' + numActive + ' annotation';
                   if(numActive>1) hoverText += 's';
                   this.hoverTextElement.setProperty('fillColor', window.styles.hoverText.box.fill);
               }
               break;
           case ACTIONS.DO_NOTHING:
               // find box over coordinates
               var anno = null;
               for(var key in this.annotations) {
                   if(this.annotations[key].geometry.isInDistance(coords, window.annotationProximityTolerance / Math.min(this.viewport.canvas.width(), this.viewport.canvas.height()))) {
                       anno = this.annotations[key];
                       break;
                   }
               }
               // set text
               if(anno === undefined || anno === null) {
                   hoverText = null;
                   break;
               }
               if(!this.mouseDrag && anno.isActive() && anno.label != window.labelClassHandler.getActiveClassID()) {
                   hoverText = 'move or change label to "' + window.labelClassHandler.getActiveClassName() + '"';
                   this.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getActiveColor());
               } else {
                   // show current class name of box
                   hoverText = window.labelClassHandler.getName(anno.label);
                   this.hoverTextElement.setProperty('fillColor', window.labelClassHandler.getColor(anno.label));
               }
               break;
       }
       // this.hoverTextElement.setProperty('position', coords);

       // flip text color if needed
       var htFill = this.hoverTextElement.getProperty('fillColor');
       if(htFill != null && window.getBrightness(htFill) >= 92) {
           this.hoverTextElement.setProperty('textColor', '#000000');
       } else {
           this.hoverTextElement.setProperty('textColor', '#FFFFFF');
       }

       this.hoverTextElement.setProperty('text', hoverText);

       this.render();
   }

   _canvas_mouseup(event) {
       if(window.uiBlocked) return;

       // check functionality
       if(window.uiControlHandler.getAction() == ACTIONS.ADD_ANNOTATION) {
           // new annotation completed
           //TODO: may fire before other rectangle's events, making them unwantedly active while finishing new rect
           // window.uiControlHandler.getAction() = ACTIONS.DO_NOTHING;

       } else if(window.uiControlHandler.getAction() == ACTIONS.REMOVE_ANNOTATIONS) {
           this._deleteActiveAnnotations(event);
           // window.uiControlHandler.getAction() = ACTIONS.DO_NOTHING;

       } else if(window.uiControlHandler.getAction() == ACTIONS.DO_NOTHING) {
           // update annotations to current label (if active and no dragging [resizing] was going on)
           if(!this.mouseDrag) {
               var coords = this.viewport.getRelativeCoordinates(event, 'validArea');
               for(var key in this.annotations) {
                   if(this.annotations[key].isActive()) {
                       if(event.shiftKey || this.annotations[key].getRenderElement().containsPoint(coords)) {
                           this.annotations[key].setProperty('label', window.labelClassHandler.getActiveClassID());
                       }
                   }
               }

               // activate or deactivate
               this._toggleActive(event);
           }
       }

       this.render();

       window.dataHandler.updatePresentClasses();

       this.mouseDrag = false;
       this.mouseDown = false;
   }

   _canvas_mouseleave(event) {
       if(window.uiBlocked) return;
       this.hoverTextElement.setProperty('text', null);
       this._drawCrosshairLines(null, false);
       this.render();
   }
}




class SemanticSegmentationEntry extends AbstractDataEntry {
    /*
        Implementation for segmentation maps.
    */
    constructor(entryID, properties) {
        super(entryID, properties);

        this.loadingPromise.then(response => {
            if(response) {
                // store natural image dimensions for annotations
                properties['width'] = this.imageEntry.image.naturalWidth;
                properties['height'] = this.imageEntry.image.naturalHeight;
                this._init_data(properties);
            }
        });
        
        this._setup_markup();
    }

   getAnnotationType() {
       return 'segmentationMap';
   }

   setAnnotationsVisible(visible) {
       super.setAnnotationsVisible(visible);
       this.annotation.setVisible(visible);
   }

   removeAllAnnotations() {
       this.segMap.clearAll();
       this.render();

       window.dataHandler.updatePresentClasses();
   }

   setLabel(label) {
       return;
   }

   _addElement(element) {
       // allow only one annotation for segmentation entry
       var key = element['annotationID'];
       if(element['type'] == 'annotation') {
           if(Object.keys(this.annotations).length > 0) {
               // replace current annotation
               var currentKey = Object.keys(this.annotations)[0];
               this.viewport.removeRenderElement(this.annotations[currentKey]);
               delete this.annotations[currentKey];
           }

           // add new item
           this.annotations[key] = element;
           this.viewport.addRenderElement(element.getRenderElement());            
       }
       // else if(element['type'] == 'prediction' && window.showPredictions) {
       //     this.predictions[key] = element;
       //     this.viewport.addRenderElement(element.getRenderElement());
       // }

       window.dataHandler.updatePresentClasses();
   }

   _init_data(properties) {
       var annoKeys = Object.keys(this.annotations);
       if(annoKeys.length) {
           this.annotation = this.annotations[annoKeys[0]]; 
       } else {
           this.annotation = new Annotation(window.getRandomID(), properties, 'segmentationMasks', 'annotation');
           this._addElement(this.annotation);
       }
       this.segMap = this.annotation.geometry;
       this.size = this.segMap.getSize();
   }

   _setup_markup() {
       var self = this;
       super._setup_markup();
       this.canvas.css('cursor', window.uiControlHandler.getDefaultCursor());

       // brush symbol
       this.brush = new PaintbrushElement(this.id+'_brush', null, null, 5);
       this.viewport.addRenderElement(this.brush);

       // interaction handlers
       if(!this.disableInteractions) {
           this.viewport.addCallback(this.entryID, 'mousedown', function(event) {
               self._canvas_mousedown(event);
           });
           this.viewport.addCallback(this.entryID, 'mousemove', function(event) {
               self._canvas_mousemove(event);
           });
           this.viewport.addCallback(this.entryID, 'mouseup', function(event) {
               self._canvas_mouseup(event);
           });
           this.viewport.addCallback(this.entryID, 'mouseleave', function(event) {
               self._canvas_mouseleave(event);
           });
       }
   }

   _set_default_brush() {
       this.viewport.removeRenderElement(this.brush);
       this.brush = this.brushes[window.uiControlHandler.getBrushType()];
       this.viewport.addRenderElement(this.brush);
   }


   __paint(event) {
       if([ACTIONS.ADD_ANNOTATION, ACTIONS.REMOVE_ANNOTATIONS].includes(window.uiControlHandler.getAction())) {

           // update mouse position
           this.mousePos = this.viewport.getRelativeCoordinates(event, 'validArea');
           var mousePos_abs = [
               this.mousePos[0] * this.size[0],
               this.mousePos[1] * this.size[1]
           ];

            // show brush
            this.brush.setProperty('x', this.mousePos[0]);
            this.brush.setProperty('y', this.mousePos[1]);

            // paint with brush at current position
            if(this.mouseDown) {
                var canvasScale = [
                    this.viewport.canvas.width() / this.imageEntry.image.naturalWidth,
                    this.viewport.canvas.height() / this.imageEntry.image.naturalHeight
                ];
                var brushSize = [
                    window.uiControlHandler.segmentation_properties.brushSize / canvasScale[1],
                    window.uiControlHandler.segmentation_properties.brushSize / canvasScale[0]
                ];
                if(window.uiControlHandler.getAction() === ACTIONS.REMOVE_ANNOTATIONS || event.altKey) {
                    this.segMap.clear(mousePos_abs,
                        window.uiControlHandler.segmentation_properties.brushType,
                        brushSize);
                } else {
                    this.segMap.paint(mousePos_abs,
                        window.labelClassHandler.getActiveColor(),
                        window.uiControlHandler.segmentation_properties.brushType,
                        brushSize);
                }
            }

       } else {
           // hide brush
           this.brush.setProperty('x', null);
       }
       this.render();
   }


   // callbacks
   _canvas_mousedown(event) {
       if(window.uiBlocked) return;
       this.mouseDown = true;
       this.__paint(event);
   }

   _canvas_mousemove(event) {
       if(window.uiBlocked) return;
       this.__paint(event);
   }

   _canvas_mouseup(event) {
       this.mouseDown = false;
   }

   _canvas_mouseleave(event) {
       this.mouseDown = false;
   }
}