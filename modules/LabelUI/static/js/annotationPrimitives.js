class Annotation {

    constructor(annotationID, properties, geometryType, type, autoConverted) {
        this.annotationID = annotationID;
        this.geometryType = geometryType;
        this.type = type;
        this.autoConverted = autoConverted;
        this._parse_properties(properties);
    }

    _parse_properties(properties) {
        this.loadingPromise = Promise.resolve();
        if(properties.hasOwnProperty('label')) {
            this.label = properties['label'];
        } else {
            this.label = null;
        }
        
        var unsure = false;
        if(properties.hasOwnProperty('unsure')) {
            var unsure = (properties['unsure'] == null || properties['unsure'] == undefined ? false : properties['unsure']);    //TODO: should be property of "Annotation", but for drawing reasons we assign it to the geometry...
        }
        if(!window.enableEmptyClass && this.label == null) {
            // no empty class allowed; assign selected label
            this.label = window.labelClassHandler.getActiveClassID();
        }

        if(properties.hasOwnProperty('confidence')) {
            this.confidence = properties['confidence'];
        } else {
            this.confidence = null;
        }

        // drawing styles
        var color = window.labelClassHandler.getColor(this.label);
        var style = JSON.parse(JSON.stringify(window.styles.annotations));  // copy default style
        if(this.type === 'prediction') {
            style = JSON.parse(JSON.stringify(window.styles.predictions));
        } else if(this.autoConverted) {
            // special style for annotations converted from predictions
            style = JSON.parse(JSON.stringify(window.styles.annotations_converted));
        }
        style['strokeColor'] = window.addAlpha(color, style.lineOpacity);
        style['fillColor'] = window.addAlpha(color, style.fillOpacity);

        if(this.geometryType === 'segmentationMasks') {
            // Semantic segmentation map
            this.geometry = new SegmentationElement(
                this.annotationID + '_geom',
                properties['segmentationmask'],
                properties['segmentationmask_predicted'],
                properties['width'],
                properties['height']
            );

        } else if(this.geometryType === 'polygons') {
            // Polygon
            if(properties['magnetic_polygon']) {
                let self = this;
                this.loadingPromise = properties['edge_map'].then((edgeMap) => {
                    self.geometry = new MagneticPolygonElement(
                        self.annotationID + '_geom',
                        edgeMap,
                        properties['coordinates'],
                        style,
                        unsure
                    );
                });
            } else {
                this.geometry = new PolygonElement(
                    this.annotationID + '_geom',
                    properties['coordinates'],
                    style,
                    unsure
                );
            }

        } else if(this.geometryType === 'boundingBoxes') {
            // Bounding Box
            this.geometry = new RectangleElement(
                this.annotationID + '_geom',
                properties['x'], properties['y'],
                properties['width'], properties['height'],
                style,
                unsure
            );

        } else if(this.geometryType === 'points') {
            // Point
            this.geometry = new PointElement(
                this.annotationID + '_geom',
                properties['x'], properties['y'],
                style,
                unsure
            );

        } else if(this.geometryType === 'labels') {
            // Classification label
            var borderText = window.labelClassHandler.getName(this.label);
            if(this.confidence != null) {
                borderText += ' (' + 100*this.confidence + '%)';        //TODO: round to two decimals
            }
            this.geometry = new BorderStrokeElement(
                this.annotationID + '_geom',
                borderText,
                style,
                unsure
            );
            
        } else {
            throw Error('Unknown geometry type (' + this.geometryType + ').')
        }
    }

    isValid() {
        return this.geometry.isValid;
    }

    isActive() {
        return this.geometry.isActive;
    }

    setActive(active, viewport) {
        this.geometry.setActive(active, viewport);
    }

    getChanged() {
        // returns true if the user has modified the annotation
        return this.geometry.changed;
    }

    getTimeChanged() {
        return this.geometry.getLastUpdated();
    }

    getProperties() {
        let geometry = this.geometry.getGeometry();
        if(geometry === null) return null;      // only return item if geometry is valid
        return {
            'id' : this.annotationID,
            'type' : (this.type.includes('annotation') ? 'annotation' : 'prediction'),
            'label' : this.label,
            'confidence' : this.confidence,
            'autoConverted': (this.autoConverted === null || this.autoConverted === undefined ? false : this.autoConverted),
            'geometry' : geometry
        };
    }

    getProperty(propertyName) {
        if(this.hasOwnProperty(propertyName)) {
            return this[propertyName];
        }
        return this.geometry.getProperty(propertyName);
    }

    setProperty(propertyName, value) {
        if(this.hasOwnProperty(propertyName)) {
            this[propertyName] = value;
        }
        if(propertyName == 'label') {
            if(this.geometry instanceof BorderStrokeElement) {
                // show label text
                if(value == null) {
                    this.geometry.setProperty('color', null);
                    this.geometry.setProperty('text', null);
                } else {
                    this.geometry.setProperty('color', window.labelClassHandler.getColor(value));
                    this.geometry.setProperty('text', window.labelClassHandler.getName(value));
                }
            } else {
                this.geometry.setProperty('color', window.labelClassHandler.getColor(value));
            }
        } else if(this.geometry.hasOwnProperty(propertyName)) {
            this.geometry.setProperty(propertyName, value);
        }
    }

    getRenderElement() {
        return this.geometry;
    }

    isVisible() {
        return this.geometry.visible;
    }

    setVisible(visible) {
        if(typeof(visible) === 'number' && !isNaN(this.confidence)) {
            // confidence-based
            this.geometry.setVisible(this.confidence >= visible);
        } else {
            // boolean
            this.geometry.setVisible(visible);
        }
    }

    styleChanged() {
        /**
         * To be called when the global window.styles object
         * has been changed. Updates the new styles to all
         * entries and re-renders them.
         */
        var color = window.labelClassHandler.getColor(this.label);
        var style = JSON.parse(JSON.stringify(window.styles.annotations));  // copy default style
        if(this.type === 'prediction') {
            style = JSON.parse(JSON.stringify(window.styles.predictions));
        } else if(this.autoConverted) {
            // special style for annotations converted from predictions
            style = JSON.parse(JSON.stringify(window.styles.annotations_converted));
        }
        style['strokeColor'] = window.addAlpha(color, style.lineOpacity);
        style['fillColor'] = window.addAlpha(color, style.fillOpacity);

        this.geometry.setProperty('style', style);
    }
}