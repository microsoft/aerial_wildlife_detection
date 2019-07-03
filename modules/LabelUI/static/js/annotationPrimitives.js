class Annotation {

    constructor(annotationID, properties, type) {
        this.annotationID = annotationID;
        this.type = type;
        this._parse_properties(properties);
    }

    _parse_properties(properties) {
        this.label = properties['label'];
        if(!window.enableEmptyClass && this.label == null) {
            // no empty class allowed; assign selected label
            this.label = window.labelClassHandler.getActiveClassID();
        }
        this.confidence = properties['confidence'];
        var lineWidth = 4;      //TODO
        var pointSize = 8;
        if(this.type == 'userAnnotation') {
            lineWidth = 8;
            pointSize = 10;
        } else if(this.type == 'annotation') {
            lineWidth = 6;
            pointSize = 12;
        }
        var lineDash = [];
        if(this.type == 'prediction') {
            lineDash = [4, 4];      //TODO
        }
        if('segMapFileName' in properties) {
            // Semantic segmentation map
            throw Error('Segmentation maps not yet implemented');

        } else if('coordinates' in properties) {
            // Polygon
            throw Error('Polygons not yet implemented');

        } else if('width' in properties) {
            // Bounding Box
            this.geometry = new RectangleElement(
                this.annotationID + '_geom',
                properties['x'], properties['y'],
                properties['width'], properties['height'],
                null,
                window.labelClassHandler.getColor(this.label),
                lineWidth,
                lineDash);

        } else if('x' in properties) {
            // Point
            this.geometry = new PointElement(
                this.annotationID + '_geom',
                properties['x'], properties['y'],
                window.labelClassHandler.getColor(this.label),
                pointSize
            );
        } else {
            // Classification label
            var borderText = window.labelClassHandler.getName(this.label);
            if(this.confidence != null) {
                borderText += ' (' + 100*this.confidence + '%)';        //TODO: round to two decimals
            }
            this.geometry = new BorderStrokeElement(
                this.annotationID + '_geom',
                window.labelClassHandler.getColor(this.label),
                2*lineWidth,
                lineDash,
                borderText
            )
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
        return {
            'id' : this.annotationID,
            'type' : this.type,
            'label' : this.label,
            'confidence' : this.confidence,
            'geometry' : this.geometry.getGeometry()
        };
    }

    setProperty(propertyName, value) {
        if(this.hasOwnProperty(propertyName)) {
            this[propertyName] = value;
        }
        if(propertyName == 'label') {
            this.geometry.setProperty('color', window.labelClassHandler.getColor(value));
            if(this.geometry instanceof BorderStrokeElement) {
                // show label text
                this.geometry.setProperty('text', window.labelClassHandler.getActiveClassName());
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
        this.geometry.setVisible(visible);
    }
}