class Annotation {

    constructor(annotationID, properties, type) {
        this.annotationID = annotationID;
        this.type = type;
        this._parse_properties(properties);
    }

    _parse_properties(properties) {
        this.label = properties['label'];
        this.confidence = properties['confidence'];
        var lineWidth = 4;      //TODO
        if(this.type == 'userAnnotation') {
            lineWidth = 8;
        } else if(this.type == 'annotation') {
            lineWidth = 6;
        }
        if('segMapFileName' in properties) {
            // Semantic segmentation map
            throw Error('Segmentation maps not yet implemented');

        } else if('coordinates' in properties) {
            // Polygon
            //TODO
        } else if('width' in properties) {
            // Bounding Box
            this.geometry = new RectangleElement(
                this.annotationID + '_geom',
                properties['x'], properties['y'],
                properties['width'], properties['height'],
                null,
                window.labelClassHandler.getColor(this.label),
                lineWidth);
        } else if('x' in properties) {
            // Point
            this.geometry = new PointElement(
                this.annotationID + '_geom',
                properties['x'], properties['y'],
                window.labelClassHandler.getColor(this.label)
            );
        } else {
            // Classification label
            this.geometry = new BorderStrokeElement(
                this.annotationID + '_geom',
                window.labelClassHandler.getColor(this.label),
                2*lineWidth,
                []
            )
        }
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

    getProperties() {
        return {
            'annotationID' : this.annotationID,
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
        } else if(this.geometry.hasOwnProperty(propertyName)) {
            this.geometry.setProperty(propertyName, value);
        }
    }

    getAnnotationType() {
        return this.geometry.getType();     //TODO: implemented only for backwards compatibility
    }

    getRenderElement() {
        return this.geometry;               //TODO: ditto (?)
    }
}



// class AbstractAnnotation {

//     constructor(annotationID, type) {
//         this.annotationID = annotationID;
//         this.type = type;
//     }

//     setProperty(propertyName, value) {
//         if(this.hasOwnProperty(propertyName)) {
//             this[propertyName] = value;
//         }
        
//         if(propertyName == 'label') {
//             this.renderElement.setProperty('color', window.labelClassHandler.getColor(value));
//         } else if(this.renderElement.hasOwnProperty(propertyName)) {
//             this.renderElement.setProperty(propertyName, value);
//         }
//     }

//     getProperties() {
//         return {'annotationID':this.annotationID};
//     }

//     getAnnotationType() {
//         throw Error('Not implemented.');
//     }

//     getRenderElement() {
//         return this.renderElement;
//     }
// }



// class LabelAnnotation extends AbstractAnnotation {
//     /*
//         Labeling annotation.
//     */
//     constructor(annotationID, properties, type) {
//         super(annotationID, type);
//         this.label = properties['label'];
//         this.confidence = properties['confidence'];
//         this._init_render_element();
//     }

//     _init_render_element() {
//         var lineWidth = 4;
//         if(this.type == 'userAnnotation') {
//             lineWidth = 8;
//         } else if(this.type == 'annotation') {
//             lineWidth = 6;
//         }
//         this.renderElement = new BorderStrokeElement(window.labelClassHandler.getColor(this.label),
//             lineWidth);
//     }

//     getProperties() {
//         var props = super.getProperties();
//         props['label'] = this.label;
//         if(this.confidence != null) {
//             props['confidence'] = this.confidence;
//         }
//         return props;
//     }

//     getAnnotationType() {
//         return 'label';
//     }
// }



// class PointAnnotation extends LabelAnnotation {
//     /*
//         Point primitive.
//     */
//     constructor(annotationID, properties, type) {
//         super(annotationID, properties, type);
//         this.x = properties['x'];
//         this.y = properties['y'];
//         this._init_render_element();
//     }

//     _init_render_element() {
//         var radius = 5;
//         if(this.type == 'userAnnotation') {
//             radius = 15;
//         } else if(this.type == 'annotation') {
//             radius = 10;
//         }
//         this.renderElement = new PointElement(this.x, this.y,
//             window.labelClassHandler.getColor(this.label),
//             radius);
//     }

//     getProperties() {
//         var props = super.getProperties();
//         props['x'] = this.x;
//         props['y'] = this.y;
//         return props;
//     }

//     getAnnotationType() {
//         return 'point';
//     }
// }


// class BoundingBoxAnnotation extends PointAnnotation {
//     /*
//         Bounding Box primitive.
//         Bounding boxes are defined as [X,Y,W,H], with X and Y denoting
//         the center of the box.
//         As such, they are a natural extension of the Point primitive.
//     */
//     constructor(annotationID, properties, type) {
//         super(annotationID, properties, type);
//         this.width = properties['width'];
//         this.height = properties['height'];
//         this._init_render_element();
//     }

//     _init_render_element() {
//         var lineWidth = (this.type == 'annotation'? 4 : 2);
//         this.renderElement = new RectangleElement(this.x, this.y,
//             this.width, this.height,
//             null,
//             window.labelClassHandler.getColor(this.label),
//             lineWidth);
//     }

//     getProperties() {
//         var props = super.getProperties();
//         props['width'] = this.width;
//         props['height'] = this.height;
//         return props;
//     }

//     setProperty(propertyName, value) {
//         super.setProperty(propertyName, value);
//         this.width = Math.max(window.minObjSize, this.width);
//         this.height = Math.max(window.minObjSize, this.height);

//         // carry over to resize handles
//         this._updateResizeHandles();
//     }

//     getAnnotationType() {
//         return 'boundingBox';
//     }

//     getExtent() {
//         return [this.x - this.width/2, this.y - this.height/2, this.x + this.width/2, this.y + this.height/2];
//     }

//     isInDistance(coordinates, tolerance, forceCorner) {
//         /*
//             Returns true if any parts of the bounding box are
//             within a tolerance's distance of the provided coordinates.
//             If 'forceCorner' is true, coordinates have to be within
//             reach of one of the bounding box's corners.
//         */
//         if(forceCorner) {
//             return (this.getClosestHandle(coordinates, tolerance) != null);

//         } else {
//             var extentsTolerance = [this.x-this.width/2, this.y-this.height/2, this.x+this.width/2, this.y+this.height/2];
//             return (coordinates[0] >= extentsTolerance[0] && coordinates[0] <= extentsTolerance[2]) &&
//                 (coordinates[1] >= extentsTolerance[1] && coordinates[1] <= extentsTolerance[3]);
//         }
//     }

//     containsPoint(coordinates) {
//         var extent = this.getExtent();
//         return (coordinates[0] >= extent[0] && coordinates[0] <= extent[2]) &&
//             (coordinates[1] >= extent[1] && coordinates[1] <= extent[3]);
//     }

//     getClosestHandle(coordinates, tolerance) {
//         /*
//             Returns one of {'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'} if the coordinates
//             are close to one of the adjustment handles within a given tolerance.
//             Returns 'c' if coordinates are not close to handle, but within bounding box.
//             Else returns null.
//         */
//         var matchL = Math.abs((this.x - this.width/2) - coordinates[0]) <= tolerance;
//         var matchT = Math.abs((this.y - this.height/2) - coordinates[1]) <= tolerance;
//         var matchR = Math.abs((this.x + this.width/2) - coordinates[0]) <= tolerance;
//         var matchB = Math.abs((this.y + this.height/2) - coordinates[1]) <= tolerance;

//         if(matchT) {
//             if(matchL) return 'nw';
//             if(matchR) return 'ne';
//             return 'n';
//         } else if(matchB) {
//             if(matchL) return 'sw';
//             if(matchR) return 'se';
//             return 's';
//         } else if(matchL) {
//             return 'w';
//         } else if(matchR) {
//             return 'e';
//         } else if(this.containsPoint(coordinates)) {
//             return 'c';
//         } else {
//             return null;
//         }
//     }

//     getResizeHandles(cornersOnly, zIndex) {
//         /*
//             Returns small drawable rectangles at the corner coordinates
//             of this rectangle that look like resize handles.
//         */
//         if(this.resizeHandles != null) {
//             return this.resizeHandles;
//         }

//         var drawingProperties = window.styles.resizeHandles;
//         var getHandle = function(x, y) {
//             return new RectangleElement(x, y,
//                 drawingProperties.size, drawingProperties.size,
//                 drawingProperties.fillColor,
//                 drawingProperties.strokeColor,
//                 drawingProperties.lineWidth,
//                 null,
//                 zIndex);
//         }
//         var handles = [];

//         // corners
//         handles.push(getHandle(this.x - this.width/2, this.y - this.height/2));
//         handles.push(getHandle(this.x - this.width/2, this.y + this.height/2));
//         handles.push(getHandle(this.x + this.width/2, this.y - this.height/2));
//         handles.push(getHandle(this.x + this.width/2, this.y + this.height/2));

//         if(!cornersOnly) {
//             // sides
//             handles.push(getHandle(this.x, this.y - this.height/2));
//             handles.push(getHandle(this.x, this.y + this.height/2));
//             handles.push(getHandle(this.x - this.width/2, this.y));
//             handles.push(getHandle(this.x + this.width/2, this.y));
//         }
//         this.resizeHandles = new ElementGroup(handles);
//         return this.resizeHandles;
//     }

//     _updateResizeHandles() {
//         /*
//             Re-positions the resize handles (if available)
//             to match the bounding box position.
//         */
//         if(this.resizeHandles == null) return;

//         this.resizeHandles.elements[0].setProperty('x', this.x - this.width/2);
//         this.resizeHandles.elements[0].setProperty('y', this.y - this.height/2);

//         this.resizeHandles.elements[1].setProperty('x', this.x - this.width/2);
//         this.resizeHandles.elements[1].setProperty('y', this.y + this.height/2);

//         this.resizeHandles.elements[2].setProperty('x', this.x + this.width/2);
//         this.resizeHandles.elements[2].setProperty('y', this.y - this.height/2);

//         this.resizeHandles.elements[3].setProperty('x', this.x + this.width/2);
//         this.resizeHandles.elements[3].setProperty('y', this.y + this.height/2);

//         if(this.resizeHandles.elements.length > 4) {
//             this.resizeHandles.elements[4].setProperty('x', this.x);
//             this.resizeHandles.elements[4].setProperty('y', this.y - this.height/2);

//             this.resizeHandles.elements[5].setProperty('x', this.x);
//             this.resizeHandles.elements[5].setProperty('y', this.y + this.height/2);

//             this.resizeHandles.elements[6].setProperty('x', this.x - this.width/2);
//             this.resizeHandles.elements[6].setProperty('y', this.y);

//             this.resizeHandles.elements[7].setProperty('x', this.x + this.width/2);
//             this.resizeHandles.elements[7].setProperty('y', this.y);
//         }
//     }
// }