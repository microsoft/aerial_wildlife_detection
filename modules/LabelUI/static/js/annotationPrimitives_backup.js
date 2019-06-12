window.parseAnnotation = function(annotationID, properties, type) {
    /*
        Reads the properties object and automatically initializes
        and returns the appropriate annotation primitive based on the contents.
    */
   if('w' in properties) {
       // width and height: bounding box
       return new BoundingBoxAnnotation(annotationID, properties, type);

   } else if('x' in properties) {
        // no width and height, but coordinates: point
        return new PointAnnotation(annotationID, properties, type);

   } else if('segMapFileName' in properties) {
       //TODO: segmentation map
       throw Error('Segmentation maps not yet implemented');

   } else {
       // classification entry
       return new LabelAnnotation(annotationID, properties, type);
   }
};


class AbstractAnnotation {

    constructor(annotationID, type) {
        this.annotationID = annotationID;
        this.type = type; 
    }

    getProperties() {
        return {'annotationID':this.annotationID};
    }

    getAnnotationType() {
        throw Error('Not implemented.');
    }

    draw(context, canvas, scaleFun) {
        return;
    }
}



class LabelAnnotation extends AbstractAnnotation {
    /*
        Labeling annotation.
    */
    constructor(annotationID, properties, type) {
        super(annotationID, type);
        this.label = properties['label'];
        this.confidence = properties['confidence'];
    }

    getProperties() {
        var props = super.getProperties();
        props['label'] = this.label;
        if(this.confidence != null) {
            props['confidence'] = this.confidence;
        }
        return props;
    }

    getAnnotationType() {
        return 'label';
    }

    draw(context, canvas, scaleFun) {
        // draw rectangle border around canvas if label exists
        if(this.label == null) return;

        context.strokeStyle = window.classes[this.label]['color'];
        var lineWidth = 4;
        if(this.type == 'userAnnotation') {
            lineWidth = 8;
        } else if(this.type == 'annotation') {
            lineWidth = 6;
        }
        context.lineWidth = lineWidth;
        var wh = scaleFun([canvas.width(), canvas.height()]);
        context.strokeRect(lineWidth/2, lineWidth/2, wh[0]-lineWidth, wh[1]-lineWidth);
    }
}



class PointAnnotation extends LabelAnnotation {
    /*
        Point primitive.
    */
    constructor(annotationID, properties, type) {
        super(annotationID, properties, type);
        this.x = properties['x'];
        this.y = properties['y'];
    }

    getProperties() {
        var props = super.getProperties();
        props['x'] = this.x;
        props['y'] = this.y;
        return props;
    }

    getAnnotationType() {
        return 'point';
    }

    draw(context, canvas, scaleFun) {
        if(this.label == null) return;
        context.fillStyle = window.labelClassHandler.getColor(this.label, '#000000');
        var radius = 5;
        if(this.type == 'userAnnotation') {
            radius = 15;
        } else if(this.type == 'annotation') {
            radius = 10;
        }
        var centerCoords = scaleFun([this.x, this.y]);
        context.beginPath();
        context.arc(centerCoords[0], centerCoords[1], radius, 0, 2*Math.PI);
        context.fill();
        context.closePath();
    }

    euclideanDistance(that) {
        return Math.sqrt(Math.pow(this.x - that[0],2) + Math.pow(this.y - that[1],2));
    }
}


class BoundingBoxAnnotation extends PointAnnotation {
    /*
        Bounding Box primitive.
        Bounding boxes are defined as [X,Y,W,H], with X and Y denoting
        the center of the box.
        As such, they are a natural extension of the Point primitive.
    */
    constructor(annotationID, properties, type) {
        super(annotationID, properties, type);
        this.w = properties['w'];
        this.h = properties['h'];
    }

    getProperties() {
        var props = super.getProperties();
        props['w'] = this.w;
        props['h'] = this.h;
        return props;
    }

    getAnnotationType() {
        return 'boundingBox';
    }

    draw(context, canvas, scaleFun) {
        //TODO: outline width
        context.strokeStyle = window.labelClassHandler.getColor(this.label, '#000000');
        context.lineWidth = (this.type == 'annotation'? 8 : 4);
        var wh = scaleFun([this.w, this.h]);
        var center = scaleFun([this.x, this.y]);
        context.beginPath();
        context.strokeRect(center[0] - wh[0]/2, center[1] - wh[1]/2, wh[0], wh[1]);
        context.closePath();
    }

    getExtent() {
        return [this.x - this.w/2, this.y - this.h/2, this.x + this.w/2, this.y + this.h/2];
    }

    isInDistance(coordinates, tolerance, forceCorner) {
        /*
            Returns true if any parts of the bounding box are
            within a tolerance's distance of the provided coordinates.
            If 'forceCorner' is true, coordinates have to be within
            reach of one of the bounding box's corners.
        */
        if(forceCorner) {
            return (this.getClosestHandle(coordinates, tolerance) != null);

        } else {
            var extentsTolerance = [this.x-this.w/2, this.y-this.h/2, this.x+this.w/2, this.y+this.h/2];
            return (coordinates[0] >= extentsTolerance[0] && coordinates[0] <= extentsTolerance[2]) &&
                (coordinates[1] >= extentsTolerance[1] && coordinates[1] <= extentsTolerance[3]);
        }
    }

    containsPoint(coordinates) {
        var extent = this.getExtent();
        return (coordinates[0] >= extent[0] && coordinates[0] <= extent[2]) &&
            (coordinates[1] >= extent[1] && coordinates[1] <= extent[3]);
    }

    getClosestHandle(coordinates, tolerance) {
        /*
            Returns one of {'nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'} if the coordinates
            are close to one of the adjustment handles within a given tolerance.
            Returns 'c' if coordinates are not close to handle, but within bounding box.
            Else returns null.
        */
        var matchL = Math.abs((this.x - this.w/2) - coordinates[0]) <= tolerance;
        var matchT = Math.abs((this.y - this.h/2) - coordinates[1]) <= tolerance;
        var matchR = Math.abs((this.x + this.w/2) - coordinates[0]) <= tolerance;
        var matchB = Math.abs((this.y + this.h/2) - coordinates[1]) <= tolerance;

        if(matchT) {
            if(matchL) return 'nw';
            if(matchR) return 'ne';
            return 'n';
        } else if(matchB) {
            if(matchL) return 'sw';
            if(matchR) return 'se';
            return 's';
        } else if(matchL) {
            return 'w';
        } else if(matchR) {
            return 'e';
        } else if(this.containsPoint(coordinates)) {
            return 'c';
        } else {
            return null;
        }
    }
}