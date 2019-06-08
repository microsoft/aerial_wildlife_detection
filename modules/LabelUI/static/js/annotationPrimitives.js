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
        context.strokeRect(lineWidth/2, lineWidth/2, wh[0]-lineWidth/2, wh[1]-lineWidth/2);
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

    draw(context, canvas, scaleFun) {
        context.fillStyle = window.labelClassHandler.getColor(this.label, '#000000');
        var radius = 5;
        if(this.type == 'userAnnotation') {
            radius = 15;
        } else if(this.type == 'annotation') {
            radius = 10;
        }
        var centerCoords = scaleFun([this.x, this.y]);
        context.arc(centerCoords[0], centerCoords[1], radius, 0, 2*Math.PI);
        context.fill();

        // // adjust to canvas size on screen
        // var scaleX = window.defaultImage_w / canvas.width();
        // var scaleY = window.defaultImage_h / canvas.height();

        // //TODO: shape, size, etc.
        // var size = 20;
        
        // context.lineWidth = (this.type == 'annotation'? 8 : 4);
        // context.beginPath();
        // context.moveTo(scaleX * (this.x - size/2), scaleY * this.y);
        // context.lineTo(scaleX * (this.x + size/2), scaleY * this.y);
        // context.moveTo(scaleX * this.x, scaleY * (this.y - size/2));
        // context.lineTo(scaleX * this.x, scaleY * (this.y + size/2));
        // context.stroke();
    }

    euclideanDistance(that) {
        return Math.sqrt(Math.pow(this.x - that.x,2) + Math.pow(this.y - that.y,2));
    }

    euclideanDistance(thatX, thatY) {
        return Math.sqrt(Math.pow(this.x - thatX,2) + Math.pow(this.y - thatY,2));
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
       this.w = w;
       this.h = h;
   }

   getProperties() {
       var props = super.getProperties();
       props['w'] = this.w;
       props['h'] = this.h;
       return props;
   }

   draw(context, canvas, scaleFun) {
        //TODO: outline width
        context.strokeStyle = window.labelClassHandler.getColor(this.label, '#000000');
        context.lineWidth = (this.type == 'annotation'? 8 : 4);
        var wh = scaleFun([this.w, this.h]);
        var center = scaleFun([this.x, this.y])
        context.strokeRect(center[0] - wh[0]/2, center[1] - wh[1]/2, wh[0], wh[1]);
    }
}