class AbstractAnnotation {
    constructor(annotationID) {
        this.annotationID = annotationID;
    }

    getProperties() {
        return {'annotationID':this.annotationID};
    }

    draw(context) {
        return;
    }
}


class PointAnnotation extends AbstractAnnotation {
    /*
        Point primitive.
    */
    constructor(annotationID, properties) {
        super(annotationID);
        this.x = properties['x'];
        this.y = properties['y'];
        this.predictedLabel = properties['predictedLabel'];
        this.predictedConfidence = properties['predictedConfidence'];
        this.userLabel = properties['userLabel'];
    }

    getProperties() {
        var props = super.getProperties();
        props['x'] = this.x;
        props['y'] = this.y;
        props['predictedLabel'] = this.predictedLabel;
        props['predictedConfidence'] = this.predictedConfidence;
        props['userLabel'] = this.userLabel;
        return props;
    }

    draw(context, canvas) {
        // adjust to canvas size on screen
        var scaleX = window.defaultImage_w / canvas.width();
        var scaleY = window.defaultImage_h / canvas.height();

        //TODO: shape, size, etc.
        var size = 20;
        var label = (this.userLabel != null? this.userLabel : this.predictedLabel);
        context.strokeStyle = window.labelClassHandler.getColor(label, '#000000');
        context.lineWidth = (this.userLabel != null? 8 : 4);
        context.beginPath();
        context.moveTo(scaleX * (this.x - size/2), scaleY * this.y);
        context.lineTo(scaleX * (this.x + size/2), scaleY * this.y);
        context.moveTo(scaleX * this.x, scaleY * (this.y - size/2));
        context.lineTo(scaleX * this.x, scaleY * (this.y + size/2));
        context.stroke();
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
   constructor(annotationID, properties) {
       super(annotationID, properties);
       this.w = w;
       this.h = h;
   }

   getProperties() {
       var props = super.getProperties();
       props['w'] = this.w;
       props['h'] = this.h;
       return props;
   }

   draw(context) {
        //TODO: outline width
        var label = (this.userLabel != null? this.userLabel : this.predictedLabel);
        context.strokeStyle = window.labelClassHandler.getColor(label, '#000000');
        context.lineWidth = (this.userLabel != null? 8 : 4);
        context.strokeRect(this.x - this.w/2, this.y - this.h/2, this.w, this.h);
    }
}