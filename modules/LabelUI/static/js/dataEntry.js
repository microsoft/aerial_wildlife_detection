/*
    Definition of a data entry, as shown on a grid on the screen.

    2019 Benjamin Kellenberger
 */

 class AbstractDataEntry {
     /*
        Abstract base class for data entries.
     */
    constructor(entryID, fileName) {
        this.entryID = entryID;
        this.fileName = fileName;
        this._cache_image()
    }

    _cache_image() {
        var self = this;
        this.image = new Image();
        this.image.width = '400'
        this.image.height = '300'
        this.image.src = window.dataServerURI + this.fileName;
    }
 }




 class ClassificationEntry extends AbstractDataEntry {
     /*
        Implementation for image classification.
        Inputs:
        - entryID: identifier for the data entry
        - fileName: name of the image file to retrieve from data server
        - predictedLabel: optional, array of floats for model predictions

        If predictedLogits is provided, a thin border tinted according to the
        arg max (i.e., the predicted label) is drawn around the image.

        As soon as the user clicks into the image, a thick border is drawn,
        colored w.r.t. the user-selected class. A second click removes the user
        label again.
     */
    constructor(entryID, fileName, predictedLabel, predictedConfidence) {
        super(entryID, fileName);
        this.predictedLabel = predictedLabel;
        this.predictedConfidence = predictedConfidence;
        this.userLabel = null;

        this._setup_markup();
    }

    _setup_markup() {
        var self = this;
        this.markup = $('<div class="entry"></div>');

        // click handler
        this.markup.click(function() {
            self.toggleUserLabel()
        });

        // image
        this.markup.append(this.image);
    }

    _set_border_style() {
        // specify border decoration
        var style = 'none';
        if(this.userLabel!=null) {
            style = '4px solid ' + window.classColors[this.userLabel];
        } else if(this.predictedLabel!=null) {
            style = '2px solid ' + window.classColors[this.predictedLabel];
        }
        this.markup.css('border', style);
    }

    toggleUserLabel() {
        if(this.userLabel!=null) {
            this.userLabel = null;
        } else {
            //TODO: get label from current setting instead
            this.userLabel = Math.floor(Math.random() * window.classColors.length);
        }
        this._set_border_style();
    }
 }