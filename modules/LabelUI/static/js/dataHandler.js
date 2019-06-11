/*
    Maintains the data entries currently on display.

    2019 Benjamin Kellenberger
*/

class DataHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.dataEntries = {};
    }

    loadNextBatch() {
        var self = this;

        // clear current entries
        this.parentDiv.empty();
        this.dataEntries = [];

        $.getJSON('getLatestImages?limit=1', function(data) {
            for(var d in data) {
                // create new data entry
                switch(String(window.annotationType)) {
                    case 'labels':
                        var entry = new ClassificationEntry(d, data[d]);
                        break;
                    case 'points':
                        var entry = new PointAnnotationEntry(d, data[d]);
                        break;
                    case 'boundingBoxes':
                        var entry = new BoundingBoxAnnotationEntry(d, data[d]);
                        break;
                    default:
                        break;
                }

                // append
                self.parentDiv.append(entry.markup);
                self.dataEntries.push(entry);
            }
        });
    }


    _entriesToJSON(minimal) {
        var entries = {};
        for(var e=0; e<this.dataEntries.length; e++) {
            entries[this.dataEntries[e].entryID] = this.dataEntries[e].getProperties(minimal);
        }

        return JSON.stringify({
            'entries': entries
        })
    }


    submitAnnotations() {
        var self = this;
        var entries = this._entriesToJSON(true);
        $.post('submitAnnotations', entries, function(response) {
            console.log(response);

            // next batch
            self.loadNextBatch();
        })
    }
}