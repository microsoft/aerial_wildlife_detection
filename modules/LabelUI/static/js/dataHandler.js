/*
    Maintains the data entries currently on display.

    2019 Benjamin Kellenberger
*/

class DataHandler {

    constructor(parentDiv) {
        this.parentDiv = parentDiv;
        this.dataEntries = {};
        this.numImages = window.numImages_x * window.numImages_y;
    }

    loadNextBatch() {
        var self = this;

        // clear current entries
        this.parentDiv.empty();
        this.dataEntries = [];

        var url = 'getLatestImages?limit=' + this.numImages;
        $.ajax({
            url: url,
            dataType: 'json',
            success: function(data) {
                console.log(data['entries'])
                for(var d in data['entries']) {
                    // create new data entry
                    switch(String(window.annotationType)) {
                        case 'labels':
                            var entry = new ClassificationEntry(d, data['entries'][d]);
                            break;
                        case 'points':
                            var entry = new PointAnnotationEntry(d, data['entries'][d]);
                            break;
                        case 'boundingBoxes':
                            var entry = new BoundingBoxAnnotationEntry(d, data['entries'][d]);
                            break;
                        default:
                            break;
                    }

                    // append
                    self.parentDiv.append(entry.markup);
                    self.dataEntries.push(entry);
                }
            },
            error: function(xhr, status, error) {
                if(error == 'Unauthorized') {
                    // redirect to login page
                    window.location.href = '/';
                }
            }
        });
    }


    _entriesToJSON(minimal, onlyUserAnnotations) {
        var entries = {};
        for(var e=0; e<this.dataEntries.length; e++) {
            entries[this.dataEntries[e].entryID] = this.dataEntries[e].getProperties(minimal, onlyUserAnnotations);
        }

        return JSON.stringify({
            'entries': entries
        })
    }


    submitAnnotations() {
        var self = this;
        var entries = this._entriesToJSON(true, true);
        $.ajax({
            url: 'submitAnnotations',
            type: 'POST',
            contentType: 'application/json; charset=utf-8',
            data: entries,
            dataType: 'json',
            success: function(response) {
                console.log(response);

                // next batch
                self.loadNextBatch();
            },
            error: function(xhr, status, error) {
                console.log(xhr)
                console.log(status)
                console.log(error)
                if(error == 'Unauthorized') {
                    // redirect to login page
                    window.location.href = '/';
                }
            }
        });
    }
}