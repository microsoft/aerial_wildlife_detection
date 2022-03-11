/**
 * Classes for quiz functionality.
 * 
 * 2022 Benjamin Kellenberger
 */

function isInt(n) {
    return n % 1 === 0;
}

const awaitTimeout = delay =>
    new Promise(resolve => setTimeout(resolve, delay));


const RATINGS_TEXT = [
    [0, 'Better luck next time!'],
    [50, 'Great start!'],
    [65, 'Not bad!'],
    [75, 'Pretty good!'],
    [85, 'Great work!'],
    [90, 'Impressive!'],
    [100, 'Perfect score!']
];

function delay(t, v) {  // required for flashing of correct predictions
    return new Promise(function(resolve) { 
        setTimeout(resolve.bind(null, v), t)
    });
 }


class QuizMeter {
    /**
     * Class that displays a title, meter (bar) and value for an accuracy
     * measure.
     */
    constructor(title, value, min, max, multiplier, suffix, showMax) {
        this.title = title;
        this.value = value;
        this.min = min;
        this.max = max;
        this.multiplier = typeof(multiplier) === 'number'? multiplier : 1.0;    // multiplication factor (e.g., to convert fractions to percentages)
        this.suffix = typeof(suffix) === 'string'? suffix : '';                 // suffix to display (e.g., "%")
        this.showMax = showMax;
        this.markup = this._setup_markup();
    }

    _setup_markup() {
        let markup = $('<div class="quiz-meter-container"></div>');
        let titleMarkup = $('<div>' + this.title + '</div>');
        markup.append(titleMarkup);
        let quizMeterPanel = $('<div class="quiz-meter-panel"></div>');
        if(typeof(this.max) === 'number') {
            // only show a bar if values are bounded
            let barContainer = $('<div class="quiz-meter-bar-container"></div>');
            this.bar = $('<div class="quiz-meter-bar"></div>');
            this.bar.css('width', 100*parseFloat(this.value)/(this.max - this.min) + '%');
            barContainer.append(this.bar);
            quizMeterPanel.append(barContainer);
        }

        let statusText = (this.value * this.multiplier);
        if(isInt(statusText)) statusText = statusText.toString();
        else statusText = statusText.toFixed(2);
        if(this.showMax) statusText += '/' + this.max;
        statusText += this.suffix;
        this.statusTextDiv = $('<span>' + statusText + '</span>');
        quizMeterPanel.append(this.statusTextDiv);
        markup.append(quizMeterPanel);

        return markup;
    }

    getMarkup(clone) {
        if(clone) return this._setup_markup();
        else return this.markup;
    }

    setValue(value) {
        this.value = value;
        if(typeof(this.max) === 'number') this.value = Math.min(this.max, value);
        if(!isFinite(this.value)) this.value = 0;
        if(this.bar !== undefined) {
            this.bar.animate({
                'width': 100*parseFloat(this.value)/(this.max - this.min) + '%'
            }, 2000);
        }
        let statusText = (this.value * this.multiplier);
        if(!isInt(statusText)) statusText = statusText.toFixed(2);
        if(this.showMax) statusText += '/' + this.max;
        statusText += this.suffix;
        this.statusTextDiv.html(statusText);
    }
}


class QuizHandler {

    CORRECT_FLASH_TIMEOUT = 500;    // timeout [ms] between flashes of correct annotations
    CORRECT_FLASH_NUM = 3;          // number of times to flash correct annotations

    TRY_AGAIN_TIMEOUT = 30000;      // waiting time until "try again" button is automatically clicked

    STATISTICS_TOKENS = {           // key: [title, min, max, multiplier, suffix, showMax]
        'boundingBoxes': {
            'num_tp': ['# correct (true positives)', 0, 0, 1.0, '', true],      // we'll dynamically update the maximum as we move along
            'num_fp': ['# errors (false positives)', 0, null, 1.0, ''],         // no max for FP, so no meter bar
            'num_fn': ['# missed (false negatives)', 0, 0, 1.0, '', true],      // same as for TP                                                     
            'precision': ['Correctness score (precision)', 0, 1, 100.0, '%', false],
            'recall': ['Completeness score (recall)', 0, 1, 100.0, '%', false],
            'avg_iou_correct': ['Geometrical precision (Intersection-over-Union)', 0, 1, 100.0, '%', false],
            'num_imgs_match': null
        }
    }


    constructor() {
        // running stats
        this.reset();

        // set up markup for running stats
    }

    _step_stats(result) {
        /**
         * Extracts "global" statistics (over the current batch) from the
         * results as per annotationType and augments the instance's stats
         * tracker accordingly.
         */
        this.stats_count += result['num_img'];
        for(var key in this.stats_tokens) {
            this.stats[key] += result[key];
        }

        // set maximums for TP, FP and FN meters (TODO: also hacky)
        if(this.quizMeters.hasOwnProperty('num_tp')) {
            this.quizMeters['num_tp'].max = this.stats['num_tp'] + this.stats['num_fn'];
            this.quizMeters['num_tp'].setValue(this.stats['num_tp']);
            this.quizMeters['num_fn'].max = this.stats['num_tp'] + this.stats['num_fn'];
            this.quizMeters['num_fn'].setValue(this.stats['num_fn']);
        }

        // update actual values
        let currentStats = this.getResult(false);
        for(var key in this.quizMeters) {
            this.quizMeters[key].setValue(currentStats[key]);
        }

        // image counter meter
        this.quizMeters['_num_images'].setValue(this.stats_count);
    }

    step() {
        /**
         * Queries the data handler for the latest annotations on all visible
         * data entries and calculates accuracies.
         * TODO: flash correct annotations?
         */
        return new Promise(resolve => {
            let entries = window.dataHandler.getEntryDetails(true, true);
            entries = entries['entries'];       // discard metadata
            let self = this;
            return $.ajax({
                url: window.baseURL + 'getAccuracy',
                method: 'POST',
                contentType: 'application/json; charset=utf-8',
                dataType: 'json',
                data: JSON.stringify({
                    entries: entries
                }),
                success: function(data) {
                    if(data.hasOwnProperty('result')) {
                        data = data['result'];

                        // hide all annotations (for now)
                        window.dataHandler.setAnnotationsVisible(false);

                        // deactivate UI temporarily
                        window.setUIblocked(true);

                        // add targets as solutions
                        if(data.hasOwnProperty('targets')) {
                            let targets = data['targets'];
                            for(var key in targets) {
                                // find correct entry       //TODO: dirty
                                let eIdx = null;
                                for(var e=0; e<window.dataHandler.dataEntries.length; e++) {
                                    if(window.dataHandler.dataEntries[e].entryID === key) {
                                        eIdx = e;
                                        break;
                                    }
                                }
                                if(eIdx !== null) {
                                    let annos = targets[key]['annotations'];
                                    for(var annoKey in annos) {
                                        let annoDef = annos[annoKey];
                                        let annotation = new Annotation(annoKey, annoDef, window.annotationType, 'annotation');
                                        window.dataHandler.dataEntries[eIdx]._addElement(annotation);       //TODO
                                    }
                                }
                            }
                            window.dataHandler.renderAll();
                        }

                        // flash correct predictions
                        let keys_correct = {};      // key: dataEntry ID, value: list of annotation IDs
                        for(var entryKey in data['img']) {
                            let tp = data['img'][entryKey]['tp'];
                            keys_correct[entryKey] = tp;
                        }
                        
                        let flashPredictions = function(visible) {
                            for(var entryKey in keys_correct) {
                                let entryPos = entries[entryKey]['position'];
                                for(var annoKey in keys_correct[entryKey]) {
                                    window.dataHandler.dataEntries[entryPos].annotations[keys_correct[entryKey][annoKey]].setVisible(visible);
                                }
                            }
                            window.dataHandler.renderAll();
                        }

                        let promise = awaitTimeout(self.CORRECT_FLASH_TIMEOUT).then(() => flashPredictions(true));
                        for(let n=0; n<self.CORRECT_FLASH_NUM; n++) {
                            promise = promise.then(function() {
                                return awaitTimeout(self.CORRECT_FLASH_TIMEOUT).then(() => flashPredictions(n%2!==0));
                            });
                        }
                        promise.then(function() {
                            // update tracked stats
                            self._step_stats(data['global']);

                            // check if finished
                            if(self.stats_count >= window.quizProperties['num_images']) {
                                // number of images completed; show final results
                                let statsMarkup = self.getResult(true);
                                window.showOverlay(statsMarkup, true, true);
                            } else {
                                // continue as usual
                                window.setUIblocked(false);
                            }
                            return resolve();
                        });
                    } else {
                        //TODO
                        window.setUIblocked(false);
                        return resolve();
                    }
                }
            });
        });
    }

    getResult(asHTML) {
        /**
         * Calculates and returns the current result, either as an object or as
         * a full markup to be displayed in an overlay.
         */
        // calculate normalized stats   TODO: occasional hackiness
        let stats_current = JSON.parse(JSON.stringify(this.stats));
        for(var key in this.stats_tokens) {
            if(key === 'precision') {
                stats_current[key] = stats_current['num_tp'] / (stats_current['num_tp'] + stats_current['num_fp']);
            } else if(key === 'recall') {
                stats_current[key] = stats_current['num_tp'] / (stats_current['num_tp'] + stats_current['num_fn']);
            } else if(key === 'avg_iou_correct') {
                stats_current[key] /= stats_current['num_imgs_match'];
            } else if(!key.startsWith('num')) {
                stats_current[key] /= parseFloat(this.stats_count);
            }
        }

        if(!asHTML) return stats_current;

        // create markup
        let markup = $('<div></div>');
        markup.append($('<h2>Finished!</h2>'));
        let statsContainer = $('<div></div>');
        let resultText = 'You viewed ' + this.stats_count + ' images ';
        let scoringVal = 0;
        switch(window.annotationType) {
            case 'labels':
                resultText += 'and correctly annotated ' + stats_current['num_tp'] + ' of them.';
                scoringVal = 100.0 * stats_current['num_tp'] / this.stats_count;
                break;
            case 'boundingBoxes':
                let numObj = stats_current['num_tp'] + stats_current['num_fn'];
                let iou = (100.0 * stats_current['avg_iou_correct']).toFixed(2);
                let recall = (100.0 * stats_current['recall']).toFixed(2);
                resultText += 'and found ' + stats_current['num_tp'] + ' out of ' + numObj + ' objects (' + recall + '%) with a geometric accuracy of ' + iou + '%.<br />';
                resultText += 'You mistakenly identified ' + stats_current['num_fp'] + ' regions as objects.';
                // F1 score for scoring value
                scoringVal = 100.0 * 2 * stats_current['precision'] * stats_current['recall'] / (stats_current['precision'] + stats_current['recall']); //TODO: add IoU?
                if(!isFinite(scoringVal)) scoringVal = 0;
                break;
            //TODO: others
        }
        // get verbal feedback
        for(var r=0; r<RATINGS_TEXT.length; r++) {
            if(scoringVal < RATINGS_TEXT[r][0]) {
                break;
            }
        }
        resultText += ' ' + RATINGS_TEXT[r-1][1];
        statsContainer.append($('<p>' + resultText + '</p>'));

        // detailed stats
        statsContainer.append($('<h3>Detailed statistics</h3>'));
        let detStatsCont = $('<div id="#quiz-container-panel-result"></div>');
        for(var key in this.quizMeters) {
            detStatsCont.append(this.quizMeters[key].getMarkup(true));
        }
        statsContainer.append(detStatsCont);

        // restart button
        let self = this;
        let restartButton = $('<button class="btn btn-lg btn-primary">Try again</button>');
        restartButton.on('click', function() {
            window.setUIblocked(false);
            window.showOverlay(null);
            window.dataHandler.nextBatch(true);
            self.reset();
        });
        statsContainer.append(restartButton);

        // // restart timer //TODO: doesn't work with JS
        // let restartTimer = $('<span style="margin-left:20px">' + parseInt(self.TRY_AGAIN_TIMEOUT/1000).toString() + '...</span>');
        // statsContainer.append(restartTimer);
        // let clockTimer = function(timeout) {
        //     if(timeout <= 0) {
        //         restartButton.trigger('click');
        //     } else {
        //         restartTimer.html(parseInt(timeout/1000).toString() + '...');
        //         setTimeout(clockTimer(timeout - 1000), 1000);
        //     }
        // }
        // new Promise(() => {
        //     clockTimer(self.TRY_AGAIN_TIMEOUT);
        // });

        return statsContainer;
    }

    reset() {
        /**
         * Resets all running stats. Also initializes meters in the interface
         * for progress.
         */
        this.stats_count = 0;
        this.quizMeters = {};
        $('#quiz-container-panel').empty();

        // general quiz meter for number of images
        this.quizMeters['_num_images'] = new QuizMeter('Images', 0, 0, window.quizProperties['num_images'], 1.0, null, true);
        $('#quiz-container-panel').append(this.quizMeters['_num_images'].getMarkup());
        
        this.stats = {};
        this.stats_tokens = this.STATISTICS_TOKENS[window.annotationType];
        for(var key in this.stats_tokens) {
            this.stats[key] = 0;
            let meterSpec = this.stats_tokens[key];
            if(meterSpec !== null) {
                this.quizMeters[key] = new QuizMeter(meterSpec[0], 0, meterSpec[1], meterSpec[2], meterSpec[3], meterSpec[4], meterSpec[5]); // title, value, min, max, multiplier, suffix, showMax
                $('#quiz-container-panel').append(this.quizMeters[key].getMarkup());
            }
        }
    }
}