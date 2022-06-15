/*
    Helper classes responsible for displaying the available label classes on the screen.

    2019-22 Benjamin Kellenberger
*/

// for segmentation: avoid having duplicate colors
window.usedColors = {
    '#000000':1,
    '#ffffff':1
}

window.parseClassdefEntry = function(id, entry, parent) {
    if(entry.hasOwnProperty('entries') && entry['entries'] != undefined) {
        // label class group
        if(Object.keys(entry['entries']).length > 0) {
            return new LabelClassGroup(id, entry, parent);
        } else {
            // empty group
            return null;
        }
    } else {
        // label class
        return new LabelClass(id, entry, parent);
    }
}

window._rainbow = function(numOfSteps, step) {
    // This function generates vibrant, "evenly spaced" colours (i.e. no clustering). This is ideal for creating easily distinguishable vibrant markers in Google Maps and other apps.
    // Adam Cole, 2011-Sept-14
    // HSV to RBG adapted from: http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
    var r, g, b;
    var h = step / numOfSteps;
    var i = ~~(h * 6);
    var f = h * 6 - i;
    var q = 1 - f;
    switch(i % 6){
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = q; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = q; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        case 5: r = 1; g = 0; b = q; break;
    }
    var c = "#" + ("00" + (~ ~(r * 255)).toString(16)).slice(-2) + ("00" + (~ ~(g * 255)).toString(16)).slice(-2) + ("00" + (~ ~(b * 255)).toString(16)).slice(-2);
    return (c);
}

window.initClassColors = function(numColors) {
    window.defaultColors = [];
    for(var c=0; c<numColors; c++) {
        window.defaultColors.push(
            window._rainbow(numColors, c)
        );
    }
}

window.getDefaultColor = function(idx) {
    return window.defaultColors[idx % window.defaultColors.length];
}


class LabelClass {
    constructor(classID, properties, parent) {
        this.classID = classID;
        this.name = (properties['name']===null || properties['name'] === undefined ? '[Label Class '+this.classID+']' : properties['name']);
        this.index = properties['index'];
        this.color = (properties['color']===null  || properties['color'] === undefined ? window.getDefaultColor(this.index) : properties['color']);
        if('segmentationMasks'.includes([window.annotationType, window.predictionType])) {
            // we disallow duplicate colors for segmentation masks
            this.color = window.rgbToHex(this.color);
            if(window.usedColors.hasOwnProperty(this.color)) {
                this.color = window.getDefaultColor(this.index);    //TODO: this should work, but better use random color instead?
            }
            window.usedColors[this.color] = 1;
        }

        this.colorValues = window.getColorValues(this.color);   // [R, G, B, A]

        this.keystroke = properties['keystroke'];


        // flip active foreground color if background is too bright
        this.darkForeground = (window.getBrightness(this.color) >= 92);
        this.parent = parent;

        // markups
        this.markup = null;
        this.markup_alt = null;
    }

    getMarkup(altStyle) {

        if(altStyle) {
            if(this.markup_alt != undefined && this.markup_alt != null) return this.markup_alt;
        } else {
            if(this.markup != undefined && this.markup != null) return this.markup;
        }

        var self = this;
        var name = this.name;
        var hasKeystroke = false;
        if(this.keystroke != null && this.keystroke != undefined && Number.isInteger(this.keystroke) &&
            this.keystroke > 0 && this.keystroke <= 9) {
            name = '(' + (this.keystroke) + ') ' + this.name;
            hasKeystroke = true;
        }

        var foregroundStyle = '';
        if(altStyle || this.darkForeground) {
            foregroundStyle = 'color:black;';
        }
        var legendInactive = 'legend-inactive';
        if(this.parent.getActiveClassID() === this.classID) legendInactive = '';

        var id = 'labelLegend_' + this.classID;
        var colorStyle = 'background:' + this.color;
        if(altStyle) {
            id = 'labelLegend_alt_' + this.classID;
            var markup = $('<div class="label-class-legend ' + legendInactive + '" id="' + id + '" style="'+foregroundStyle + '"><div class="legend-color-dot" style="' + colorStyle + '"></div><span class="label-text">'+name+'</span></div>');
        } else {
            var markup = $('<div class="label-class-legend ' + legendInactive + '" id="' + id + '" style="'+foregroundStyle + colorStyle + '"><span class="label-text">'+name+'</span></div>');
        }
        
        // setup click handler to activate label class
        markup.click(function() {
            if(window.uiBlocked) return;
            self.parent.setActiveClass(self);
        });

        // listener for keypress if keystroke defined
        if(hasKeystroke) {
            $(window).keyup(function(event) {
                if(window.uiBlocked || window.shortcutsDisabled || window.fieldInFocus()) return;
                try {
                    var key = parseInt(String.fromCharCode(event.which));
                    if(key == self.keystroke) {
                        self.parent.setActiveClass(self);

                        window.dataHandler.renderAll();
                    }
                } catch {
                    return;
                }
            });
        }

        // save for further use
        if(altStyle) this.markup_alt = markup;
        else this.markup = markup;

        return markup;
    }


    filter(keywords) {
        /*
            Shows (or hides) this entry if it matches (or does not match)
            one or more of the keywords specified according to the Leven-
            shtein distance.
        */
        if(keywords === null || keywords === undefined) {
            if(this.markup != null) {
                this.markup.show();
            }
            if(this.markup_alt != null) {
                this.markup_alt.show();
            }
            return { dist: 0, bestMatch: this };
        }
        var target = this.name.toLowerCase();
        var minLevDist = 1e9;
        for(var k=0; k<keywords.length; k++) {
            var kw = keywords[k].toLowerCase();
            var levDist = window.levDist(target, kw);
            minLevDist = Math.min(minLevDist, levDist);
            if(target.includes(kw) || levDist <= 3) {
                if(this.markup != null) {
                    this.markup.show();
                }
                if(this.markup_alt != null) {
                    this.markup_alt.show();
                }
                if(target === kw) minLevDist = 0;
                else if(target.includes(kw)) minLevDist = 0.5;
                return { dist: minLevDist, bestMatch: this };
            }
        }

        // invisible
        if(this.markup != null) {
            this.markup.hide();
        }
        if(this.markup_alt != null) {
            this.markup_alt.hide();
        }
        return { dist: minLevDist, bestMatch: this };
    }
}



class LabelClassGroup {
    constructor(id, properties, parent) {
        this.id = id;
        this.parent = parent;
        this.children = [];
        this.labelClasses = {};
        this.markup = null;
        this._parse_properties(properties);
    }

    _parse_properties(properties) {
        this.name = properties['name'];
        this.color = properties['color'];

        // append children in order
        for(var key in properties['entries']) {

            var nextItem = window.parseClassdefEntry(key, properties['entries'][key], this);
            if(nextItem === null) continue;

            this.children.push(nextItem);
            if(nextItem instanceof LabelClass) {
                this.labelClasses[key] = nextItem;
            } else {
                // append label class group's entries
                this.labelClasses = {...this.labelClasses, ...nextItem.labelClasses};
            }
        }
    }

    getMarkup() {
        if(this.markup != null) return this.markup;

        this.markup = $('<div class="labelGroup"></div>');
        var childrenDiv = $('<div class="labelGroup-children"></div>');

        // append all children
        for(var c=0; c<this.children.length; c++) {
            childrenDiv.append($(this.children[c].getMarkup()));
        }

        // expand/collapse on header click
        if(this.name != null && this.name != undefined) {
            var markupHeader = $('<h3 class="expanded">' + this.name + '</h3>');
            markupHeader.click(function() {
                $(this).toggleClass('expanded');
                if(childrenDiv.is(':visible')) {
                    childrenDiv.slideUp();
                } else {
                    childrenDiv.slideDown();
                }
            });
            this.markup.append(markupHeader);
        }

        this.markup.append(childrenDiv);

        return this.markup;
    }

    getActiveClassID() {
        return this.parent.getActiveClassID();
    }

    setActiveClass(labelClassInstance) {
        this.parent.setActiveClass(labelClassInstance);
    }

    filter(keywords) {
        /*
            Delegates the command to the children and awaits their
            response. If none of the children are visible after
            filtering, the group itself is being hidden. Propagates
            the state of itself to the parent through the return
            statement.
        */
        var childVisible = false;
        var minLevDist = 1e9;
        var argMin = null;
        for(var c=0; c<this.children.length; c++) {
            var result = this.children[c].filter(keywords);
            if(result != null && result.dist < minLevDist) {
                minLevDist = Math.min(result.dist, minLevDist);
                argMin = result.bestMatch;
                if(result.dist <= 3) {
                    childVisible = true;
                }
            }
        }

        // show or hide group depending on children's visibility
        if(childVisible) this.markup.show();
        else this.markup.hide();
        return { dist: minLevDist, bestMatch: argMin };
    }
}


class LabelClassHandler {
    constructor(classLegendDiv) {
        this.classLegendDiv = classLegendDiv;
        this.items = [];    // may be both labelclasses and groups
        this.dummyClass = new LabelClass('00000000-0000-0000-0000-000000000000',
            {
                name: 'Label class',
                index: 1,
                color: '#17a2b8',
                keystroke: null
            });
        this._setupLabelClasses();

        this.setActiveClass(this.labelClasses[Object.keys(this.labelClasses)[0]]);
        
    }

    _setupLabelClasses() {
        // parse label classes and class groups
        this.labelClasses = {};
        this.indexToLabelclassMapping = []; // LUT for indices to label classes
        this.labelToColorMapping = {};  // LUT for color hex strings to label classes

        // initialize default rainbow colors
        window.initClassColors(window.classes.numClasses+1)
        for(var c in window.classes['entries']) {
            var nextItem = window.parseClassdefEntry(c, window.classes['entries'][c], this);
            if(nextItem === null) continue;

            if(nextItem instanceof LabelClass) {
                this.labelClasses[c] = nextItem;
            } else {
                // append label class group's entries
                this.labelClasses = {...this.labelClasses, ...nextItem.labelClasses};
            }
            this.items.push(nextItem);

            // append to div
            this.classLegendDiv.append(nextItem.getMarkup());
        }

        // create labelclass color LUT
        for(var lc in this.labelClasses) {
            var nextItem = this.labelClasses[lc];
            var colorString = window.rgbToHex(nextItem.color);
            this.labelToColorMapping[colorString] = nextItem;
            this.indexToLabelclassMapping[nextItem.index] = nextItem;
        }
    }

    getClass(id) {
        if(id == '00000000-0000-0000-0000-000000000000') {
            // dummy class for UI tutorial
            return this.dummyClass;
        }
        return this.labelClasses[id];
    }

    getActiveClass() {
        return this.activeClass;
    }

    getActiveClassID() {
        return (this.activeClass == null? null : this.activeClass['classID']);
    }

    getActiveClassName() {
        return (this.activeClass == null? null : this.activeClass['name']);
    }

    getActiveColor() {
        return (this.activeClass == null? null : this.activeClass['color']);
    }

    getColor(classID) {
        if(classID == '00000000-0000-0000-0000-000000000000') {
            // dummy class for UI tutorial
            return this.dummyClass['color'];
        }
        return this.labelClasses[classID]['color'];
    }

    getColor(classID, defaultColor) {
        if(classID == '00000000-0000-0000-0000-000000000000') {
            // dummy class for UI tutorial
            return this.dummyClass['color'];
        }
        try {
            return this.labelClasses[classID]['color'];
        } catch {
            return defaultColor;
        }
    }

    getName(classID) {
        if(classID == '00000000-0000-0000-0000-000000000000') {
            // dummy class for UI tutorial
            return this.dummyClass['name'];
        }
        return (classID == null || !this.labelClasses.hasOwnProperty(classID)? null : this.labelClasses[classID]['name']);
    }

    setActiveClass(labelClassInstance) {
        // reset style of currently active class
        if(this.activeClass != null) {
            $('#labelLegend_'+this.activeClass.classID).toggleClass('legend-inactive');
            $('#labelLegend_alt_'+this.activeClass.classID).toggleClass('legend-inactive');
        }

        this.activeClass = labelClassInstance;

        // apply style to new active class
        if(this.activeClass != null) {
            $('#labelLegend_'+this.activeClass.classID).toggleClass('legend-inactive');
            $('#labelLegend_alt_'+this.activeClass.classID).toggleClass('legend-inactive');
        }

        window.activeClassColor = this.getActiveColor();
    }


    filter(keywords, autoActivateBestMatch) {
        /*
            Hides label class entries and groups if they do not match
            one or more of the keywords given.
            Matching is done through the Levenshtein distance.
            If autoActivateBestMatch is true, the best-matching entry
            (with the lowest Lev. dist.) is automatically set active.
        */
        var minDist = 1e9;
        var bestMatch = null;
        for(var c=0; c<this.items.length; c++) {
            var response = this.items[c].filter(keywords);
            if(autoActivateBestMatch && response != null && response.dist <= minDist) {
                minDist = response.dist;
                bestMatch = response.bestMatch;
            }
        }

        if(autoActivateBestMatch && bestMatch != null && minDist <= 3) {
            this.setActiveClass(bestMatch);
        }
    }


    getByIndex(index) {
        /*
            Returns the label class whose assigned index matches the value
            provided. Returns null if no match found.
        */
        try {
            return this.indexToLabelclassMapping[index];
        } catch {
            return null;
        }
    }

    getByColor(color) {
        /*
            Returns the label class whose assigned color matches the values
            provided. Returns null if no match found.
        */
        color = window.rgbToHex(color);
        try {
            return this.labelToColorMapping[color];
        } catch {
            return null;
        }
    }
}