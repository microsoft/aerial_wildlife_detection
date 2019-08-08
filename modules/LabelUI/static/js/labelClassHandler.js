/*
    Helper classes responsible for displaying the available label classes on the screen.

    2019 Benjamin Kellenberger
*/

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

    // shuffle order for easier discrimination
    // window.shuffle(window.defaultColors);    //TODO: gets re-shuffled at load, which is confusing
};


class LabelClass {
    constructor(classID, properties, parent) {
        this.classID = classID;
        this.name = (properties['name']===null || properties['name'] === undefined ? '[Label Class '+this.classID+']' : properties['name']);
        this.index = properties['index'];
        this.color = (properties['color']===null  || properties['color'] === undefined ? window.defaultColors[this.index] : properties['color']);

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
        if(this.index >= 0 && this.index < 9) {
            name = '(' + (this.index+1) + ') ' + this.name;
        }
        var foregroundStyle = '';
        if(altStyle || this.darkForeground) {
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

        // listener for keypress if in [1, 9]
        if(this.index >= 0 && this.index < 9) {
            $(window).keyup(function(event) {
                if(window.uiBlocked || window.shortcutsDisabled) return;
                try {
                    var key = parseInt(String.fromCharCode(event.which));
                    if(key == self.index+1) {
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
            return true;
        }
        var target = this.name.toLowerCase();
        for(var k=0; k<keywords.length; k++) {
            var kw = keywords[k].toLowerCase();
            if(target.includes(kw) || window.levDist(target, kw) <= 3) {
                if(this.markup != null) {
                    this.markup.show();
                }
                if(this.markup_alt != null) {
                    this.markup_alt.show();
                }
                return true;
            }
        }

        // invisible
        if(this.markup != null) {
            this.markup.hide();
        }
        if(this.markup_alt != null) {
            this.markup_alt.hide();
        }
        return false;
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
        for(var c=0; c<this.children.length; c++) {
            if(this.children[c].filter(keywords)) {
                childVisible = true;
            }
        }

        // show or hide group depending on children's visibility
        if(childVisible) this.markup.show();
        else this.markup.hide();
        return childVisible;
    }
}


class LabelClassHandler {
    constructor(classLegendDiv) {
        this.classLegendDiv = classLegendDiv;
        this.items = [];    // may be both labelclasses and groups
        this._setupLabelClasses();

        this.setActiveClass(this.labelClasses[Object.keys(this.labelClasses)[0]]);
    }

    _setupLabelClasses() {
        // parse label classes and class groups
        this.labelClasses = {};

        // initialize default rainbow colors
        window.initClassColors(window.classes.numClasses)

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
    }

    getClass(id) {
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
        return this.labelClasses[classID]['color'];
    }

    getColor(classID, defaultColor) {
        try {
            return this.labelClasses[classID]['color'];
        } catch {
            return defaultColor;
        }
    }

    getName(classID) {
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
    }


    filter(keywords) {
        /*
            Hides label class entries and groups if they do not match
            one or more of the keywords given.
            Matching is done through the Levenshtein distance.
        */
        for(var c=0; c<this.items.length; c++) {
            this.items[c].filter(keywords);
        }
    }
}