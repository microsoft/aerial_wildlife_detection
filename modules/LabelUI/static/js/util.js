/**
 * Utility functions for the labeling interface.
 * 
 * 2020-22 Benjamin Kellenberger
 */


 // enable/disable interface
 window.setUIblocked = function(blocked) {
    window.uiBlocked = blocked;
    $('button').prop('disabled', blocked);
}

// loading overlay
window.showLoadingOverlay = function(visible) {
    if(visible) {
        window.setUIblocked(true);
        $('#overlay').css('display', 'block');
        $('#overlay-loader').css('display', 'block');
        $('#overlay-card').css('display', 'none');

    } else {
        $('#overlay').fadeOut({
            complete: function() {
                $('#overlay-loader').css('display', 'none');
            }
        });
        window.setUIblocked(false);
    }
}

window.fieldInFocus = function() {
    /**
     * Returns true if at least one HTML input field is in focus
     */
    return document.activeElement.nodeName.toLowerCase() === 'input';
}

window.getCookie = function(name, decodeToObject) {
    let match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
    if (match) match = match[2];

    if(typeof(match) === 'string' && decodeToObject) {
        let tokens = match.split(',');
        if(tokens.length >= 2 && tokens.length % 2 == 0) {
            match = {};
            for(var t=0; t<tokens.length; t+=2) {
                match[tokens[t]] = tokens[t+1];
            }
        }
    }
    return match;
}
window.setCookie = function(name, value, days) {
    if(typeof(value) === 'object') {
        let objStr = '';
        for(var key in value) {
            let val = value[key];
            objStr += key + ',' + val + ',';
        }
        value = objStr.slice(0, -1);
    }
    var d = new Date;
    d.setTime(d.getTime() + 24*60*60*1000*days);
    document.cookie = name + "=" + value + ";path=/;expires=" + d.toGMTString();
}


// time util
window.msToTime = function(duration) {
    var seconds = Math.floor((duration / 1000) % 60),
        minutes = Math.floor((duration / (1000 * 60)) % 60),
        hours = Math.floor((duration / (1000 * 60 * 60)) % 24);

    if(hours > 0) {
        hours = (hours < 10) ? '0' + hours : hours;
        minutes = (minutes < 10) ? '0' + minutes : minutes;
        seconds = (seconds < 10) ? '0' + seconds : seconds;
        result = hours + ':' + minutes + ':' + seconds;
        return result;

    } else {
        minutes = (minutes < 10) ? '0' + minutes : minutes;
        seconds = (seconds < 10) ? '0' + seconds : seconds;
        return minutes + ':' + seconds;
    }
}


/* color functions */

// source: https://stackoverflow.com/questions/1573053/javascript-function-to-convert-color-names-to-hex-codes/24390910
var WEB_COLORS = {
    "aliceblue":"#f0f8ff","antiquewhite":"#faebd7","aqua":"#00ffff","aquamarine":"#7fffd4","azure":"#f0ffff",
    "beige":"#f5f5dc","bisque":"#ffe4c4","black":"#000000","blanchedalmond":"#ffebcd","blue":"#0000ff","blueviolet":"#8a2be2","brown":"#a52a2a","burlywood":"#deb887",
    "cadetblue":"#5f9ea0","chartreuse":"#7fff00","chocolate":"#d2691e","coral":"#ff7f50","cornflowerblue":"#6495ed","cornsilk":"#fff8dc","crimson":"#dc143c","cyan":"#00ffff",
    "darkblue":"#00008b","darkcyan":"#008b8b","darkgoldenrod":"#b8860b","darkgray":"#a9a9a9","darkgreen":"#006400","darkkhaki":"#bdb76b","darkmagenta":"#8b008b","darkolivegreen":"#556b2f",
    "darkorange":"#ff8c00","darkorchid":"#9932cc","darkred":"#8b0000","darksalmon":"#e9967a","darkseagreen":"#8fbc8f","darkslateblue":"#483d8b","darkslategray":"#2f4f4f","darkturquoise":"#00ced1",
    "darkviolet":"#9400d3","deeppink":"#ff1493","deepskyblue":"#00bfff","dimgray":"#696969","dodgerblue":"#1e90ff",
    "firebrick":"#b22222","floralwhite":"#fffaf0","forestgreen":"#228b22","fuchsia":"#ff00ff",
    "gainsboro":"#dcdcdc","ghostwhite":"#f8f8ff","gold":"#ffd700","goldenrod":"#daa520","gray":"#808080","green":"#008000","greenyellow":"#adff2f",
    "honeydew":"#f0fff0","hotpink":"#ff69b4",
    "indianred ":"#cd5c5c","indigo":"#4b0082","ivory":"#fffff0","khaki":"#f0e68c",
    "lavender":"#e6e6fa","lavenderblush":"#fff0f5","lawngreen":"#7cfc00","lemonchiffon":"#fffacd","lightblue":"#add8e6","lightcoral":"#f08080","lightcyan":"#e0ffff","lightgoldenrodyellow":"#fafad2",
    "lightgrey":"#d3d3d3","lightgreen":"#90ee90","lightpink":"#ffb6c1","lightsalmon":"#ffa07a","lightseagreen":"#20b2aa","lightskyblue":"#87cefa","lightslategray":"#778899","lightsteelblue":"#b0c4de",
    "lightyellow":"#ffffe0","lime":"#00ff00","limegreen":"#32cd32","linen":"#faf0e6",
    "magenta":"#ff00ff","maroon":"#800000","mediumaquamarine":"#66cdaa","mediumblue":"#0000cd","mediumorchid":"#ba55d3","mediumpurple":"#9370d8","mediumseagreen":"#3cb371","mediumslateblue":"#7b68ee",
    "mediumspringgreen":"#00fa9a","mediumturquoise":"#48d1cc","mediumvioletred":"#c71585","midnightblue":"#191970","mintcream":"#f5fffa","mistyrose":"#ffe4e1","moccasin":"#ffe4b5",
    "navajowhite":"#ffdead","navy":"#000080",
    "oldlace":"#fdf5e6","olive":"#808000","olivedrab":"#6b8e23","orange":"#ffa500","orangered":"#ff4500","orchid":"#da70d6",
    "palegoldenrod":"#eee8aa","palegreen":"#98fb98","paleturquoise":"#afeeee","palevioletred":"#d87093","papayawhip":"#ffefd5","peachpuff":"#ffdab9","peru":"#cd853f","pink":"#ffc0cb","plum":"#dda0dd","powderblue":"#b0e0e6","purple":"#800080",
    "rebeccapurple":"#663399","red":"#ff0000","rosybrown":"#bc8f8f","royalblue":"#4169e1",
    "saddlebrown":"#8b4513","salmon":"#fa8072","sandybrown":"#f4a460","seagreen":"#2e8b57","seashell":"#fff5ee","sienna":"#a0522d","silver":"#c0c0c0","skyblue":"#87ceeb","slateblue":"#6a5acd","slategray":"#708090","snow":"#fffafa","springgreen":"#00ff7f","steelblue":"#4682b4",
    "tan":"#d2b48c","teal":"#008080","thistle":"#d8bfd8","tomato":"#ff6347","turquoise":"#40e0d0",
    "violet":"#ee82ee",
    "wheat":"#f5deb3","white":"#ffffff","whitesmoke":"#f5f5f5",
    "yellow":"#ffff00","yellowgreen":"#9acd32"
};

window.getColorValues = function(color) {
    if(color instanceof Array || color instanceof Uint8ClampedArray) return color;
    color = color.toLowerCase();
    if(color.startsWith('rgb')) {
        var match = /rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,\s*\d+[\.\d+]*)*\)/g.exec(color);
        return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3]), (match.length===4 ? parseInt(match[4]) : 255)];
    } else {
        return window.getColorValues(window.hexToRgb(color));
    }
}

window.rgbToHex = function(rgb) {
    if(WEB_COLORS.hasOwnProperty(rgb)) {
        return WEB_COLORS[rgb];
    }
    var componentToHex = function(c) {
        var hex = parseInt(c).toString(16);
        return hex.length == 1 ? "0" + hex : hex;
    }
    if(!(rgb instanceof Array || rgb instanceof Uint8ClampedArray)) {
        rgb = rgb.toLowerCase();
        if(rgb.startsWith('#')) {
            if(rgb.length < 6) {
                rgb = rgb.split('').map((item)=>{
                    if(item == '#'){return item}
                        return item + item;
                }).join('');
            }
            return rgb;
        } else if(rgb.startsWith('rgb')) {
            rgb = /rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,\s*\d+[\.\d+]*)*\)/g.exec(rgb);
        }
    }
    let offset = rgb.length === 3 ? 0 : 1;
    return "#" + componentToHex(rgb[offset]) + componentToHex(rgb[offset+1]) + componentToHex(rgb[offset+2]);
}

window.hexToRgb = function(hex, as_array) {
    if(hex.toLowerCase().startsWith('rgb')) return hex;
    // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
    var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, function(m, r, g, b) {
        return r + r + g + g + b + b;
    });

    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if(!result) return null;
    if(as_array) {
        return [
            parseInt(result[1], 16),
            parseInt(result[2], 16),
            parseInt(result[3], 16)
        ]
    } else {
        return 'rgb(' + 
            parseInt(result[1], 16) + ',' + 
            parseInt(result[2], 16) + ',' + 
            parseInt(result[3], 16) + ')';
    }
}

window._addAlpha = function(color, alpha) {
    a = alpha > 1 ? (alpha / 100) : alpha;
    if(color.startsWith('#')) {
        // HEX color string
        color = window.hexToRgb(color);
    }
    var match = /rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,\s*\d+[\.\d+]*)*\)/g.exec(color);
    return "rgba(" + [match[1],match[2],match[3],a].join(',') +")";
}

window.addAlpha = function(color, alpha) {
    if(color === null || color === undefined) return null;
    if(alpha === null || alpha === undefined) return color;
    if(alpha <= 0.0) return '#FFFFFF00';
    alpha = alpha > 1 ? (alpha / 100) : alpha;
    if(alpha >= 1.0) return color;
    return window._addAlpha(color, alpha);
}

window.getBrightness = function(color) {
    var rgb = window.hexToRgb(color);
    var match = /rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(,\s*\d+[\.\d+]*)*\)/g.exec(rgb);
    return (parseInt(match[1]) + parseInt(match[2]) + parseInt(match[3])) / 3;
}

window.shuffle = function(a) {
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
}


// base64 conversion
window.bufferToBase64 = function(buf) {
    var binstr = Array.prototype.map.call(buf, function (ch) {
        return String.fromCharCode(ch);
    }).join('');
    return btoa(binstr);
}

window.base64ToBuffer = function(base64) {
    var binstr = atob(base64);
    var buf = new Uint8Array(binstr.length);
    Array.prototype.forEach.call(binstr, function (ch, i) {
      buf[i] = ch.charCodeAt(0);
    });
    return buf;
}

// Levenshtein distance for word comparison
window.levDist = function(s, t) {
    var d = []; //2d matrix

    // Step 1
    var n = s.length;
    var m = t.length;

    if (n == 0) return m;
    if (m == 0) return n;

    //Create an array of arrays in javascript (a descending loop is quicker)
    for (var i = n; i >= 0; i--) d[i] = [];

    // Step 2
    for (var i = n; i >= 0; i--) d[i][0] = i;
    for (var j = m; j >= 0; j--) d[0][j] = j;

    // Step 3
    for (var i = 1; i <= n; i++) {
        var s_i = s.charAt(i - 1);

        // Step 4
        for (var j = 1; j <= m; j++) {

            //Check the jagged ld total so far
            if (i == j && d[i][j] > 4) return n;

            var t_j = t.charAt(j - 1);
            var cost = (s_i == t_j) ? 0 : 1; // Step 5

            //Calculate the minimum
            var mi = d[i - 1][j] + 1;
            var b = d[i][j - 1] + 1;
            var c = d[i - 1][j - 1] + cost;

            if (b < mi) mi = b;
            if (c < mi) mi = c;

            d[i][j] = mi; // Step 6

            //Damerau transposition
            if (i > 1 && j > 1 && s_i == t.charAt(j - 2) && s.charAt(i - 2) == t_j) {
                d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + cost);
            }
        }
    }

    // Step 7
    return d[n][m];
}

// misc.
window.parseBoolean = function(value) {
    if(value === null || value === undefined) return false;
    return (value===1 || ['yes', '1', 'true'].includes(value.toString().toLowerCase()));
}

window.getCurrentDateString = function() {
    var date = new Date();
    return date.toString();
}

window.getRandomString = function() {
    // only used for temporary IDs, never for sensitive hashing
    return Math.random().toString(36).substring(7);
}

window.getRandomID = function() {
    return window.getCurrentDateString() + window.getRandomString();
}

window.argMin = function(array) {
    return [].reduce.call(array, (m, c, i, arr) => c < arr[m] ? i : m, 0)
}

window.argMax = function(array) {
    return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0)
}

window.argsort = function(array, descending) {
    const arrayObject = array.map((value, idx) => { return { value, idx }; });
    arrayObject.sort((a, b) => {
        if (a.value < b.value) {
            return descending? 1 : -1;
        }
        if (a.value > b.value) {
            return descending? -1 : 1;
        }
        return 0;
    });
    const argIndices = arrayObject.map(data => data.idx);
    return argIndices;
 }