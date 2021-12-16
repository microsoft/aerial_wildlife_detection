/**
 * Image manipulation utilities.
 * 
 * 2021 Benjamin Kellenberger
 */

/**
 * TODO: math utils from here:
 * https://stackoverflow.com/questions/48719873/how-to-get-median-and-quartiles-percentiles-of-an-array-in-javascript-or-php
 */
// sort array ascending
const asc = arr => arr.sort((a, b) => a - b);
const sum = arr => arr.reduce((a, b) => a + b, 0);
const mean = arr => sum(arr) / arr.length;

const roundNumber = (value, multiplier) => {
    return Math.round(value * multiplier) / multiplier;
}

// sample standard deviation
const std = (arr) => {
    const mu = mean(arr);
    const diffArr = arr.map(a => (a - mu) ** 2);
    return Math.sqrt(sum(diffArr) / (arr.length - 1));
};

const quantile = (arr_in, q) => {
    const arr = arr_in.slice();
    const sorted = asc(arr);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
        return sorted[base];
    }
};

const linspace = (a,b,n) => {
    if(typeof n === "undefined") n = Math.max(Math.round(b-a)+1,1);
    if(n<2) { return n===1?[a]:[]; }
    var i,ret = Array(n);
    n--;
    for(i=n;i>=0;i--) { ret[i] = (i*b+(n-i)*a)/n; }
    return ret;
}

const quantiles = (arr_in, quants) => {
    /**
     * Quantiles calculation. Works on a single array (band) only.
     */
    const arr = new Float32Array(arr_in);
    const sorted = asc(arr);
    const vals_out = [];
    for(var q in quants) {
        const pos = (sorted.length - 1) * quants[q];
        const base = Math.floor(pos);
        const rest = pos - base;
        if (sorted[base + 1] !== undefined) {
            vals_out.push(sorted[base] + rest * (sorted[base + 1] - sorted[base]));
        } else {
            vals_out.push(sorted[base]);
        }
    }
    return vals_out;
}

const normalizeImage = (arr, makeUint8) => {
    /**
     *  Performs 0-1 or 0-255 (if "makeUint8" is true) rescaling of image.
     */
    let multiplier = (makeUint8 ? 255 : 1.0);
    let promises = arr.map(band => {
        return new Promise((resolve) => {
            let quants = quantiles(band, [0.0,1.0]);
            let quantDiff = quants[1] - quants[0];
            let arr_out = new Float32Array(band);
            let band_norm = arr_out.map(function(v) { 
                return multiplier * (v - quants[0]) / quantDiff;
            });
            if(makeUint8) band_norm = new Uint8ClampedArray(band_norm);
            return resolve(band_norm);
        });
    });
    return Promise.all(promises);
}

const stretchImage = (arr_in, mins, maxs, makeUint8) => {
    /**
     * Rescales the image from band-wise [mins, maxs] ranges into either [0, 1]
     * or [0, 255] if "makeUint8" is true.
     */
    let arr = new Float32Array(arr_in);
    let numBands = mins.length;
    let bands = Array.from(Array(numBands).keys());
    let multiplier = (makeUint8 ? 255 : 1.0);
    if(is_interleaved(arr)) {
        let promises = [];
        for(var b in bands) {
            let rangeDiff = maxs[b] - mins[b];
            promises.push(new Promise((resolve) => {
                let arr_out = [];
                for(var p=parseInt(b); p<arr.length; p+=numBands) {
                    let val = multiplier * (arr[p] - mins[b]) / rangeDiff;
                    arr_out.push(val);
                }
                return resolve(arr_out);
            }));
        }
        return Promise.all(promises)
        .then((bands) => {
            return bsqtobip(bands, (makeUint8 ? 'uint8' : 'float32'));
        });
    } else {
        let promises = bands.map(b => {
            let rangeDiff = maxs[b] - mins[b];
            return new Promise((resolve) => {
                let band_stretch = arr[b].map(pixel => {
                    return multiplier * (pixel - mins[b]) / rangeDiff;
                });
                return resolve(band_stretch);
            });
        });
        return Promise.all(promises);
    }
}

const quantileStretchImage = (arr, low, high, brightness) => {
    let promises = arr.map(band => {
        return new Promise((resolve) => {
            // calc. quantiles
            let quantvals = quantiles(band, [low, high]);
            let quantdiff = parseFloat(quantvals[1] - quantvals[0]);
            let band_stretch = band.map(pixel => {
                return 255 * (pixel - quantvals[0]) / quantdiff + brightness;
            });
            return resolve(band_stretch);
        });
    });
    return Promise.all(promises);
}

const is_interleaved = arr => {
    // if array contains arrays: sequential
    return (
        !Array.isArray(arr[0]) &&
        !ArrayBuffer.isView(arr[0])
    );
}

const biptobsq = (arr, numBands) => {
    //TODO: multithreaded?
    return new Promise((resolve) => {
        let arr_out = [];
        for(var b=0; b<numBands; b++) {
            arr_out.push([]);
        }
        for(var p=0; p<arr.length; p++) {
            arr_out[p%numBands].push(arr[p]);
        }
        return resolve(arr_out)
    });
}

const bsqtobip = (arr, type) => {
    //TODO: multithreaded?
    return new Promise((resolve) => {
        let numVals = 4*arr[0].length;      // 4 for RGBA

        //TODO: make more elegant
        let preparedArray = null;
        if(type === 'float32') preparedArray = new Float32Array(new Array(numVals));
        else if(type === 'uint8') preparedArray = new Uint8ClampedArray(new Array(numVals));
        else preparedArray = new Uint8ClampedArray(new Array(numVals));
        [0, 1, 2, -1].map(bIdx => {
            if(bIdx>=0) {
                let band = arr[Math.min(bIdx, arr.length-1)];
                for(var s=0; s<numVals; s++) {
                    preparedArray[(s*4)+bIdx] = band[s];
                }
            } else {
                // alpha
                for(var s=0; s<numVals; s++) {
                    preparedArray[(s*4)+3] = 255;
                }
            }
        });
        return resolve(preparedArray);
    });
}

const ndim = (arr) => {
    /**
     * Returns the number of dimensions for an array.
     */
    return Array.isArray(arr) ? 
        1 + Math.max(...arr.map(ndim)) :
        0;
}

const band_select = (arr, bands, arr_num_bands) => {
    /**
     * Performs band selection on an array. Can handle both interleaved and band
     * sequential arrays. For interleaved arrays "arr_num_bands" (i.e., number of
     * bands in the original input "arr") must be specified.
     */
    let arr_out = [];
    let promises = [];
    if(!is_interleaved(arr)) {
        // array of arrays: band sequential
        promises = bands.map((band) => {
            arr_out.push(arr[band]);
        });
    } else {
        // interleaved
        let numPix = arr.length / arr_num_bands;
        arr_out = new Array(bands.length*numPix);
        let bandIndices = Array.from(Array(bands.length).keys());
        promises = bandIndices.map((bIdx) => {
            for(var p=0; p<numPix; p++) {
                arr_out[(p*bands.length)+bIdx] = arr[(p*arr_num_bands)+bands[bIdx]];
            }
        });
    }
    return Promise.all(promises).then(() => {
        return arr_out;
    })
}

const to_grayscale = (arr) => {
    /**
     * Works on both interleaved and band sequential arrays with four bands
     * (RGBA). Ignores the alpha band.
     */
    //TODO: multithreaded?
    if(arr[0].length) {
        // array of arrays: band sequential
        return new Promise((resolve) => {
            for(var p=0; p<arr[0].length; p++) {
                let g = (arr[0][p] + arr[1][p] + arr[2][p]) / 3.0;
                arr[0][p] = g;
                arr[1][p] = g;
                arr[2][p] = g;
            }
            return resolve(arr);
        });
    } else {
        // interleaved
        return new Promise((resolve) => {
            for(var p=0; p<arr.length; p+=4) {
                let g = (arr[p] + arr[p+1] + arr[p+2]) / 3.0;
                arr[p] = g;
                arr[p+1] = g;
                arr[p+2] = g;
            }
            return resolve(arr);
        });
    }
}

const white_on_black = (arr) => {
    /**
     * Works on BSQ arrays; expects 255 as maximum value
     */
    return new Promise((resolve) => {
        arr.map((band) => {
            for(var b=0; b<band.length; b++) {
                band[b] = 255 - band[b];
            }
        });
        return resolve(arr);
    });
}

const averageAcrossBands = (arr) => {
    /**
     * Averages values in an array at each pixel location across all bands
     * (first dimension).
     * TODO: generalize better? Speedup? Etc.
     */
    return new Promise((resolve) => {
        let arr_out = new Array();
        for(var x=0; x<arr[0].length; x++) {
            arr_out.push(new Array());
            for(var y=0; y<arr[0][0].length; y++) {
                let avg = 0.0;
                for(var b=0; b<arr.length; b++) {
                    avg += arr[b][x][y];
                }
                avg /= arr.length;
                arr_out[x].push(avg);
            }
        }
        return resolve(arr_out);
    });
}

const padarray = (arr, size, value) => {
    /**
     * Adds padding to a [BxWxH] array of image pixels.
     */
    size = parseInt(size);
    if(size === 0) return Promise.resolve(arr);
    if(value === undefined) value = 0;

    let arr_out = math.clone(arr);
    let promises = [];
    arr_out.map((band) => {
        promises.push(new Promise(resolve => {
            for(var b=0; b<band.length; b++) {
                for(var s=0; s<size; s++) {
                    band[b].splice(s, 0, value);
                    band[b].splice(band[b].length-s, 0, value);
                }
            }
            let pad = new Array(band[0].length).fill(value);
            for(var s=0; s<size; s++) {
                band.splice(s, 0, pad.slice());
                band.splice(band.length-s, 0, pad.slice());
            }
            return resolve(band);
        }));
    });
    return Promise.all(promises);
}

const conv2d = (arr, filter, stride, pad) => {
    /**
     * Performs simple 2D convolution of an image ("arr") with a filter.
     *
     * Inputs:
     * - "arr": BxWxH array (B=bands, W=width, H=height)
     * - "filter": BxWxH array; can also be 1xWxH (same filter applied to all
     *   bands)
     * - "stride": int
     * - "pad": int, size of zero-padding
     */
    stride = Math.max(1, parseInt(stride));
    pad = Math.max(0, parseInt(pad));

    return padarray(arr, pad).then((arr_pad) => {
        let bands = Array.from(Array(arr_pad.length).keys())
        let promises = [];
        let xLen = Math.ceil((arr[0].length - filter[0].length + 2*pad) / stride + 1);
        let yLen = Math.ceil((arr[0][0].length - filter[0][0].length + 2*pad) / stride + 1);
        let arr_out = new Array(bands.length).fill(null);
        arr_out = arr_out.map(() => {
            let xarr = new Array(xLen).fill(null);
            xarr = xarr.map(() => {
                let yarr = new Array(yLen);
                return yarr;
            });
            return xarr;
        });
        let maxX = arr_pad[0].length - Math.ceil(filter[0].length/2.0);
        let maxY = arr_pad[0][0].length - Math.ceil(filter[0][0].length/2.0);
        bands.map((band) => {
            promises.push(new Promise((resolve) => {
                let fband = Math.min(band, filter.length-1);
                let xC = 0;
                for(var x=0; x<maxX; x+=stride) {
                    let yC = 0;
                    for(var y=0; y<maxY; y+=stride) {
                        let val = 0;
                        for(var fx=0; fx<filter[fband].length; fx++) {
                            for(var fy=0; fy<filter[fband][0].length; fy++) {
                                val += filter[fband][fx][fy] * arr_pad[band][x+fx][y+fy];
                            }
                        }
                        arr_out[band][xC][yC] = val;
                        yC++;
                    }
                    xC++;
                }
                return resolve();
            }));
        });
        return Promise.all(promises).then(() => {
            return arr_out;
        });
    });
}

const sobel = (arr) => {
    /**
     * Applies a Sobel Edge detector over an image array.
     * TODO: mask white border around image?
     */
    let filters = [
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ],
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]
    ];
    let promises = [
        conv2d(arr, [filters[0]], 1, 1),
        conv2d(arr, [filters[1]], 1, 1)
    ]
    return Promise.all(promises).then((edges) => {
        // average activations across bands
        promises = [
            averageAcrossBands(edges[0]),
            averageAcrossBands(edges[1])
        ]
        return Promise.all(promises);
    }).then((edges_avg) => {
        // calculate magnitude
        let magnitude = [];
        for(var x=0; x<edges_avg[0].length; x++) {
            magnitude.push(new Array());
            for(var y=0; y<edges_avg[0][0].length; y++) {
                magnitude[x].push(
                    Math.sqrt(
                        Math.pow(edges_avg[0][x][y], 2) + Math.pow(edges_avg[1][x][y], 2)
                    )
                );
            }
        }
        return magnitude;
    });
}