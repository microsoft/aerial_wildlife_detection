parcelRequire=function(e,r,t,n){var i,o="function"==typeof parcelRequire&&parcelRequire,u="function"==typeof require&&require;function f(t,n){if(!r[t]){if(!e[t]){var i="function"==typeof parcelRequire&&parcelRequire;if(!n&&i)return i(t,!0);if(o)return o(t,!0);if(u&&"string"==typeof t)return u(t);var c=new Error("Cannot find module '"+t+"'");throw c.code="MODULE_NOT_FOUND",c}p.resolve=function(r){return e[t][1][r]||r},p.cache={};var l=r[t]=new f.Module(t);e[t][0].call(l.exports,p,l,l.exports,this)}return r[t].exports;function p(e){return f(p.resolve(e))}}f.isParcelRequire=!0,f.Module=function(e){this.id=e,this.bundle=f,this.exports={}},f.modules=e,f.cache=r,f.parent=o,f.register=function(r,t){e[r]=[function(e,r){r.exports=t},{}]};for(var c=0;c<t.length;c++)try{f(t[c])}catch(e){i||(i=e)}if(t.length){var l=f(t[t.length-1]);"object"==typeof exports&&"undefined"!=typeof module?module.exports=l:"function"==typeof define&&define.amd?define(function(){return l}):n&&(this[n]=l)}if(parcelRequire=f,i)throw i;return f}({"JAiC":[function(require,module,exports) {
"use strict";Object.defineProperty(exports,"__esModule",{value:!0}),exports.default=void 0;var e=a(require("@babel/runtime/helpers/classCallCheck")),t=a(require("@babel/runtime/helpers/createClass")),r=a(require("@babel/runtime/helpers/inherits")),u=a(require("@babel/runtime/helpers/possibleConstructorReturn")),n=a(require("@babel/runtime/helpers/getPrototypeOf")),l=require("pako"),o=a(require("./basedecoder"));function a(e){return e&&e.__esModule?e:{default:e}}function i(e){var t=f();return function(){var r,l=(0,n.default)(e);if(t){var o=(0,n.default)(this).constructor;r=Reflect.construct(l,arguments,o)}else r=l.apply(this,arguments);return(0,u.default)(this,r)}}function f(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],function(){})),!0}catch(e){return!1}}var c=function(u){(0,r.default)(o,u);var n=i(o);function o(){return(0,e.default)(this,o),n.apply(this,arguments)}return(0,t.default)(o,[{key:"decodeBlock",value:function(e){return(0,l.inflate)(new Uint8Array(e)).buffer}}]),o}(o.default);exports.default=c;
},{"@babel/runtime/helpers/classCallCheck":"fcMS","@babel/runtime/helpers/createClass":"P8NW","@babel/runtime/helpers/inherits":"d4H2","@babel/runtime/helpers/possibleConstructorReturn":"pxk2","@babel/runtime/helpers/getPrototypeOf":"UJE0","pako":"DDDl","./basedecoder":"FJDe"}]},{},[], "GeoTIFF")
//# sourceMappingURL=deflate.91612a6b.js.map