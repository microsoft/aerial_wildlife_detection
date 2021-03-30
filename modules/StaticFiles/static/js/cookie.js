/**
 * Helper, responsible for the legally required cookie notice everyone loves.
 * 
 * 2021 Benjamin Kellenberger
 */

$(document).ready(function() {
    $('#cookie-message-close').on('click', function() {
        $('#cookie-message').slideUp();
        try {
            window.setCookie('cookieMsgSuppressed', true)
        } catch {}
    })

    let showCookieMessage = true;
    try {
        let cookiesSuppressed = window.parseBoolean(window.getCookie('cookieMsgSuppressed', false));
        if(typeof(cookiesSuppressed) === 'boolean') {
            showCookieMessage = !cookiesSuppressed;
        }
    } catch {
        showCookieMessage = true;
    }

    if(showCookieMessage) {
        $('#cookie-message').slideDown();
    }
});