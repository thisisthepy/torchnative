// torchnative guide: language and theme toggles, current-page marking.
// Storage is a convenience only; every access is guarded so the page works without it.
(function () {
  var root = document.documentElement;

  function load(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function save(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ }
  }

  // Language: stored choice, else the browser's preference, else English.
  var lang = load("tn-lang");
  if (lang !== "en" && lang !== "ko") {
    var nav = (navigator.languages && navigator.languages[0]) || navigator.language || "en";
    lang = /^ko\b/i.test(nav) ? "ko" : "en";
  }
  root.setAttribute("lang", lang);

  // Theme: only set when the reader chose one; otherwise prefers-color-scheme decides.
  var theme = load("tn-theme");
  if (theme === "light" || theme === "dark") root.setAttribute("data-theme", theme);

  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  document.addEventListener("DOMContentLoaded", function () {
    var langBtn = document.getElementById("lang-toggle");
    var themeBtn = document.getElementById("theme-toggle");

    if (langBtn) langBtn.addEventListener("click", function () {
      lang = root.getAttribute("lang") === "ko" ? "en" : "ko";
      root.setAttribute("lang", lang);
      save("tn-lang", lang);
    });
    if (themeBtn) themeBtn.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      save("tn-theme", next);
    });

    // Mark the current page in the (identical) navigation of every page.
    var here = location.pathname.split("/").pop() || "index.html";
    var links = document.querySelectorAll("nav a[href]");
    for (var i = 0; i < links.length; i++) {
      if (links[i].getAttribute("href") === here) links[i].setAttribute("aria-current", "page");
    }
    // The menu starts open on wide screens and closed on narrow ones.
    var menu = document.querySelector(".side details");
    if (menu && window.matchMedia && window.matchMedia("(max-width: 860px)").matches) menu.removeAttribute("open");
  });
})();
