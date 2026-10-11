/* One bar chart per problem. No chart library. */
(function () {
  var NS = "http://www.w3.org/2000/svg";

  function compact(value) {
    var abs = Math.abs(value);
    if (abs >= 1000) {
      var scaled = Math.round((value / 1000) * 10) / 10;
      return String(scaled).replace(/\.0$/, "") + "k";
    }
    return String(Math.round(value));
  }

  function place(scale, lo, hi, inner, value) {
    if (scale === "log") {
      var start = Math.log(lo);
      var span = Math.log(hi) - start;
      return span === 0 ? 0 : ((Math.log(value) - start) / span) * inner;
    }
    var spanLinear = hi - lo;
    return spanLinear === 0 ? 0 : ((value - lo) / spanLinear) * inner;
  }

  function node(name, attrs, text) {
    var el = document.createElementNS(NS, name);
    Object.keys(attrs).forEach(function (key) {
      el.setAttribute(key, attrs[key]);
    });
    if (text !== undefined) {
      el.textContent = text;
    }
    return el;
  }

  function draw(section, knownWidth) {
    var mount = section.querySelector(".bench-plot-mount");
    var dataEl = section.querySelector(".bench-plot-data");
    if (!mount || !dataEl) {
      return;
    }
    var spec;
    try {
      spec = JSON.parse(dataEl.textContent);
    } catch (error) {
      return;
    }
    var points = (spec.points || []).filter(function (point) {
      return typeof point.value === "number" && isFinite(point.value);
    });
    if (points.length < 2) {
      return;
    }
    var width = knownWidth || 640;
    var row = 26;
    var margin = { top: 8, right: 72, bottom: 28, left: Math.min(220, Math.round(width * 0.42)) };
    var inner = Math.max(40, width - margin.left - margin.right);
    var height = margin.top + margin.bottom + points.length * row;
    var values = points.map(function (point) { return point.value; });
    var lo = Math.min.apply(null, values);
    var hi = Math.max.apply(null, values);
    var scale = spec.scale;
    if (!(scale === "log" && lo > 0 && lo !== hi)) {
      scale = "linear";
      lo = Math.min(0, lo);
      hi = Math.max(0, hi);
    }
    var color = getComputedStyle(document.body).color;
    var accent = getComputedStyle(document.body).getPropertyValue("--sy-c-link").trim() || color;
    var svg = node("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": (spec.metric || "score").replace(/_/g, " ") + ", " + points.length + " entries"
    });
    var band = (height - margin.top - margin.bottom) / points.length;
    var bar = band * 0.78;
    points.forEach(function (point, index) {
      var y = margin.top + index * band;
      var widthBar = Math.max(0, place(scale, lo, hi, inner, point.value));
      var link = node("a", {
        href: "#" + point.id,
        "aria-label": point.entry + " " + point.value
      });
      link.appendChild(node("rect", {
        x: String(margin.left),
        y: String(y),
        width: String(widthBar),
        height: String(bar),
        fill: point.best ? accent : color,
        "fill-opacity": point.held ? "0.35" : (point.best ? "0.95" : "0.55")
      }));
      link.appendChild(node("text", {
        class: "value",
        x: String(margin.left + widthBar + 6),
        y: String(y + bar / 2),
        dy: "0.32em",
        fill: color
      }, compact(point.value)));
      svg.appendChild(link);
      var entry = point.entry.length > 32 ? point.entry.slice(0, 30) + "…" : point.entry;
      svg.appendChild(node("text", {
        class: "label",
        x: String(margin.left - 8),
        y: String(y + bar / 2),
        dy: "0.32em",
        fill: color,
        "text-anchor": "end"
      }, entry));
    });
    svg.appendChild(node("path", {
      d: "M" + margin.left + " " + (height - margin.bottom) + "H" + (margin.left + inner),
      fill: "none",
      stroke: color
    }));
    mount.replaceChildren(svg);
  }

  function drawAll() {
    var sections = document.querySelectorAll(".bench");
    var widths = [];
    sections.forEach(function (section) {
      var mount = section.querySelector(".bench-plot-mount");
      widths.push(mount ? mount.clientWidth : 0);
    });
    sections.forEach(function (section, index) {
      draw(section, widths[index] || 640);
    });
  }

  function start() {
    drawAll();
    var timer = 0;
    window.addEventListener("resize", function () {
      window.clearTimeout(timer);
      timer = window.setTimeout(drawAll, 120);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
}());
