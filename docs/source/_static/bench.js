/* One bar chart per problem, from the JSON beside the mount. */
(function () {
  function box(x, scale, value) {
    if (scale === "log") {
      return [0, Math.max(0, x(value))];
    }
    var origin = x(0);
    var end = x(value);
    return [Math.min(origin, end), Math.abs(end - origin)];
  }

  function draw(section, animate) {
    var mount = section.querySelector(".bench-plot-mount");
    var dataEl = section.querySelector(".bench-plot-data");
    if (!mount || !dataEl || !window.d3) {
      return;
    }
    var spec = JSON.parse(dataEl.textContent);
    var points = spec.points.filter(function (point) {
      return typeof point.value === "number" && isFinite(point.value);
    });
    if (points.length < 2) {
      return;
    }
    var width = mount.clientWidth || 640;
    var row = 26;
    var margin = { top: 8, right: 72, bottom: 28, left: Math.min(220, Math.round(width * 0.42)) };
    var inner = Math.max(40, width - margin.left - margin.right);
    var height = margin.top + margin.bottom + points.length * row;
    var values = points.map(function (point) { return point.value; });
    var lo = Math.min.apply(null, values);
    var hi = Math.max.apply(null, values);
    var color = getComputedStyle(document.body).color;
    var accent = getComputedStyle(document.body).getPropertyValue("--sy-c-link").trim() || color;
    var x;
    if (spec.scale === "log" && lo > 0 && lo !== hi) {
      x = d3.scaleLog().domain([lo, hi]).range([0, inner]).nice();
    } else {
      spec.scale = "linear";
      x = d3.scaleLinear().domain([Math.min(0, lo), Math.max(0, hi)]).range([0, inner]).nice();
    }
    var y = d3.scaleBand()
      .domain(points.map(function (point) { return point.id; }))
      .range([margin.top, height - margin.bottom])
      .padding(0.22);
    mount.replaceChildren();
    var svg = d3.select(mount).append("svg")
      .attr("viewBox", "0 0 " + width + " " + height)
      .attr("role", "img")
      .attr("aria-label", spec.metric.replace(/_/g, " ") + ", " + points.length + " archived entries");
    var root = svg.append("g").attr("transform", "translate(" + margin.left + ",0)");
    var axis = svg.append("g")
      .attr("transform", "translate(" + margin.left + "," + (height - margin.bottom) + ")")
      .call(d3.axisBottom(x).ticks(4, spec.scale === "log" ? "~s" : "~g"));
    axis.selectAll("text").attr("fill", color);
    axis.selectAll("line, path").attr("stroke", color);
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var links = root.selectAll("a").data(points).join("a")
      .attr("href", function (d) { return "#" + d.id; })
      .attr("aria-label", function (d) { return d.entry + " " + d.value; });
    links.append("rect")
      .attr("y", function (d) { return y(d.id); })
      .attr("height", y.bandwidth())
      .attr("x", function (d) { return box(x, spec.scale, d.value)[0]; })
      .attr("width", function (d) {
        return (!animate || reduce) ? box(x, spec.scale, d.value)[1] : 0;
      })
      .attr("fill", function (d) { return d.best ? accent : color; })
      .attr("fill-opacity", function (d) {
        if (d.held) { return 0.35; }
        return d.best ? 0.95 : 0.55;
      });
    if (animate && !reduce) {
      links.select("rect").transition().duration(520).ease(d3.easeCubicOut)
        .attr("width", function (d) { return box(x, spec.scale, d.value)[1]; });
    }
    links.append("text")
      .attr("class", "value")
      .attr("x", function (d) {
        var geom = box(x, spec.scale, d.value);
        return geom[0] + geom[1] + 6;
      })
      .attr("y", function (d) { return y(d.id) + y.bandwidth() / 2; })
      .attr("dy", "0.32em")
      .attr("fill", color)
      .text(function (d) { return d3.format(spec.scale === "log" ? ".3~s" : ".4~g")(d.value); });
    root.selectAll("text.label").data(points).join("text")
      .attr("class", "label")
      .attr("x", -8)
      .attr("y", function (d) { return y(d.id) + y.bandwidth() / 2; })
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .attr("fill", color)
      .text(function (d) {
        return d.entry.length > 32 ? d.entry.slice(0, 30) + "…" : d.entry;
      });
  }

  function drawAll(animate) {
    document.querySelectorAll(".bench").forEach(function (section) {
      draw(section, animate);
    });
  }

  function start() {
    drawAll(true);
    var timer = 0;
    window.addEventListener("resize", function () {
      window.clearTimeout(timer);
      timer = window.setTimeout(function () { drawAll(false); }, 120);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
}());
