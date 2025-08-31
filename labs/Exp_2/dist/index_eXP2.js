/* Earthquake Location — Tikhonov (no SVD), now inverts Z too
 * -----------------------------------------------------------
 * - Inverts [x, y, z, t0, v]  (depth is NOT fixed anymore)
 * - Stations: triangles; True/Estimated event: star
 * - One synthetic outlier (Mild=3σ, Strong=6σ) and highlight EXACTLY that station
 * - Draws map (truth+estimate), misfit heatmap, RMS vs iteration
 * Deps: D3 v7, numeric.js
 */

(function () {
  // -----------------------------
  // Globals / State
  // -----------------------------
  let stations = []; // [{x,y,z,_isOutlier?}]
  let eventTrue = { x: 10, y: 5, z: 8 };
  let vTrue = 5.0; // km/s (true)
  let vStart = 4.0; // km/s (misfit grid + start)
  let sigma = 0.1; // s, Gaussian noise std
  let lambda = 0.1; // Tikhonov damping
  const maxIter = 8;
  const XY = { min: -60, max: 60 }; // map/misfit extent
  let injectedOutlierIndex = null; // which station got the ±kσ hit
  let highlightIdx = new Set(); // indices to highlight (amber)

  const C = {
    station: "#111111",
    outlier: "#f59e0b", // amber
    trueEvt: "#e11d48",
    estEvt: "#0ea5e9",
  };

  // SVGs
  const mapSel = d3.select("#map");
  const misfitSel = d3.select("#misfit");
  const convSel = d3.select("#conv");

  // -----------------------------
  // UI handles (ids from HTML)
  // -----------------------------
  const el = {
    numStations: document.getElementById("numStations"),
    xInput: document.getElementById("xInput"),
    yInput: document.getElementById("yInput"),
    zInput: document.getElementById("zInput"),
    vTrue: document.getElementById("vTrue"),
    vTrueLbl: document.getElementById("vTrueLbl"),
    vStart: document.getElementById("vStart"),
    sigma: document.getElementById("sigma"),
    lambda: document.getElementById("lambda"),
    outlier: document.getElementById("outlier"), // None/Mild/Strong
    btnGenerate: document.getElementById("btnGenerate"),
    btnRun: document.getElementById("btnRun"),
    btnReset: document.getElementById("btnReset"),
    kIter: document.getElementById("kIter"),
    kRMS: document.getElementById("kRMS"),
    kCond: document.getElementById("kCond"), // will be "—"
    kLambda: document.getElementById("kLambda"),
  };

  el.vTrue?.addEventListener("input", () => {
    el.vTrueLbl.textContent = Number(el.vTrue.value).toFixed(1);
  });

  // -----------------------------
  // Utilities
  // -----------------------------
  const rand = (a, b) => a + Math.random() * (b - a);

  function gaussian(std) {
    if (std <= 0) return 0;
    const u = Math.random() || 1e-12,
      v = Math.random() || 1e-12;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
  }

  // Travel time (homogeneous)
  function travelTime(xr, yr, zr, xs, ys, zs, t0, v) {
    const dx = xr - xs,
      dy = yr - ys,
      dz = zr - zs;
    const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return t0 + R / v;
  }

  function randomStations(n) {
    return Array.from({ length: n }, () => ({
      x: rand(XY.min, XY.max),
      y: rand(XY.min, XY.max),
      z: 0,
      _isOutlier: false,
    }));
  }

  function parseOutlierLevel() {
    const val = (el.outlier?.value || "none").toLowerCase();
    if (val.includes("mild") || val.includes("3")) return 3;
    if (val.includes("strong") || val.includes("6")) return 6;
    return 0;
  }

  // -----------------------------
  // Observations (noise + optional outlier)
  // -----------------------------
  function makeObservations(evt, vTrue, sigma, k) {
    const t0True = 0;
    let tObs = stations.map((s) =>
      travelTime(s.x, s.y, s.z, evt.x, evt.y, evt.z, t0True, vTrue)
    );
    tObs = tObs.map((t) => t + gaussian(sigma));

    // clear flags
    stations.forEach((s) => (s._isOutlier = false));
    injectedOutlierIndex = null;

    // inject ±kσ outlier at one station
    if (k > 0 && stations.length) {
      const i = Math.floor(Math.random() * stations.length);
      const sign = Math.random() < 0.5 ? -1 : 1;
      tObs[i] += sign * k * sigma;
      stations[i]._isOutlier = true;
      injectedOutlierIndex = i;
      console.log(`[Outlier] Station #${i} got ${sign > 0 ? "+" : "-"}${k}σ`);
    }
    return tObs;
  }

  // -----------------------------
  // Normal-Equations Solver (NO SVD)
  // Solves: (Gwᵀ Gw + λ² I) Δm = Gwᵀ dw
  // -----------------------------
  function solveNormalEq(Gw, dw, lambda) {
    const GT = numeric.transpose(Gw); // (P x N)
    let AtA = numeric.dot(GT, Gw); // (P x P)
    const P = AtA.length;
    for (let i = 0; i < P; i++) AtA[i][i] += lambda * lambda;
    const Atb = numeric.dot(GT, dw); // (P)
    const delta_m = numeric.solve(AtA, Atb); // LU solve
    return { delta_m };
  }

  // -----------------------------
  // Inversion for [x, y, z, t0, v]  (NO SVD)
  // -----------------------------
  function invertXYZT0V(tObs, start, lambda, sigma) {
    // m = [xs, ys, zs, t0, v]
    let m = [start.x, start.y, start.z, start.t0, start.v];
    const hist = [];

    for (let it = 0; it < maxIter; it++) {
      // forward
      const tPred = stations.map((s) =>
        travelTime(s.x, s.y, s.z, m[0], m[1], m[2], m[3], m[4])
      );
      const delta = numeric.sub(tObs, tPred);
      const rms = Math.sqrt(
        numeric.sum(delta.map((d) => d * d)) / delta.length
      );
      hist.push(rms);

      // Jacobian (N x 5)
      const G = stations.map((s) => {
        const dx = m[0] - s.x,
          dy = m[1] - s.y,
          dz = m[2] - s.z;
        const R = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-12;
        const v = m[4];
        // [∂t/∂xs, ∂t/∂ys, ∂t/∂zs, ∂t/∂t0, ∂t/∂v]
        return [dx / (v * R), dy / (v * R), dz / (v * R), 1.0, -R / (v * v)];
      });

      // simple whitening with σ (scalar)
      const w = sigma > 0 ? 1 / sigma : 1;
      const Gw = G.map((row) => row.map((e) => e * w));
      const dw = delta.map((d) => d * w);

      // Solve normal equations (Tikhonov)
      const { delta_m } = solveNormalEq(Gw, dw, lambda);

      // Update
      m = numeric.add(m, delta_m);
      if (numeric.norm2(delta_m) < 1e-4) break;
    }
    // No SVD → no condition number
    return { m, hist, cond: "—" };
  }

  // -----------------------------
  // Single-station highlighter: ONLY the injected outlier
  // -----------------------------
  function setSingleHighlight(k) {
    highlightIdx.clear();
    if (k > 0 && injectedOutlierIndex != null) {
      highlightIdx.add(injectedOutlierIndex);
    }
  }

  // -----------------------------
  // Misfit grid (x,y), t0 optimized, v fixed to vStart
  // Uses current true Z to keep the 2D view (rest unchanged)
  // -----------------------------
  function misfitGrid(tObs, vFixed, N = 60) {
    const xs = d3.scaleLinear().domain([XY.min, XY.max]).ticks(N);
    const ys = d3.scaleLinear().domain([XY.min, XY.max]).ticks(N);
    const Z = [];
    let zmin = Infinity,
      zmax = -Infinity;
    for (let j = 0; j < ys.length; j++) {
      const row = [];
      for (let i = 0; i < xs.length; i++) {
        const x = xs[i],
          y = ys[j];

        // include current eventTrue.z so depth participates in forward scan
        const R_over_v = stations.map(
          (s) =>
            Math.sqrt(
              (x - s.x) ** 2 + (y - s.y) ** 2 + (eventTrue.z - s.z) ** 2
            ) / vFixed
        );
        const t0star = d3.mean(tObs.map((t, k) => t - R_over_v[k])); // analytic optimum for t0
        const val = d3.mean(
          tObs.map((t, k) => {
            const r = t - (t0star + R_over_v[k]);
            return r * r;
          })
        );
        row.push(val);
        if (val < zmin) zmin = val;
        if (val > zmax) zmax = val;
      }
      Z.push(row);
    }
    return { xs, ys, Z, zmin, zmax };
  }

  // -----------------------------
  // Drawing helpers
  // -----------------------------
  function getWH(svgSel) {
    const w = +svgSel.attr("width") || svgSel.node().clientWidth || 600;
    const h = +svgSel.attr("height") || svgSel.node().clientHeight || 400;
    return { w, h };
  }

  // Reusable symbol generators
  const tri = d3.symbol().type(d3.symbolTriangle).size(120);
  const triSmall = d3.symbol().type(d3.symbolTriangle).size(70);
  const star = d3.symbol().type(d3.symbolStar).size(240);
  const starSmall = d3.symbol().type(d3.symbolStar).size(140);

  function drawMap(est) {
    const svg = mapSel;
    svg.selectAll("*").remove();
    const { w, h } = getWH(svg),
      pad = 28;
    const x = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([pad, w - pad]);
    const y = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([h - pad, pad]);

    svg
      .append("g")
      .attr("transform", `translate(0,${h - pad})`)
      .call(d3.axisBottom(x).ticks(7));
    svg
      .append("g")
      .attr("transform", `translate(${pad},0)`)
      .call(d3.axisLeft(y).ticks(7));

    // stations (triangles)
    svg
      .append("g")
      .selectAll("path.station")
      .data(
        stations.map((s, i) => ({ ...s, i })),
        (d) => d.i
      )
      .join("path")
      .attr("class", "station")
      .attr("d", tri())
      .attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`)
      .attr("fill", (d) => (highlightIdx.has(d.i) ? C.outlier : C.station));

    // outlier ring(s) on top
    const gRing = svg.append("g").attr("pointer-events", "none");
    highlightIdx.forEach((i) => {
      const s = stations[i];
      gRing
        .append("circle")
        .attr("cx", x(s.x))
        .attr("cy", y(s.y))
        .attr("r", 10)
        .attr("fill", "none")
        .attr("stroke", C.outlier)
        .attr("stroke-width", 3);
    });

    // truth (star)
    svg
      .append("path")
      .attr("d", star())
      .attr("transform", `translate(${x(eventTrue.x)},${y(eventTrue.y)})`)
      .attr("fill", C.trueEvt)
      .attr("stroke", "white")
      .attr("stroke-width", 1.2);

    // estimate (star)
    if (est) {
      svg
        .append("path")
        .attr("d", starSmall())
        .attr("transform", `translate(${x(est[0])},${y(est[1])})`)
        .attr("fill", C.estEvt)
        .attr("stroke", "white")
        .attr("stroke-width", 1.2);
    }
  }

  function drawMisfit(tObs) {
    const svg = misfitSel;
    svg.selectAll("*").remove();
    const { w, h } = getWH(svg),
      pad = 28;
    const x = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([pad, w - pad]);
    const y = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([h - pad, pad]);

    svg
      .append("g")
      .attr("transform", `translate(0,${h - pad})`)
      .call(d3.axisBottom(x).ticks(7));
    svg
      .append("g")
      .attr("transform", `translate(${pad},0)`)
      .call(d3.axisLeft(y).ticks(7));

    const { xs, ys, Z, zmin, zmax } = misfitGrid(tObs, vStart, 60);
    const color = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([zmax, zmin]);
    const cellW = x(xs[1]) - x(xs[0]) || 6;
    const cellH = y(ys[ys.length - 2]) - y(ys[ys.length - 1]) || 6;

    const gHeat = svg.append("g");
    for (let j = 0; j < ys.length; j++) {
      for (let i = 0; i < xs.length; i++) {
        gHeat
          .append("rect")
          .attr("x", x(xs[i]) - cellW / 2)
          .attr("y", y(ys[j]) - cellH / 2)
          .attr("width", cellW)
          .attr("height", cellH)
          .attr("fill", color(Z[j][i]));
      }
    }

    // stations (triangles) overlaid
    svg
      .append("g")
      .selectAll("path.station")
      .data(
        stations.map((s, i) => ({ ...s, i })),
        (d) => d.i
      )
      .join("path")
      .attr("class", "station")
      .attr("d", triSmall())
      .attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`)
      .attr("fill", (d) => (highlightIdx.has(d.i) ? C.outlier : C.station));

    // ring(s)
    const gRing = svg.append("g").attr("pointer-events", "none");
    highlightIdx.forEach((i) => {
      const s = stations[i];
      gRing
        .append("circle")
        .attr("cx", x(s.x))
        .attr("cy", y(s.y))
        .attr("r", 8)
        .attr("fill", "none")
        .attr("stroke", C.outlier)
        .attr("stroke-width", 2.5);
    });

    // truth (star)
    svg
      .append("path")
      .attr("d", starSmall())
      .attr("transform", `translate(${x(eventTrue.x)},${y(eventTrue.y)})`)
      .attr("fill", C.trueEvt)
      .attr("stroke", "white")
      .attr("stroke-width", 1);
  }

  function drawConvergence(hist) {
    const svg = convSel;
    svg.selectAll("*").remove();
    const { w, h } = getWH(svg),
      pad = 36;
    const x = d3
      .scaleLinear()
      .domain([0, Math.max(1, hist.length - 1)])
      .range([pad, w - pad]);
    const y = d3
      .scaleLinear()
      .domain([0, d3.max(hist) || 1])
      .nice()
      .range([h - pad, pad]);

    svg
      .append("g")
      .attr("transform", `translate(0,${h - pad})`)
      .call(d3.axisBottom(x).ticks(hist.length));
    svg
      .append("g")
      .attr("transform", `translate(${pad},0)`)
      .call(d3.axisLeft(y));

    const line = d3
      .line()
      .x((d, i) => x(i))
      .y((d) => y(d));
    svg
      .append("path")
      .datum(hist)
      .attr("fill", "none")
      .attr("stroke", C.estEvt)
      .attr("stroke-width", 2)
      .attr("d", line);
    svg
      .append("g")
      .selectAll("circle")
      .data(hist)
      .join("circle")
      .attr("r", 3)
      .attr("fill", C.estEvt)
      .attr("cx", (d, i) => x(i))
      .attr("cy", (d) => y(d));
  }

  // -----------------------------
  // App flow
  // -----------------------------
  function validate() {
    const n = +el.numStations.value;
    if (!Number.isFinite(n) || n < 4) {
      alert("Need at least 4 stations.");
      return false;
    }
    return true;
  }

  function generate() {
    const n = +el.numStations.value || 6;
    stations = randomStations(n);
    highlightIdx.clear();
    injectedOutlierIndex = null;
    drawMap(null);
  }

  function run() {
    if (!validate()) return;

    // read UI
    eventTrue = {
      x: +el.xInput.value,
      y: +el.yInput.value,
      z: +el.zInput.value,
    };
    vTrue = +el.vTrue.value;
    vStart = +el.vStart.value;
    sigma = Math.max(0, +el.sigma.value);
    lambda = Math.max(0, +el.lambda.value);
    const k = parseOutlierLevel(); // 0/3/6

    // forward: observations
    const tObs = makeObservations(eventTrue, vTrue, sigma, k);

    // start model (keep simple like before; z starts at 0)
    const start = {
      x: d3.mean(stations, (s) => s.x) || 0,
      y: d3.mean(stations, (s) => s.y) || 0,
      z: 0,
      t0: 0,
      v: vStart,
    };

    // invert (no SVD)
    const { m, hist, cond } = invertXYZT0V(tObs, start, lambda, sigma);

    // highlight EXACTLY the injected station (if any)
    setSingleHighlight(k);

    // draw
    drawMap(m);
    drawMisfit(tObs);
    drawConvergence(hist);

    // KPIs
    el.kIter.textContent = String(hist.length);
    el.kRMS.textContent = (hist.at(-1) ?? NaN).toFixed(3);
    el.kCond.textContent = "—"; // no SVD cond
    el.kLambda.textContent = lambda.toFixed(2);

    console.log({
      stations,
      injectedOutlierIndex,
      highlightIdx: [...highlightIdx],
      finalModel: {
        xs: m[0],
        ys: m[1],
        zs: m[2],
        t0: m[3],
        v: m[4],
      },
      rmsHistory: hist,
    });
  }

  function reset() {
    stations = [];
    eventTrue = { x: 10, y: 5, z: 0 };
    highlightIdx.clear();
    injectedOutlierIndex = null;
    if (el.xInput) el.xInput.value = 10;
    if (el.yInput) el.yInput.value = 5;
    if (el.zInput) el.zInput.value = 0;
    if (el.vTrue) {
      el.vTrue.value = 5.0;
      el.vTrueLbl.textContent = "5.0";
    }
    if (el.vStart) el.vStart.value = 4.0;
    if (el.sigma) el.sigma.value = 0.1;
    if (el.lambda) el.lambda.value = 0.1;
    if (el.numStations) el.numStations.value = 6;
    if (el.outlier)
      el.outlier.value = el.outlier.querySelector('option[value="None"]')
        ? "None"
        : "none";

    mapSel.selectAll("*").remove();
    misfitSel.selectAll("*").remove();
    convSel.selectAll("*").remove();
    el.kIter.textContent = "—";
    el.kRMS.textContent = "—";
    el.kCond.textContent = "—";
    el.kLambda.textContent = "—";
  }

  el.btnGenerate.addEventListener("click", generate);
  el.btnRun.addEventListener("click", run);
  el.btnReset.addEventListener("click", reset);

  // Init
  reset();
  generate();
})();
