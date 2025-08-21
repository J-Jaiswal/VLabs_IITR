/*  Earthquake Location Simulator — Improved
 *  -------------------------------------------------------------
 *  What’s new (vs original):
 *   - NEW: Velocity slider drives synthetic “true” v, separate v_start for inversion.
 *   - NEW: Damped least squares (Tikhonov) solved via SVD. Condition number shown.
 *   - NEW: 1-σ covariance ellipse for (x,y) drawn on map.
 *   - NEW: Stations are draggable; Add mode (click to add); Alt/Option+click to remove.
 *   - NEW: Noise σ control and optional outlier injection.
 *   - NEW: Misfit map with per-grid optimized t0 (v fixed to v_start) + colorbar.
 *   - NEW: Input validation and clearer UI states; removed artificial delays.
 *
 *  Dependencies: D3 v7, numeric.js 1.2.6
 */

(function () {
  // -----------------------------
  // Globals / State
  // -----------------------------
  let stations = []; // [{x,y,z}]
  let eventTrue = { x: 10, y: 5, z: 0 };
  let vTrue = 5.0; // km/s from slider
  let vStart = 4.0; // km/s initial guess for inversion
  let sigma = 0.1; // s (Gaussian noise)
  let outlier = "none"; // "none" | "mild" | "strong"
  let lambda = 0.1; // damping
  let addMode = false; // add-station mode

  const maxIter = 8; // a few Gauss-Newton steps
  const mapSel = d3.select("#map");
  const misfitSel = d3.select("#misfit");
  const convSel = d3.select("#conv");

  // Axes ranges (keep symmetric for nicer views)
  const XY = { min: -60, max: 60 };
  const Z_FIXED = 0; // we keep stations at z=0; eventTrue.z can vary (3-D inversion still possible)

  // -----------------------------
  // DOM elements
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
    outlier: document.getElementById("outlier"),
    lambda: document.getElementById("lambda"),
    btnGenerate: document.getElementById("btnGenerate"),
    btnRun: document.getElementById("btnRun"),
    btnReset: document.getElementById("btnReset"),
    toggleAdd: document.getElementById("toggleAdd"),
    kIter: document.getElementById("kIter"),
    kRMS: document.getElementById("kRMS"),
    kCond: document.getElementById("kCond"),
    kLambda: document.getElementById("kLambda"),
  };

  // Bind slider label
  el.vTrue.addEventListener("input", () => {
    el.vTrueLbl.textContent = Number(el.vTrue.value).toFixed(1);
  });

  // -----------------------------
  // Utilities (math & helpers)
  // -----------------------------
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // Travel time in homogeneous medium
  function travelTime(xr, yr, zr, xs, ys, zs, t0, v) {
    const dx = xr - xs,
      dy = yr - ys,
      dz = zr - zs;
    const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return t0 + R / v;
  }

  function rand(min, max) {
    return min + Math.random() * (max - min);
  }

  function gaussianNoise(std) {
    // Box-Muller
    const u = Math.random() || 1e-12;
    const v = Math.random() || 1e-12;
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
  }

  function extentXY(points) {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }

  // Damped least-squares via SVD: Δm = V * diag(s/(s^2+λ^2)) * U^T * Δd
  function solveDampedSVD(G, delta_d, lambda) {
    const svd = numeric.svd(G); // {U, S, V}
    const U = svd.U,
      S = svd.S,
      V = svd.V;
    const Ut_d = numeric.dot(numeric.transpose(U), delta_d);
    const filt = S.map((s) => (s !== 0 ? s / (s * s + lambda * lambda) : 0));
    const y = new Array(filt.length);
    for (let i = 0; i < filt.length; i++) y[i] = Ut_d[i] * filt[i];
    const delta_m = numeric.dot(V, y);
    const cond =
      S.length > 1 && Math.min(...S.filter((s) => s > 0)) > 0
        ? Math.max(...S) / Math.min(...S.filter((s) => s > 0))
        : Infinity;
    return { delta_m, cond, svd };
  }

  // Covariance ≈ σ^2 * V diag(1/(s^2 + λ^2)) V^T (reporting 1-σ)
  function covarianceFromSVD(svd, sigma, lambda) {
    const V = svd.V,
      S = svd.S;
    const diag = S.map((s) => 1.0 / (s * s + lambda * lambda));
    const VD = numeric.dot(V, numeric.diag(diag));
    const cov = numeric.dot(VD, numeric.transpose(V));
    // scale by noise variance:
    return numeric.mul(cov, sigma * sigma);
  }

  // Eigen decomposition of 2x2 symmetric matrix; returns {vals:[λ1,λ2], vecs:[[vx1,vy1],[vx2,vy2]]}
  function eig2x2(a, b, c, d) {
    const tr = a + d;
    const det = a * d - b * c;
    const disc = Math.sqrt(Math.max(0, tr * tr - 4 * det));
    const l1 = (tr + disc) / 2;
    const l2 = (tr - disc) / 2;
    // eigenvector for l1: (b, l1 - a)
    let v1 = [b, l1 - a];
    if (Math.abs(v1[0]) < 1e-12 && Math.abs(v1[1]) < 1e-12) v1 = [l1 - d, c];
    const n1 = Math.hypot(v1[0], v1[1]) || 1;
    v1 = [v1[0] / n1, v1[1] / n1];
    // v2 orthogonal to v1
    const v2 = [-v1[1], v1[0]];
    return { vals: [l1, l2], vecs: [v1, v2] };
  }

  // -----------------------------
  // Data generation & validation
  // -----------------------------
  function randomStations(n) {
    const stas = [];
    for (let i = 0; i < n; i++) {
      stas.push({
        x: rand(XY.min, XY.max),
        y: rand(XY.min, XY.max),
        z: Z_FIXED,
      });
    }
    return stas;
  }

  function validateInputs() {
    const n = +el.numStations.value;
    const ex = +el.xInput.value,
      ey = +el.yInput.value,
      ez = +el.zInput.value;
    const vt = +el.vTrue.value,
      vs = +el.vStart.value,
      sd = +el.sigma.value,
      lam = +el.lambda.value;
    if ([ex, ey, ez, vt, vs, sd, lam].some((v) => Number.isNaN(v))) {
      alert(
        "Please enter numeric values for event location, velocities, sigma, and lambda."
      );
      return false;
    }
    if (stations.length < 4) {
      alert(
        "Need at least 4 stations for a stable solution. Use Generate or Add mode to place stations."
      );
      return false;
    }
    return true;
  }

  // Create synthetic observations t_obs with noise and optional outlier
  function makeObservations(event, vTrue, sigma, outlierMode) {
    const t0True = 0.0; // we can fix true origin time to 0
    let tObs = stations.map((s) =>
      travelTime(s.x, s.y, s.z, event.x, event.y, event.z, t0True, vTrue)
    );
    // Add Gaussian noise
    if (sigma > 0) tObs = tObs.map((t) => t + gaussianNoise(sigma));
    // Inject one outlier if requested
    if (outlierMode !== "none" && stations.length > 0) {
      const idx = Math.floor(Math.random() * stations.length);
      const k = outlierMode === "strong" ? 6 : 3;
      tObs[idx] += k * sigma * (Math.random() < 0.5 ? -1 : 1);
    }
    return { tObs, t0True };
  }

  // -----------------------------
  // Inversion (x, y, z, t0, v)
  // -----------------------------
  function invert(tObs, start, lambda, sigma) {
    let m = [start.x, start.y, start.z, start.t0, start.v]; // [xs,ys,zs,t0,v]
    const hist = []; // residual per iteration
    let condReport = "—";
    let lastCov = null;

    for (let it = 0; it < maxIter; it++) {
      // Forward at current m
      const tPred = stations.map((s) =>
        travelTime(s.x, s.y, s.z, m[0], m[1], m[2], m[3], m[4])
      );
      const delta = numeric.sub(tObs, tPred); // residuals t_obs - t_pred
      const rms = Math.sqrt(
        numeric.sum(delta.map((d) => d * d)) / delta.length
      );
      hist.push(rms);

      // Build Jacobian G (N x 5)
      const G = stations.map((s) => {
        const dx = m[0] - s.x,
          dy = m[1] - s.y,
          dz = m[2] - s.z;
        const R = Math.hypot(dx, dy, dz) || 1e-12;
        const v = m[4];
        return [
          dx / (v * R), // ∂t/∂x_s
          dy / (v * R), // ∂t/∂y_s
          dz / (v * R), // ∂t/∂z_s
          1.0, // ∂t/∂t0
          -R / (v * v), // ∂t/∂v
        ];
      });

      // Weighted by sigma (if constant, just scale)
      const invSigma = sigma > 0 ? 1 / sigma : 1;
      const Gw = G.map((row) => row.map((e) => e * invSigma));
      const dw = delta.map((d) => d * invSigma);

      // Solve with damping via SVD
      const { delta_m, cond, svd } = solveDampedSVD(Gw, dw, lambda);
      condReport = cond;
      lastCov = covarianceFromSVD(svd, sigma, lambda);

      // Update model
      m = numeric.add(m, delta_m);

      // Early stop if tiny update
      if (numeric.norm2(delta_m) < 1e-4) break;
    }
    return { m, hist, cond: condReport, cov: lastCov };
  }

  // -----------------------------
  // Misfit grid (forward scan)
  // -----------------------------
  function misfitGrid(tObs, vFixed, gridN = 60) {
    // For each gridpoint (x,y), we optimize t0 in closed form: t0* = mean(t_obs - R/v)
    const xs = d3.scaleLinear().domain([XY.min, XY.max]).ticks(gridN);
    const ys = d3.scaleLinear().domain([XY.min, XY.max]).ticks(gridN);
    const Z = [];
    let min = Infinity,
      max = -Infinity;

    for (let j = 0; j < ys.length; j++) {
      const row = [];
      for (let i = 0; i < xs.length; i++) {
        const x = xs[i],
          y = ys[j];
        const Rv = stations.map((s) => {
          const R = Math.hypot(x - s.x, y - s.y, 0 - s.z);
          return R / vFixed;
        });
        const t0star = d3.mean(tObs.map((t, k) => t - Rv[k]));
        const tPred = Rv.map((rv) => t0star + rv);
        const res = tObs.map((t, k) => t - tPred[k]);
        const ss = d3.sum(res.map((r) => r * r));
        row.push(ss / stations.length);
        if (row[i] < min) min = row[i];
        if (row[i] > max) max = row[i];
      }
      Z.push(row);
    }
    return { xs, ys, Z, min, max };
  }

  // -----------------------------
  // Drawing: Map, Misfit, Convergence
  // -----------------------------
  function drawMap(est, cov) {
    const svg = mapSel;
    svg.selectAll("*").remove();
    const w = +svg.attr("width"),
      h = +svg.attr("height");
    const pad = 28;

    const x = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([pad, w - pad]);
    const y = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([h - pad, pad]);

    // Axes
    svg
      .append("g")
      .attr("transform", `translate(0,${h - pad})`)
      .call(d3.axisBottom(x).ticks(7));
    svg
      .append("g")
      .attr("transform", `translate(${pad},0)`)
      .call(d3.axisLeft(y).ticks(7));

    // Stations
    const gStations = svg.append("g");

    const drag = d3
      .drag()
      .on("start", function (event, d) {
        d3.select(this).raise().classed("active", true);
      })
      .on("drag", function (event, d) {
        const nx = clamp(x.invert(event.x), XY.min, XY.max);
        const ny = clamp(y.invert(event.y), XY.min, XY.max);
        d.x = nx;
        d.y = ny;
        d3.select(this).attr("cx", x(d.x)).attr("cy", y(d.y));
      })
      .on("end", function () {
        d3.select(this).classed("active", false);
      });

    gStations
      .selectAll("circle.sta")
      .data(stations, (d, i) => i)
      .join("circle")
      .attr("class", "sta")
      .attr("r", 5)
      .attr("cx", (d) => x(d.x))
      .attr("cy", (d) => y(d.y))
      .attr("fill", "#111")
      .style("cursor", "grab")
      .on("click", (event, d) => {
        if (event.altKey) {
          // Alt/Option+click removes a station
          stations = stations.filter((s) => s !== d);
          drawMap(est, cov);
        }
      })
      .call(drag);

    // Add mode click-handle
    svg.on("click", (event) => {
      if (!addMode) return;
      const [px, py] = d3.pointer(event);
      const nx = clamp(x.invert(px), XY.min, XY.max);
      const ny = clamp(y.invert(py), XY.min, XY.max);
      stations.push({ x: nx, y: ny, z: Z_FIXED });
      drawMap(est, cov);
    });

    // True event
    svg
      .append("circle")
      .attr("r", 6)
      .attr("cx", x(eventTrue.x))
      .attr("cy", y(eventTrue.y))
      .attr("fill", "#e11d48")
      .attr("stroke", "white")
      .attr("stroke-width", 1.2);

    // Estimated event (if available)
    if (est) {
      svg
        .append("circle")
        .attr("r", 5)
        .attr("cx", x(est[0]))
        .attr("cy", y(est[1]))
        .attr("fill", "#0ea5e9")
        .attr("stroke", "white")
        .attr("stroke-width", 1.2);
    }

    // Uncertainty ellipse (1-σ on x,y) from covariance (5x5)
    if (cov) {
      const a = cov[0][0],
        b = cov[0][1],
        c = cov[1][0],
        d = cov[1][1];
      const eig = eig2x2(a, b, c, d);
      const [l1, l2] = eig.vals; // variances along principal axes
      const [v1] = eig.vecs;
      const rx = Math.sqrt(Math.max(0, l1));
      const ry = Math.sqrt(Math.max(0, l2));
      const angle = Math.atan2(v1[1], v1[0]); // radians

      // ellipse in screen coords
      svg
        .append("g")
        .attr(
          "transform",
          `translate(${x(est[0])},${y(est[1])}) rotate(${
            (angle * 180) / Math.PI
          })`
        )
        .append("ellipse")
        .attr("rx", Math.abs(x(est[0] + rx) - x(est[0])))
        .attr("ry", Math.abs(y(est[1]) - y(est[1] + ry)))
        .attr("fill", "none")
        .attr("stroke", "#a78bfa")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4 3");
    }
  }

  function drawMisfit(tObs) {
    const svg = misfitSel;
    svg.selectAll("*").remove();
    const w = +svg.attr("width"),
      h = +svg.attr("height");
    const pad = 28;

    const x = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([pad, w - pad]);
    const y = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([h - pad, pad]);

    // Axes
    svg
      .append("g")
      .attr("transform", `translate(0,${h - pad})`)
      .call(d3.axisBottom(x).ticks(7));
    svg
      .append("g")
      .attr("transform", `translate(${pad},0)`)
      .call(d3.axisLeft(y).ticks(7));

    // Compute grid
    const vFixed = +el.vStart.value; // NEW: keep v fixed to current "assumed" start velocity
    const { xs, ys, Z, min, max } = misfitGrid(tObs, vFixed, 60);

    // Color scale + colorbar
    const color = d3.scaleSequential(d3.interpolateViridis).domain([max, min]); // low misfit = bright
    const cellW = x(xs[1]) - x(xs[0]) || 6;
    const cellH = y(ys[ys.length - 2]) - y(ys[ys.length - 1]) || 6;

    const g = svg.append("g");
    for (let j = 0; j < ys.length; j++) {
      for (let i = 0; i < xs.length; i++) {
        g.append("rect")
          .attr("x", x(xs[i]) - cellW / 2)
          .attr("y", y(ys[j]) - cellH / 2)
          .attr("width", cellW)
          .attr("height", cellH)
          .attr("fill", color(Z[j][i]));
      }
    }

    // Colorbar
    const cbW = 12,
      cbH = 160;
    const cbX = w - 50,
      cbY = 40;
    const cb = svg.append("g").attr("transform", `translate(${cbX},${cbY})`);
    const cbScale = d3.scaleLinear().domain([min, max]).range([cbH, 0]);
    const cbGrad = svg
      .append("defs")
      .append("linearGradient")
      .attr("id", "cb")
      .attr("x1", "0")
      .attr("x2", "0")
      .attr("y1", "1")
      .attr("y2", "0");
    for (let i = 0; i <= 100; i++) {
      cbGrad
        .append("stop")
        .attr("offset", `${i}%`)
        .attr("stop-color", color(min + ((max - min) * i) / 100));
    }
    cb.append("rect")
      .attr("width", cbW)
      .attr("height", cbH)
      .style("fill", "url(#cb)")
      .style("stroke", "#ccc");
    cb.append("g")
      .attr("transform", `translate(${cbW + 6},0)`)
      .attr("class", "colorbar-axis")
      .call(d3.axisRight(cbScale).ticks(5));

    // Overlays
    // Stations
    svg
      .append("g")
      .selectAll("circle")
      .data(stations)
      .join("circle")
      .attr("r", 3.5)
      .attr("fill", "#111")
      .attr("cx", (d) => x(d.x))
      .attr("cy", (d) => y(d.y));

    // True event
    svg
      .append("circle")
      .attr("r", 4)
      .attr("cx", x(eventTrue.x))
      .attr("cy", y(eventTrue.y))
      .attr("fill", "#e11d48")
      .attr("stroke", "white")
      .attr("stroke-width", 1);
  }

  function drawConvergence(hist) {
    const svg = convSel;
    svg.selectAll("*").remove();
    const w = +svg.attr("width"),
      h = +svg.attr("height");
    const pad = 36;

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
      .attr("stroke", "#0ea5e9")
      .attr("stroke-width", 2)
      .attr("d", line);
    svg
      .append("g")
      .selectAll("circle")
      .data(hist)
      .join("circle")
      .attr("cx", (d, i) => x(i))
      .attr("cy", (d) => y(d))
      .attr("r", 3)
      .attr("fill", "#0ea5e9");
  }

  // -----------------------------
  // App flow
  // -----------------------------
  function generateStations() {
    const n = +el.numStations.value;
    stations = randomStations(n);
    drawMap(null, null);
  }

  function run() {
    if (!validateInputs()) return;

    // sync state from UI
    eventTrue = {
      x: +el.xInput.value,
      y: +el.yInput.value,
      z: +el.zInput.value,
    };
    vTrue = +el.vTrue.value;
    vStart = +el.vStart.value;
    sigma = Math.max(0, +el.sigma.value);
    outlier = el.outlier.value;
    lambda = Math.max(0, +el.lambda.value);

    // Synthetic observations
    const { tObs } = makeObservations(eventTrue, vTrue, sigma, outlier);

    // Initial guess
    const start = {
      x: d3.mean(stations, (s) => s.x) || 0,
      y: d3.mean(stations, (s) => s.y) || 0,
      z: 0,
      t0: 0,
      v: vStart,
    };

    // Inversion
    const { m, hist, cond, cov } = invert(tObs, start, lambda, sigma);

    // Render
    drawMap(m, cov);
    drawMisfit(tObs);
    drawConvergence(hist);

    // KPIs
    el.kIter.textContent = String(hist.length);
    el.kRMS.textContent = (hist.at(-1) ?? NaN).toFixed(3);
    el.kCond.textContent = cond === Infinity ? "∞" : cond.toFixed(1);
    el.kLambda.textContent = lambda.toFixed(2);
  }

  function resetAll() {
    stations = [];
    eventTrue = { x: 10, y: 5, z: 0 };
    el.xInput.value = 10;
    el.yInput.value = 5;
    el.zInput.value = 0;
    el.vTrue.value = 5.0;
    el.vTrueLbl.textContent = "5.0";
    el.vStart.value = 4.0;
    el.sigma.value = 0.1;
    el.outlier.value = "none";
    el.lambda.value = 0.1;
    el.numStations.value = 6;
    drawMap(null, null);
    misfitSel.selectAll("*").remove();
    convSel.selectAll("*").remove();
    el.kIter.textContent = "—";
    el.kRMS.textContent = "—";
    el.kCond.textContent = "—";
    el.kLambda.textContent = "—";
  }

  // -----------------------------
  // Events
  // -----------------------------
  el.btnGenerate.addEventListener("click", generateStations);
  el.btnRun.addEventListener("click", run);
  el.btnReset.addEventListener("click", resetAll);

  el.toggleAdd.addEventListener("click", () => {
    addMode = !addMode;
    el.toggleAdd.textContent = addMode ? "ON" : "OFF";
    el.toggleAdd.classList.toggle("btn-primary", addMode);
    el.toggleAdd.classList.toggle("btn-toggle", !addMode);
  });

  // Init
  resetAll();
  generateStations();
})();
