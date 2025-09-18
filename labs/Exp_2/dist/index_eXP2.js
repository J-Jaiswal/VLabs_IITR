/* Earthquake Location — Tikhonov (with detailed console logging)
 * --------------------------------------------------------------
 * - Inverts [x, y, z, t0, v]
 * - Stations: triangles; True/Estimated event: star
 * - One synthetic outlier (Mild=3σ, Strong=6σ) and highlight EXACTLY that station
 * - Draws map (truth+estimate), misfit heatmap, RMS vs iteration, simple waveforms
 * - Deps: D3 v7, numeric.js
 *
 * Debug controls:
 *   DBG.verbose              → high level step-by-step logging (true/false)
 *   DBG.logMatrices          → log Jacobians / normal matrices each iteration
 *   DBG.logEveryGridCell     → log misfit per grid cell (⚠️ very heavy)
 *   DBG.gridSamples          → sample cells per row to log when not logging all
 */

(function () {
  // -----------------------------
  // Debug / logging helpers
  // -----------------------------
  const DBG = {
    verbose: true,
    logMatrices: true,
    logEveryGridCell: false, // WARNING: 60x60 grid = 3600 logs per run
    gridSamples: 4,
  };

  const L = {
    start(label) {
      if (DBG.verbose) console.group(label);
    },
    end() {
      if (DBG.verbose) console.groupEnd();
    },
    log(...args) {
      if (DBG.verbose) console.log(...args);
    },
    table(obj) {
      if (DBG.verbose && console.table) console.table(obj);
    },
    time(label) {
      if (DBG.verbose) console.time(label);
    },
    timeEnd(label) {
      if (DBG.verbose) console.timeEnd(label);
    },
  };

  // -----------------------------
  // Globals / State
  // -----------------------------
  let stations = []; // [{x,y,z,_isOutlier?}]
  let eventTrue = { x: 10, y: 5, z: 8 };
  let vTrue = 5.0; // km/s (true)
  let vStart = 4.0; // km/s (misfit grid + start)
  let sigma = 1.5; //  Gaussian noise std
  let lambda = 0.1; // Tikhonov damping
  const maxIter = 8;
  const XY = { min: -60, max: 60 }; // map/misfit extent
  let injectedOutlierIndex = null; // which station got the ±kσ hit
  let highlightIdx = new Set(); // indices to highlight (amber)
  let ellipseParams = null; // { cx, cy, rx, ry, thetaDeg }

  // ── Map edit mode ─────────────────────────────────────────────
  let toolMode = "add";
  const modeCursor = { add: "copy", move: "grab", remove: "not-allowed" };

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
  const wavesSel = d3.select("#waves");

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
    kLambda: document.getElementById("kLambda"),
    modeAdd: document.getElementById("modeAdd"),
    modeMove: document.getElementById("modeMove"),
    modeRemove: document.getElementById("modeRemove"),
    trueXYZ: document.getElementById("trueXYZ"),
    estXYZ: document.getElementById("estXYZ"),
  };

  el.vTrue?.addEventListener("input", () => {
    el.vTrueLbl.textContent = Number(el.vTrue.value).toFixed(1);
  });

  // -----------------------------
  // Utilities
  // -----------------------------
  const rand = (a, b) => a + Math.random() * (b - a);

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function setMode(m) {
    L.start(`[UI] Set tool mode → ${m}`);
    toolMode = m;
    [el.modeAdd, el.modeMove, el.modeRemove].forEach((btn) =>
      btn?.classList.remove("active")
    );
    if (m === "add") el.modeAdd?.classList.add("active");
    if (m === "move") el.modeMove?.classList.add("active");
    if (m === "remove") el.modeRemove?.classList.add("active");
    drawMap(null);
    L.end();
  }

  function addStationAt(xkm, ykm) {
    stations.push({ x: xkm, y: ykm, z: 0, _isOutlier: false });
    if (el.numStations) el.numStations.value = stations.length;
    L.log(`[Stations] Added at (x=${xkm.toFixed(2)}, y=${ykm.toFixed(2)})`);
    drawMap(null);
  }

  function removeStationAtIndex(i) {
    if (i < 0 || i >= stations.length) return;
    const removed = stations[i];
    stations.splice(i, 1);
    if (el.numStations) el.numStations.value = stations.length;
    L.log(
      `[Stations] Removed index ${i} @ (x=${removed.x.toFixed(
        2
      )}, y=${removed.y.toFixed(2)})`
    );
    drawMap(null);
  }

  // --- 2x2 symmetric eigen-decomp (covariance -> axes) ---
  function eig2x2(a, b, c) {
    const tr = a + c,
      det = a * c - b * b;
    const disc = Math.max(0, tr * tr - 4 * det);
    const l1 = 0.5 * (tr + Math.sqrt(disc));
    const l2 = 0.5 * (tr - Math.sqrt(disc));
    let vx = b,
      vy = l1 - a;
    if (Math.abs(vx) + Math.abs(vy) < 1e-12) {
      vx = 1;
      vy = 0;
    }
    const n = Math.hypot(vx, vy) || 1;
    return { l1, l2, ux: vx / n, uy: vy / n };
  }

  function gaussian(std) {
    if (std <= 0) return 0;
    const u = Math.random() || 1e-12,
      v = Math.random() || 1e-12;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
  }

  function randomStations(n) {
    return Array.from({ length: n }, () => ({
      x: rand(XY.min, XY.max),
      y: rand(XY.min, XY.max),
      z: 0,
      _isOutlier: false,
    }));
  }

  // Travel time (homogeneous)
  function travelTime(xr, yr, zr, xs, ys, zs, t0, v) {
    const dx = xr - xs,
      dy = yr - ys,
      dz = zr - zs;
    const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return t0 + R / v;
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
    L.start("[Obs] Generate observations");
    const t0True = 0;

    const tClean = stations.map((s, idx) => {
      const t = travelTime(s.x, s.y, s.z, evt.x, evt.y, evt.z, t0True, vTrue);
      L.log(`  • Clean t[${idx}] = ${t.toFixed(6)} s`);
      return t;
    });

    let tObs = tClean.map((t, i) => {
      const n = gaussian(sigma);
      const noisy = t + n;
      L.log(
        `  • Noise for sta ${i}: ${n.toFixed(6)} → t_noisy = ${noisy.toFixed(
          6
        )} s`
      );
      return noisy;
    });

    stations.forEach((s) => (s._isOutlier = false));
    injectedOutlierIndex = null;

    if (k > 0 && stations.length) {
      const i = Math.floor(Math.random() * stations.length);
      const sign = Math.random() < 0.5 ? -1 : 1;
      const bump = sign * k * sigma;
      tObs[i] += bump;
      stations[i]._isOutlier = true;
      injectedOutlierIndex = i;
      L.log(
        `  • [Outlier] Station #${i} got ${sign > 0 ? "+" : "-"}${k}σ = ${
          bump.toFixed ? bump.toFixed(6) : bump
        } s`
      );
    }

    L.table(
      tObs.map((t, i) => ({
        station: i,
        clean_s: +tClean[i].toFixed(6),
        observed_s: +t.toFixed(6),
        outlier: i === injectedOutlierIndex,
      }))
    );
    L.end();
    return tObs;
  }

  // -----------------------------
  // Solves: (Gwᵀ Gw + λ² I) Δm = Gwᵀ dw
  // -----------------------------
  function solveNormalEq(Gw, dw, lambda) {
    if (DBG.logMatrices) {
      L.start("[Solver] Build normal equations");
      L.log("Gw (whitened Jacobian):", Gw);
      L.log("dw (whitened residuals):", dw);
    }
    const GT = numeric.transpose(Gw); // (P x N)
    let AtA = numeric.dot(GT, Gw); // (P x P)
    const P = AtA.length;
    for (let i = 0; i < P; i++) AtA[i][i] += lambda * lambda;
    const Atb = numeric.dot(GT, dw); // (P)

    if (DBG.logMatrices) {
      L.log("A = (Gw^T Gw + λ² I):", AtA);
      L.log("b = Gw^T dw:", Atb);
    }

    const delta_m = numeric.solve(AtA, Atb);
    if (DBG.logMatrices) {
      L.log("Δm solution:", delta_m);
      L.end();
    }
    return { delta_m };
  }

  // -----------------------------
  // Inversion for [x, y, z, t0, v]  -> returns { m, hist, ellipse }
  // -----------------------------
  function Inversion(tObs, start, lambda, sigma) {
    L.start("[Inv] Start Gauss–Newton with Tikhonov");
    let m = [start.x, start.y, start.z, start.t0, start.v];
    L.log("Initial model m0 = [x, y, z, t0, v] =", m);

    const hist = [];
    const sig = Math.max(1e-9, +sigma);

    for (let it = 0; it < maxIter; it++) {
      L.start(`[Iter ${it}] Forward model & residuals`);
      const tPred = stations.map((s) =>
        travelTime(s.x, s.y, s.z, m[0], m[1], m[2], m[3], m[4])
      );
      const delta = numeric.sub(tObs, tPred);
      const rms = Math.sqrt(
        numeric.sum(delta.map((d) => d * d)) / delta.length
      );
      hist.push(rms);

      L.table(
        tPred.map((tp, i) => ({
          station: i,
          t_pred_s: +tp.toFixed(6),
          dt_s: +delta[i].toFixed(6),
        }))
      );
      L.log(`RMS = ${rms.toFixed(6)} s`);
      L.end();

      // Jacobian (N x 5)
      L.start(`[Iter ${it}] Build Jacobian G (unwhitened)`);
      const G = stations.map((s) => {
        const dx = m[0] - s.x,
          dy = m[1] - s.y,
          dz = m[2] - s.z;
        const R = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-12;
        const v = m[4];
        return [dx / (v * R), dy / (v * R), dz / (v * R), 1.0, -R / (v * v)];
      });
      if (DBG.logMatrices) L.log("G:", G);
      L.end();

      // Whitening
      const w = 1 / sig;
      const Gw = G.map((row) => row.map((e) => e * w));
      const dw = delta.map((d) => d * w);

      // Solve (Gwᵀ Gw + λ² I) Δm = Gwᵀ dw
      const { delta_m } = solveNormalEq(Gw, dw, lambda);

      // Update
      const m_old = m.slice();
      m = numeric.add(m, delta_m);

      L.start(`[Iter ${it}] Update model`);
      L.log("m_old:", m_old);
      L.log("Δm:", delta_m);
      L.log("m_new:", m);
      const stepNorm = numeric.norm2(delta_m);
      L.log(`||Δm||2 = ${stepNorm.toExponential(6)}`);
      L.end();

      if (stepNorm < 1e-4) {
        L.log(`[Iter ${it}] Converged: small update norm.`);
        break;
      }
    }

    // Posterior covariance in whitened space
    L.start("[Inv] Posterior covariance & 1-σ ellipse");
    const G = stations.map((s) => {
      const dx = m[0] - s.x,
        dy = m[1] - s.y,
        dz = m[2] - s.z;
      const R = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-12;
      const v = m[4];
      return [dx / (v * R), dy / (v * R), dz / (v * R), 1.0, -R / (v * v)];
    });
    const Gw = G.map((row) => row.map((e) => e / Math.max(1e-9, +sigma)));
    const GTw = numeric.transpose(Gw);
    let H = numeric.dot(GTw, Gw);
    for (let i = 0; i < H.length; i++) H[i][i] += lambda * lambda;

    let Cov = null;
    try {
      Cov = numeric.inv(H);
      L.log("Inv(H) succeeded.");
    } catch (e) {
      L.log("Inv(H) failed → ellipse skipped.", e?.message || e);
      Cov = null;
    }

    let ellipse = null;
    if (Cov) {
      const s2 = sigma * sigma;
      const Cxx = Cov[0][0] * s2;
      const Cxy = Cov[0][1] * s2;
      const Cyy = Cov[1][1] * s2;

      const { l1, l2, ux, uy } = eig2x2(Cxx, Cxy, Cyy);
      const rx = Math.sqrt(Math.max(0, l1));
      const ry = Math.sqrt(Math.max(0, l2));
      const thetaDeg = Math.atan2(uy, ux) * (180 / Math.PI);

      if (isFinite(rx) && isFinite(ry) && rx > 0 && ry > 0) {
        ellipse = { cx: m[0], cy: m[1], rx, ry, thetaDeg };
        L.log("1-σ ellipse:", ellipse);
      } else {
        L.log("Ellipse radii non-finite or non-positive → skipped.");
      }
    }
    L.end();

    L.log("[Inv] Final model (x,y,z,t0,v):", m);
    L.log("[Inv] RMS history:", hist);
    return { m, hist, ellipse };
  }

  // -----------------------------
  // Single-station highlighter: ONLY the injected outlier
  // -----------------------------
  function setSingleHighlight(k) {
    highlightIdx.clear();
    if (k > 0 && injectedOutlierIndex != null) {
      highlightIdx.add(injectedOutlierIndex);
    }
    if (DBG.verbose)
      L.log(
        "[Highlight] indices:",
        [...highlightIdx],
        "outlierIndex:",
        injectedOutlierIndex
      );
  }

  // -----------------------------
  // Misfit grid (x,y), t0 optimized, v fixed to vStart
  // -----------------------------
  function misfitGrid(tObs, vFixed, N = 60) {
    L.start(`[Misfit] Grid N=${N} with vFixed=${vFixed}`);
    L.time("[Misfit] time");
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

        const R_over_v = stations.map(
          (s) =>
            Math.sqrt(
              (x - s.x) ** 2 + (y - s.y) ** 2 + (eventTrue.z - s.z) ** 2
            ) / vFixed
        );
        const t0star = d3.mean(tObs.map((t, k) => t - R_over_v[k]));
        const val = d3.mean(
          tObs.map((t, k) => {
            const r = t - (t0star + R_over_v[k]);
            return r * r; // MSE
          })
        );
        row.push(val);
        if (val < zmin) zmin = val;
        if (val > zmax) zmax = val;

        if (DBG.logEveryGridCell) {
          L.log(
            `  cell (i=${i}, j=${j}) x=${x.toFixed(2)} y=${y.toFixed(
              2
            )} t0*=${t0star.toFixed(6)} MSE=${val.toExponential(6)}`
          );
        }
      }

      if (!DBG.logEveryGridCell && DBG.gridSamples > 0) {
        // sample a few cells per row to keep console readable
        const idxs = d3
          .range(row.length)
          .filter((ii) => ii % Math.ceil(row.length / DBG.gridSamples) === 0);
        const sample = idxs.map((ii) => ({
          i: ii,
          x: xs[ii].toFixed(2),
          MSE: row[ii],
        }));
        L.log(`  row j=${j}: sample cells`, sample);
      }

      Z.push(row);
    }
    L.timeEnd("[Misfit] time");
    L.log(
      `[Misfit] MSE range: [${zmin.toExponential(6)}, ${zmax.toExponential(6)}]`
    );
    L.end();
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

  function drawWaveforms(tObs) {
    if (!wavesSel.node()) return;

    // --- compute CLEAN arrivals (no noise, no outlier) ---
    const tClean = stations.map((s) =>
      travelTime(s.x, s.y, s.z, eventTrue.x, eventTrue.y, eventTrue.z, 0, vTrue)
    );

    const svg = wavesSel;
    svg.selectAll("*").remove();

    const tmin = d3.min(tClean) - 0.8;
    const tmax = d3.max(tClean) + 0.8;
    const pad = { l: 70, r: 30, t: 24, b: 40 };
    const W = +svg.attr("width") || 1000;
    const rowHeight = 120; // vertical spacing per station
    const H = pad.t + pad.b + rowHeight * tClean.length;
    svg.attr("height", H);

    const x = d3
      .scaleLinear()
      .domain([tmin, tmax])
      .range([pad.l, W - pad.r]);
    const yBand = d3
      .scaleBand()
      .domain(d3.range(tClean.length))
      .range([pad.t, H - pad.b])
      .paddingInner(0.25);

    svg
      .append("g")
      .attr("transform", `translate(0,${H - pad.b})`)
      .call(d3.axisBottom(x).ticks(8).tickSizeOuter(0))
      .call((g) =>
        g
          .append("text")
          .attr("x", W - pad.r)
          .attr("y", 32)
          .attr("fill", "#555")
          .attr("text-anchor", "end")
          .text("Time (s)")
      );

    svg
      .append("g")
      .attr("transform", `translate(${pad.l},0)`)
      .call(
        d3
          .axisLeft(yBand)
          .tickFormat((i) => `Sta ${i + 1}`)
          .tickSizeOuter(0)
      );

    // baselines
    svg
      .append("g")
      .selectAll("line.row")
      .data(d3.range(tClean.length))
      .join("line")
      .attr("x1", x(tmin))
      .attr("x2", x(tmax))
      .attr("y1", (i) => yBand(i) + yBand.bandwidth() / 2)
      .attr("y2", (i) => yBand(i) + yBand.bandwidth() / 2)
      .attr("stroke", "#e5e7eb");

    const halfWidthSec = 0.15; // time half-width of zigzag
    const tailSec = 0.5; // baseline before/after
    const aspect = 2.0; // height = aspect × width

    const line = d3
      .line()
      .curve(d3.curveLinear)
      .x((d) => x(d.t))
      .y((d) => d.y);

    tClean.forEach((t0, i) => {
      const cy = yBand(i) + yBand.bandwidth() / 2;
      const zigWidthPx = x(t0 + halfWidthSec) - x(t0 - halfWidthSec);
      let A = (aspect * zigWidthPx) / 2;
      A = Math.min(A, rowHeight * 0.42);

      const pts = [
        { t: tmin, y: cy },
        { t: t0 - tailSec, y: cy },
        { t: t0 - halfWidthSec, y: cy - A },
        { t: t0, y: cy },
        { t: t0 + halfWidthSec, y: cy + A },
        { t: t0 + tailSec, y: cy },
        { t: tmax, y: cy },
      ];

      svg
        .append("path")
        .attr("d", line(pts))
        .attr("fill", "none")
        .attr("stroke", "#374151")
        .attr("stroke-width", 2);

      svg
        .append("line")
        .attr("x1", x(t0))
        .attr("x2", x(t0))
        .attr("y1", cy - A - 4)
        .attr("y2", cy + A + 4)
        .attr("stroke", "#9ca3af")
        .attr("stroke-width", 1);
    });
  }

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

    // Axes
    svg
      .append("g")
      .attr("transform", `translate(0,${h - pad})`)
      .call(d3.axisBottom(x).ticks(7));
    svg
      .append("g")
      .attr("transform", `translate(${pad},0)`)
      .call(d3.axisLeft(y).ticks(7));

    // Cursor by mode
    svg.style("cursor", modeCursor[toolMode] || "default");

    // Stations
    const gStations = svg.append("g");
    const sel = gStations
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

    // Remove mode: click a station to delete it
    if (toolMode === "remove") {
      sel.on("click", (ev, d) => {
        ev.stopPropagation();
        removeStationAtIndex(d.i);
      });
    } else {
      sel.on("click", null);
    }

    // Move mode: drag behavior
    const drag = d3
      .drag()
      .on("start", function () {
        d3.select(this).attr("opacity", 0.8);
        svg.style("cursor", "grabbing");
      })
      .on("drag", function (ev, d) {
        const xkm = clamp(x.invert(ev.x), XY.min, XY.max);
        const ykm = clamp(y.invert(ev.y), XY.min, XY.max);
        stations[d.i].x = xkm;
        stations[d.i].y = ykm;
        d3.select(this).attr("transform", `translate(${x(xkm)},${y(ykm)})`);
      })
      .on("end", function () {
        d3.select(this).attr("opacity", 1);
        svg.style("cursor", modeCursor[toolMode] || "default");
      });

    if (toolMode === "move") {
      sel.call(drag);
    } else {
      sel.on(".drag", null);
    }

    if (ellipseParams) {
      const { cx, cy, rx, ry, thetaDeg } = ellipseParams;
      const rxPx = Math.abs(x(cx + rx) - x(cx));
      const ryPx = Math.abs(y(cy + ry) - y(cy));
      svg
        .append("g")
        .attr("transform", `translate(${x(cx)},${y(cy)}) rotate(${thetaDeg})`)
        .append("ellipse")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("rx", rxPx)
        .attr("ry", ryPx)
        .attr("fill", "#f59e0b22")
        .attr("stroke", "#f59e0b")
        .attr("stroke-width", 2);
    }

    // Outlier ring(s)
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

    // Truth (star)
    svg
      .append("path")
      .attr("d", star())
      .attr("transform", `translate(${x(eventTrue.x)},${y(eventTrue.y)})`)
      .attr("fill", C.trueEvt)
      .attr("stroke", "white")
      .attr("stroke-width", 1.2);

    // Estimate (star)
    if (est) {
      svg
        .append("path")
        .attr("d", starSmall())
        .attr("transform", `translate(${x(est[0])},${y(est[1])})`)
        .attr("fill", C.estEvt)
        .attr("stroke", "white")
        .attr("stroke-width", 1.2);
    }

    // Background click for Add mode
    svg.on("click", (event) => {
      if (toolMode !== "add") return;
      if (event.target.closest("#map") !== svg.node()) return;
      const [mx, my] = d3.pointer(event, svg.node());
      if (mx < pad || mx > w - pad || my < pad || my > h - pad) return;
      const xkm = x.invert(mx);
      const ykm = y.invert(my);
      addStationAt(xkm, ykm);
    });
  }

  function drawMisfit(tObs) {
    const svg = misfitSel;
    svg.selectAll("*").remove();
    const { w, h } = getWH(svg);

    const pad = {
      l: 36,
      r: 36,
      t: 18,
      bAxis: 34,
      cbBarH: 12,
      cbGap: 42,
      cbAxisGap: 8,
    };
    const plotBottomY =
      h - (pad.bAxis + pad.cbGap + pad.cbBarH + pad.cbAxisGap);

    const x = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([pad.l, w - pad.r]);
    const y = d3
      .scaleLinear()
      .domain([XY.min, XY.max])
      .range([plotBottomY, pad.t]);

    svg
      .append("g")
      .attr("transform", `translate(0,${plotBottomY})`)
      .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));
    svg
      .append("g")
      .attr("transform", `translate(${pad.l},0)`)
      .call(d3.axisLeft(y).ticks(7).tickSizeOuter(0));

    // Misfit grid (MSE → RMS)
    const { xs, ys, Z, zmin, zmax } = misfitGrid(tObs, vStart, 60);
    const Zr = Z.map((row) => row.map((v) => Math.sqrt(Math.max(0, v))));
    let rmin = d3.min(Zr.flat());
    let rmax = d3.max(Zr.flat());
    L.log(`[Misfit] RMS range: [${rmin.toFixed(6)}, ${rmax.toFixed(6)}]`);

    const color = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([rmax, rmin]);

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
          .attr("fill", color(Zr[j][i]));
      }
    }

    // Colorbar
    const barX0 = pad.l,
      barX1 = w - pad.r;
    const barY = plotBottomY + pad.cbGap;
    const gradientId = "misfit-colorbar-rms";

    const defs = svg.append("defs");
    const grad = defs
      .append("linearGradient")
      .attr("id", gradientId)
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");

    const nStops = 32;
    d3.range(nStops).forEach((i) => {
      const t = i / (nStops - 1);
      const r = rmin + t * (rmax - rmin);
      grad
        .append("stop")
        .attr("offset", `${t * 100}%`)
        .attr("stop-color", color(r));
    });

    svg
      .append("rect")
      .attr("x", barX0)
      .attr("y", barY)
      .attr("width", barX1 - barX0)
      .attr("height", pad.cbBarH)
      .style("fill", `url(#${gradientId})`);

    const scaleBarX = d3
      .scaleLinear()
      .domain([rmin, rmax])
      .range([barX0, barX1]);
    const start = Math.ceil(rmin),
      end = Math.floor(rmax);
    const step = Math.max(1, Math.ceil((rmax - rmin) / 6));
    let rmsInts = d3.range(start, end + 1, step);
    if (rmsInts.length === 0) rmsInts = [Math.round(rmin), Math.round(rmax)];

    const axisBar = d3
      .axisBottom(scaleBarX)
      .tickValues(rmsInts)
      .tickFormat((d) => d);
    const axisY = barY + pad.cbBarH + pad.cbAxisGap;
    svg
      .append("g")
      .attr("class", "colorbar-axis")
      .attr("transform", `translate(0, ${axisY})`)
      .call(axisBar);

    const lblY = barY - 6;
    svg
      .append("text")
      .attr("x", barX0)
      .attr("y", lblY)
      .attr("text-anchor", "start")
      .attr("font-size", 11)
      .attr("fill", "#444")
      .text("Good misfit (low)");
    svg
      .append("text")
      .attr("x", barX1)
      .attr("y", lblY)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("fill", "#444")
      .text("Bad misfit (high)");
    svg
      .append("text")
      .attr("x", barX1)
      .attr("y", axisY + 16)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("fill", "#666")
      .text("RMS (seconds)");

    // Overlays
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
    L.start("[Flow] Generate random stations");
    const n = +el.numStations.value || 6;
    stations = randomStations(n);
    highlightIdx.clear();
    injectedOutlierIndex = null;
    L.table(
      stations.map((s, i) => ({ i, x: s.x.toFixed(2), y: s.y.toFixed(2) }))
    );
    drawMap(null);
    L.end();
  }

  function run() {
    if (!validate()) return;

    L.start("[Flow] Run experiment");
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

    L.log("Inputs:", {
      eventTrue,
      vTrue,
      vStart,
      sigma,
      lambda,
      outlierLevel: k,
      stations: stations.length,
    });

    // forward: observations
    const tObs = makeObservations(eventTrue, vTrue, sigma, k);

    // start model
    const start = {
      x: d3.mean(stations, (s) => s.x) || 0,
      y: d3.mean(stations, (s) => s.y) || 0,
      z: 0,
      t0: 0,
      v: vStart,
    };
    L.log("Start model guess:", start);

    // invert
    const { m, hist, ellipse } = Inversion(tObs, start, lambda, sigma);

    // use ellipse
    ellipseParams = ellipse || null;

    // highlight ONLY injected station
    setSingleHighlight(k);

    // draw
    drawMap(m);
    drawMisfit(tObs);
    drawConvergence(hist);
    drawWaveforms(tObs);

    // KPIs
    el.kIter.textContent = String(hist.length);
    el.kRMS.textContent = (hist.at(-1) ?? NaN).toFixed(3);
    el.kLambda.textContent = lambda.toFixed(2);

    const fx = (n) => Number(n).toFixed(2);
    if (el.trueXYZ)
      el.trueXYZ.textContent = `x=${fx(eventTrue.x)}, y=${fx(
        eventTrue.y
      )}, z=${fx(eventTrue.z)}`;
    if (el.estXYZ)
      el.estXYZ.textContent = `x=${fx(m[0])}, y=${fx(m[1])}, z=${fx(m[2])}`;

    // Final summary dump
    console.log("[RESULT]", {
      stations,
      injectedOutlierIndex,
      highlightIdx: [...highlightIdx],
      finalModel: { xs: m[0], ys: m[1], zs: m[2], t0: m[3], v: m[4] },
      rmsHistory: hist,
      lambda,
      sigma,
      vTrue,
      vStart,
      eventTrue,
      ellipseParams,
    });
    L.end();
  }

  function reset() {
    L.start("[Flow] Reset");
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
    if (el.sigma) el.sigma.value = 1.5;
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
    // el.kLambda.textContent = "—";
    wavesSel.selectAll("*").remove();
    L.end();
  }

  el.btnGenerate.addEventListener("click", generate);
  el.btnRun.addEventListener("click", run);
  el.btnReset.addEventListener("click", reset);
  el.modeAdd?.addEventListener("click", () => setMode("add"));
  el.modeMove?.addEventListener("click", () => setMode("move"));
  el.modeRemove?.addEventListener("click", () => setMode("remove"));

  // Init
  reset();
  generate();
})();
