/* Earthquake Location — Tikhonov (with detailed console logging)
 * --------------------------------------------------------------
 * Adds:
 *  - Max-stations warning alert
 *  - λ = 0 safe solve
 *  - Plot per-iteration model points on map (numbered)
 *  - UI tables for stations/arrivals and per-iteration model parameters
 */

(function () {
  // Debug / logging helpers
  const DBG = {
    verbose: true,
    logMatrices: true,
    logEveryGridCell: false,
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
  let vTrue = 5.0;
  let vStart = 5.0;
  let sigma = 1.5;
  let lambda = 0.1;
  let maxIter = 8;
  const XY = { min: -60, max: 60 };
  let injectedOutlierIndex = null;
  let highlightIdx = new Set();
  let ellipseParams = null;
  let modelTrail = []; // [{x,y,z,t0,v}] per iteration
  const ST_MIN = 3;
  const ST_MAX = 15;

  // λ helper: if user enters 0, use a tiny effective λ² so matrices don’t go singular
  const LAM_TINY = 0.0001;
  const effLambda = (lam) => (lam > 0 ? lam * lam : LAM_TINY);

  let toolMode = "add";
  const modeCursor = { add: "copy", move: "grab", remove: "not-allowed" };

  const C = {
    station: "#111111",
    outlier: "#f59e0b",
    trueEvt: "#e11d48",
    estEvt: "#0ea5e9",
    trail: "#0ea5e9",
  };
  const WAVES_FIXED_INNER_H = 700;

  // SVGs
  const mapSel = d3.select("#map");
  const misfitSel = d3.select("#misfit");
  const convSel = d3.select("#conv");
  const wavesSel = d3.select("#waves");

  // -----------------------------
  // UI handles
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
    outlier: document.getElementById("outlier"),
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
    maxIter: document.getElementById("maxIter"),
    alertBox: document.getElementById("alertBox"),
    tblStationsBody: document.getElementById("tblStationsBody"),
    tblModelBody: document.getElementById("tblModelBody"),
  };

  el.vTrue?.addEventListener("input", () => {
    el.vTrueLbl.textContent = Number(el.vTrue.value).toFixed(1);
  });

  // -----------------------------
  // Utilities
  // -----------------------------
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fx = (n, k = 2) => Number(n).toFixed(k);

  function warn(msg) {
    if (!el.alertBox) return alert(msg);
    el.alertBox.textContent = msg;
    el.alertBox.classList.remove("d-none");
    // auto-hide after 3s
    window.clearTimeout(warn._t);
    warn._t = window.setTimeout(() => {
      el.alertBox.classList.add("d-none");
      el.alertBox.textContent = "";
    }, 3000);
  }

  function clampStationsCount(n) {
    n = +n;
    if (!Number.isFinite(n)) n = ST_MIN;
    const nn = Math.max(ST_MIN, Math.min(ST_MAX, n));
    if (n > ST_MAX)
      warn(`Max stations is ${ST_MAX}. Extra stations are ignored.`);
    return nn;
  }

  function normalizeStationsArray() {
    let n = stations.length;
    if (n > ST_MAX) {
      stations = stations.slice(0, ST_MAX);
      warn(`Max stations is ${ST_MAX}. Extra stations were removed.`);
    } else if (n < ST_MIN) {
      stations = stations.concat(randomStations(ST_MIN - n));
    }
    if (el.numStations) el.numStations.value = stations.length;
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
  // Tables
  // -----------------------------
  function renderStationsTable(tClean, tObs) {
    if (!el.tblStationsBody) return;
    const rows = tObs.map((t, i) => {
      const out = i === injectedOutlierIndex ? "true" : "false";
      return `<tr>
        <td>${i + 1}</td>
        <td>${fx(tClean[i], 6)}</td>
        <td>${fx(t, 6)}</td>
        <td>${out}</td>
      </tr>`;
    });
    el.tblStationsBody.innerHTML = rows.join("");
  }

  function clearModelTable() {
    if (el.tblModelBody) el.tblModelBody.innerHTML = "";
  }
  function renderModelRow(it, m) {
    if (!el.tblModelBody) return;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${it + 1}</td>
      <td>${fx(m[0])}</td>
      <td>${fx(m[1])}</td>
      <td>${fx(m[2])}</td>
      <td>${fx(m[3], 4)}</td>
      <td>${fx(m[4], 3)}</td>`;
    el.tblModelBody.appendChild(row);
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
        `  • [Outlier] Station #${i} got ${
          sign > 0 ? "+" : "-"
        }${k}σ = ${bump.toFixed(6)} s`
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

    // UI table update
    renderStationsTable(tClean, tObs);

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
    const GT = numeric.transpose(Gw);
    let AtA = numeric.dot(GT, Gw);
    const P = AtA.length;

    // const tiny = 1e-8; // you can use 1e-9…1e-7; 1e-8 is a good default
    // const lam2 = lambda > 0 ? lambda * lambda : tiny;
    const lam2 = lambda * lambda;

    for (let i = 0; i < P; i++) AtA[i][i] += lam2;
    // for (let i = 0; i < P; i++) AtA[i][i] += lambda * lambda;
    const Atb = numeric.dot(GT, dw);

    if (DBG.logMatrices) {
      L.log("A = (Gw^T Gw + λ² I):", AtA);
      L.log("b = Gw^T dw:", Atb);
    }

    let delta_m;
    try {
      // Try normal solve
      delta_m = numeric.solve(AtA, Atb);
    } catch (e) {
      // Fallback: SVD-based pseudo-inverse (handles λ=0 singularities)
      L.log(
        "[Solver] numeric.solve failed → using SVD pseudoinverse fallback."
      );
      const svd = numeric.svd(AtA);
      const U = svd.U,
        S = svd.S,
        V = svd.V;
      const tol = 1e-10;
      const Sinv = numeric.diag(S.map((s) => (Math.abs(s) > tol ? 1 / s : 0)));
      // A^+ = V * S^+ * U^T
      const AtA_pinv = numeric.dot(numeric.dot(V, Sinv), numeric.transpose(U));
      delta_m = numeric.dot(AtA_pinv, Atb);
    }

    if (DBG.logMatrices) {
      L.log("Δm solution:", delta_m);
      L.end();
    }
    return { delta_m };
  }

  // -----------------------------
  // Inversion for [x, y, z, t0, v]  -> returns { m, hist, ellipse, trail }
  // -----------------------------
  function Inversion(tObs, start, lambda, sigma) {
    L.start("[Inv] Start Gauss–Newton with Tikhonov");
    let m = [start.x, start.y, start.z, start.t0, start.v];
    L.log("Initial model m0 = [x, y, z, t0, v] =", m);

    const hist = [];
    const sig = Math.max(1e-9, +sigma);
    modelTrail = [];
    clearModelTable();

    for (let it = 0; it < maxIter; it++) {
      // Forward & residuals
      const tPred = stations.map((s) =>
        travelTime(s.x, s.y, s.z, m[0], m[1], m[2], m[3], m[4])
      );
      const delta = numeric.sub(tObs, tPred);
      const rms = Math.sqrt(
        numeric.sum(delta.map((d) => d * d)) / delta.length
      );
      hist.push(rms);

      if (DBG.verbose) {
        L.start(`[Iter ${it}]`);
        L.table(
          tPred.map((tp, i) => ({
            station: i,
            t_pred_s: +tp.toFixed(6),
            dt_s: +delta[i].toFixed(6),
          }))
        );
        L.log(`RMS = ${rms.toFixed(6)} s`);
        L.end();
      }

      // Jacobian (N x 5)
      const G = stations.map((s) => {
        const dx = m[0] - s.x,
          dy = m[1] - s.y,
          dz = m[2] - s.z;
        const R = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-12;
        const v = m[4];
        return [dx / (v * R), dy / (v * R), dz / (v * R), 1.0, -R / (v * v)];
      });

      // Whitening
      const w = 1 / sig;
      const Gw = G.map((row) => row.map((e) => e * w));
      const dw = delta.map((d) => d * w);

      // Solve
      const { delta_m } = solveNormalEq(Gw, dw, lambda);

      // Update
      const m_old = m.slice();
      m = numeric.add(m, delta_m);
      const stepNorm = numeric.norm2(delta_m);

      // Log & store
      L.log(
        `[Iter ${it}] m_old:`,
        m_old,
        " Δm:",
        delta_m,
        " m_new:",
        m,
        ` ||Δm||2=${stepNorm.toExponential(6)}`
      );
      modelTrail.push({ x: m[0], y: m[1], z: m[2], t0: m[3], v: m[4] });
      renderModelRow(it, m);

      if (stepNorm < 1e-4) {
        L.log(`[Iter ${it}] Converged: small update norm.`);
        break;
      }
    }

    // covariance in whitened space
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
    return { m, hist, ellipse, trail: modelTrail.slice() };
  }

  // -----------------------------
  // Highlight only the injected outlier
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
            return r * r;
          })
        );
        row.push(val);
        if (val < zmin) zmin = val;
        if (val > zmax) zmax = val;
      }

      if (!DBG.logEveryGridCell && DBG.gridSamples > 0) {
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

  const tri = d3.symbol().type(d3.symbolTriangle).size(120);
  const triSmall = d3.symbol().type(d3.symbolTriangle).size(70);
  const star = d3.symbol().type(d3.symbolStar).size(240);
  const starSmall = d3.symbol().type(d3.symbolStar).size(140);

  function drawWaveforms(tObs) {
    if (!wavesSel.node()) return;

    const tClean = stations.map((s) =>
      travelTime(s.x, s.y, s.z, eventTrue.x, eventTrue.y, eventTrue.z, 0, vTrue)
    );

    const svg = wavesSel;
    svg.selectAll("*").remove();

    const halfWidthSec = 0.15;
    const tailSec = 0.5;
    const aspect = 2.0;

    const tmin = 0,
      tmax = 20;

    const pad = { l: 70, r: 30, t: 24, b: 40 };
    const W = wavesSel.node().clientWidth || +svg.attr("width") || 1000;
    const innerH = WAVES_FIXED_INNER_H;
    const H = pad.t + innerH + pad.b;
    svg.attr("height", H).attr("width", W);

    const x = d3
      .scaleLinear()
      .domain([tmin, tmax])
      .range([pad.l, W - pad.r - 2]);

    const N = Math.max(1, stations.length);
    const yBand = d3
      .scaleBand()
      .domain(d3.range(N))
      .range([pad.t, pad.t + innerH])
      .paddingInner(0.25);

    const tickVals = d3.range(0, 21, 2);
    const gx = svg
      .append("g")
      .attr("transform", `translate(0,${H - pad.b})`)
      .call(
        d3
          .axisBottom(x)
          .tickValues((tVals = tickVals))
          .tickSizeOuter(0)
      );
    if (
      !gx
        .selectAll(".tick")
        .filter((d) => d === 20)
        .size()
    ) {
      gx.append("g")
        .attr("class", "tick")
        .attr("transform", `translate(${x(20)},0)`)
        .append("text")
        .attr("fill", "currentColor")
        .attr("y", 9)
        .attr("dy", "0.71em")
        .attr("text-anchor", "middle")
        .text("20");
    }
    gx.append("text")
      .attr("x", W - pad.r)
      .attr("y", 32)
      .attr("fill", "#555")
      .attr("text-anchor", "end")
      .text("Time (s)");

    svg
      .append("g")
      .attr("transform", `translate(${pad.l},0)`)
      .call(
        d3
          .axisLeft(yBand)
          .tickFormat((i) => `Sta ${+i + 1}`)
          .tickSizeOuter(0)
      );

    svg
      .append("g")
      .selectAll("line.row")
      .data(d3.range(N))
      .join("line")
      .attr("x1", x(tmin))
      .attr("x2", x(tmax))
      .attr("y1", (i) => yBand(i) + yBand.bandwidth() / 2)
      .attr("y2", (i) => yBand(i) + yBand.bandwidth() / 2)
      .attr("stroke", "#e5e7eb");

    const line = d3
      .line()
      .curve(d3.curveLinear)
      .x((d) => x(d.t))
      .y((d) => d.y);
    const clampT = (t) => Math.min(tmax, Math.max(tmin, t));

    tClean.forEach((t0, i) => {
      const cy = yBand(i) + yBand.bandwidth() / 2;

      const tLeft = clampT(t0 - halfWidthSec);
      const tRight = clampT(t0 + halfWidthSec);
      const tTailL = clampT(t0 - tailSec);
      const tTailR = clampT(t0 + tailSec);
      const tMark = clampT(t0);

      const zigWidthPx = Math.max(1, x(tRight) - x(tLeft));
      let A = (aspect * zigWidthPx) / 2;
      A = Math.min(A, yBand.bandwidth() * 0.42);

      const pts = [
        { t: tmin, y: cy },
        { t: tTailL, y: cy },
        { t: tLeft, y: cy - A },
        { t: tMark, y: cy },
        { t: tRight, y: cy + A },
        { t: tTailR, y: cy },
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
        .attr("x1", x(tMark))
        .attr("x2", x(tMark))
        .attr("y1", cy - A - 4)
        .attr("y2", cy + A + 4)
        .attr("stroke", "#9ca3af")
        .attr("stroke-width", 1);
    });
  }

  function drawMap(est, trail = [], startGuess = null) {
    const svg = mapSel;
    svg.selectAll("*").remove();

    const { w, h } = getWH(svg);
    const pad = { l: 46, r: 12, t: 12, b: 42 };
    const iw = Math.max(0, w - pad.l - pad.r);
    const ih = Math.max(0, h - pad.t - pad.b);

    const x = d3.scaleLinear().domain([XY.min, XY.max]).range([0, iw]);
    const y = d3.scaleLinear().domain([XY.min, XY.max]).range([ih, 0]);

    const g = svg.append("g").attr("transform", `translate(${pad.l},${pad.t})`);
    const gAxes = svg
      .append("g")
      .attr("transform", `translate(${pad.l},${pad.t})`);

    g.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", iw)
      .attr("height", ih)
      .attr("fill", "white");

    g.append("g")
      .attr("class", "grid-x")
      .attr("transform", `translate(0,${ih})`)
      .call(
        d3.axisBottom(x).ticks(7).tickSize(-ih).tickFormat("").tickSizeOuter(0)
      )
      .selectAll("line")
      .attr("stroke", "#e5e7eb");
    g.append("g")
      .attr("class", "grid-y")
      .call(
        d3.axisLeft(y).ticks(7).tickSize(-iw).tickFormat("").tickSizeOuter(0)
      )
      .selectAll("line")
      .attr("stroke", "#e5e7eb");

    // Starting guess
    if (startGuess) {
      g.append("circle")
        .attr("cx", x(startGuess.x))
        .attr("cy", y(startGuess.y))
        .attr("r", 3)
        .attr("fill", "#10b981")
        .attr("stroke", "#10b981") // green outline
        .attr("stroke-width", 1);
      // .attr("stroke-dasharray", "3,2"); // dashed for distinction
    }
    gAxes
      .append("g")
      .attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0).tickPadding(6))
      .call((gx) => {
        gx.selectAll("text").attr("font-size", 12).attr("fill", "#374151");
        gx.selectAll("path, line").attr("stroke", "#6b7280");
      });
    gAxes
      .append("g")
      .call(d3.axisLeft(y).ticks(7).tickSizeOuter(0).tickPadding(6))
      .call((gy) => {
        gy.selectAll("text").attr("font-size", 12).attr("fill", "#374151");
        gy.selectAll("path, line").attr("stroke", "#6b7280");
      });

    // if (startGuess) {
    //   g.append("circle")
    //     .attr("cx", x(startGuess.x))
    //     .attr("cy", y(startGuess.y))
    //     .attr("r", 6)
    //     .attr("fill", "none")
    //     .attr("stroke", "#10b981") // green
    //     .attr("stroke-width", 2)
    //     .attr("stroke-dasharray", "3,2");
    // }

    svg.style("cursor", modeCursor[toolMode] || "default");

    // Stations
    const gStations = g.append("g");
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

    if (toolMode === "remove") {
      sel.on("click", (ev, d) => {
        ev.stopPropagation();
        removeStationAtIndex(d.i);
      });
    } else {
      sel.on("click", null);
    }

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

    if (toolMode === "move") sel.call(drag);
    else sel.on(".drag", null);

    // Uncertainty ellipse
    if (ellipseParams) {
      const { cx, cy, rx, ry, thetaDeg } = ellipseParams;
      const rxPx = Math.abs(x(cx + rx) - x(cx));
      const ryPx = Math.abs(y(cy + ry) - y(cy));
      g.append("g")
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

    // Outlier rings
    const gRing = g.append("g").attr("pointer-events", "none");
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

    // Truth
    g.append("path")
      .attr("d", star())
      .attr("transform", `translate(${x(eventTrue.x)},${y(eventTrue.y)})`)
      .attr("fill", C.trueEvt)
      .attr("stroke", "white")
      .attr("stroke-width", 1.2);

    // Iteration trail (small dots with labels 0,1,2,...)
    if (trail && trail.length) {
      const gTrail = g.append("g").attr("class", "trail");
      trail.forEach((mm, i) => {
        const px = x(mm.x),
          py = y(mm.y);
        gTrail
          .append("circle")
          .attr("cx", px)
          .attr("cy", py)
          .attr("r", 3.5)
          .attr("fill", C.trail)
          .attr("stroke", "white")
          .attr("stroke-width", 1.2);
        gTrail
          .append("text")
          .attr("x", px + 6)
          .attr("y", py - 6)
          .attr("class", "iter-dot-label")
          .text(String(i));
      });
    }

    // Estimate (final)
    if (est) {
      g.append("path")
        .attr("d", starSmall())
        .attr("transform", `translate(${x(est[0])},${y(est[1])})`)
        .attr("fill", C.estEvt)
        .attr("stroke", "white")
        .attr("stroke-width", 1.2);
    }

    // Add-mode clicks
    g.on("click", (event) => {
      if (toolMode !== "add") return;
      const [mx, my] = d3.pointer(event, g.node());
      if (mx < 0 || mx > iw || my < 0 || my > ih) return;
      const xkm = x.invert(mx),
        ykm = y.invert(my);
      addStationAt(xkm, ykm);
    });
  }

  function drawMisfit(tObs) {
    const svg = misfitSel;
    svg.selectAll("*").remove();
    const { w, h } = getWH(svg);

    const pad = {
      l: 24,
      r: 24,
      t: 12,
      bAxis: 26,
      cbBarH: 10,
      cbGap: 30,
      cbAxisGap: 6,
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
      .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0).tickPadding(4));
    svg
      .append("g")
      .attr("transform", `translate(${pad.l},0)`)
      .call(d3.axisLeft(y).ticks(7).tickSizeOuter(0).tickPadding(4));

    const { xs, ys, Z } = misfitGrid(tObs, vStart, 60);
    const Zr = Z.map((row) => row.map((v) => Math.sqrt(Math.max(0, v))));
    let rmin = d3.min(Zr.flat()),
      rmax = d3.max(Zr.flat());
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

    svg
      .append("text")
      .attr("x", barX0)
      .attr("y", barY - 6)
      .attr("text-anchor", "start")
      .attr("font-size", 11)
      .attr("fill", "#444")
      .text("Good misfit (low)");
    svg
      .append("text")
      .attr("x", barX1)
      .attr("y", barY - 6)
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
  function setMode(m) {
    L.start(`[UI] Set tool mode → ${m}`);
    toolMode = m;
    [el.modeAdd, el.modeMove, el.modeRemove].forEach((btn) =>
      btn?.classList.remove("active")
    );
    if (m === "add") el.modeAdd?.classList.add("active");
    if (m === "move") el.modeMove?.classList.add("active");
    if (m === "remove") el.modeRemove?.classList.add("active");
    drawMap(null, modelTrail);
    L.end();
  }

  function addStationAt(xkm, ykm) {
    if (stations.length >= ST_MAX) {
      warn(`Maximum number of stations (${ST_MAX}) reached.`);
      L.log(`[Stations] Not adding: already at max (${ST_MAX}).`);
      return;
    }
    stations.push({ x: xkm, y: ykm, z: 0, _isOutlier: false });
    normalizeStationsArray();
    L.log(`[Stations] Added at (x=${xkm.toFixed(2)}, y=${ykm.toFixed(2)})`);
    drawMap(null, modelTrail);
  }

  function removeStationAtIndex(i) {
    if (stations.length <= ST_MIN) {
      warn(`Need at least ${ST_MIN} stations.`);
      L.log(`[Stations] Not removing: already at min (${ST_MIN}).`);
      return;
    }
    if (i < 0 || i >= stations.length) return;
    const removed = stations[i];
    stations.splice(i, 1);
    normalizeStationsArray();
    L.log(
      `[Stations] Removed index ${i} @ (x=${removed.x.toFixed(
        2
      )}, y=${removed.y.toFixed(2)})`
    );
    drawMap(null, modelTrail);
  }

  function validate() {
    let n = +el.numStations.value;
    if (!Number.isFinite(n)) n = 6;
    n = clampStationsCount(n);
    if (el.numStations) el.numStations.value = n;
    if (n < 4) {
      alert("Need at least 4 stations.");
      return false;
    }
    return true;
  }

  function generate() {
    L.start("[Flow] Generate random stations");
    let n = +el.numStations.value || 6;
    n = clampStationsCount(n);
    if (el.numStations) el.numStations.value = n;
    stations = randomStations(n);
    highlightIdx.clear();
    injectedOutlierIndex = null;
    modelTrail = [];
    L.table(
      stations.map((s, i) => ({ i, x: s.x.toFixed(2), y: s.y.toFixed(2) }))
    );
    drawMap(null, modelTrail);
    L.end();
  }

  function run() {
    if (!validate()) return;

    L.start("[Flow] Run experiment");
    eventTrue = {
      x: +el.xInput.value,
      y: +el.yInput.value,
      z: +el.zInput.value,
    };
    vTrue = +el.vTrue.value;
    vStart = +el.vStart.value;
    sigma = Math.max(0, +el.sigma.value);
    // lambda = Math.max(0, +el.lambda.value);
    const lambdaRaw = Math.max(0, +el.lambda.value);
    const lambdaEff = effLambda(lambdaRaw);
    if (el.numStations)
      el.numStations.value = clampStationsCount(el.numStations.value);
    normalizeStationsArray();
    maxIter = Math.max(1, Math.min(50, +el.maxIter.value || 8));

    const k = parseOutlierLevel();

    L.log("Inputs:", {
      eventTrue,
      vTrue,
      vStart,
      sigma,
      lambda: lambdaEff,
      outlierLevel: k,
      stations: stations.length,
    });

    // forward: observations
    const tObs = makeObservations(eventTrue, vTrue, sigma, k);

    // // Optional: surface a gentle heads-up when the user typed 0
    // if (lambdaRaw === 0) {
    //   warn(
    //     `Damping λ is 0 → using a negligible ${LAM_TINY} to keep the solve stable.`
    //   );
    // }
    // start model
    const start = {
      x: d3.mean(stations, (s) => s.x) || 0,
      y: d3.mean(stations, (s) => s.y) || 0,
      z: eventTrue.z,
      t0: 0,
      v: vStart,
    };
    L.log("Start model guess:", start);

    // invert
    const { m, hist, ellipse, trail } = Inversion(
      tObs,
      start,
      lambdaEff,
      sigma
    );

    // store for drawing
    ellipseParams = ellipse || null;
    modelTrail = trail || [];

    // highlight ONLY injected station
    setSingleHighlight(k);

    // draw
    drawMap(m, modelTrail, start);
    drawMisfit(tObs);
    drawConvergence(hist);
    drawWaveforms(tObs);

    // KPIs
    el.kIter.textContent = String(hist.length);
    el.kRMS.textContent = (hist.at(-1) ?? NaN).toFixed(3);
    // el.kLambda && (el.kLambda.textContent = lambda.toFixed(2));
    if (el.kLambda) {
      el.kLambda.textContent =
        lambdaRaw === 0 ? `0 → ${LAM_TINY}` : lambdaEff.toFixed(2);
    }

    if (el.trueXYZ)
      el.trueXYZ.textContent = `x=${fx(eventTrue.x)}, y=${fx(
        eventTrue.y
      )}, z=${fx(eventTrue.z)}`;
    if (el.estXYZ)
      el.estXYZ.textContent = `x=${fx(m[0])}, y=${fx(m[1])}, z=${fx(m[2])}`;

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
      modelTrail,
    });
    L.end();
  }

  function reset() {
    L.start("[Flow] Reset");
    stations = [];
    eventTrue = { x: 10, y: 5, z: 8 };
    highlightIdx.clear();
    injectedOutlierIndex = null;
    modelTrail = [];
    if (el.xInput) el.xInput.value = 10;
    if (el.yInput) el.yInput.value = 5;
    if (el.zInput) el.zInput.value = 8;
    if (el.vTrue) {
      el.vTrue.value = 5.0;
      el.vTrueLbl.textContent = "5.0";
    }
    if (el.vStart) el.vStart.value = 5.0;
    if (el.sigma) el.sigma.value = 1.5;
    if (el.lambda) el.lambda.value = 0.1;
    if (el.numStations) el.numStations.value = 6;
    if (el.outlier) el.outlier.value = "none";
    el.tblStationsBody && (el.tblStationsBody.innerHTML = "");
    el.tblModelBody && (el.tblModelBody.innerHTML = "");

    mapSel.selectAll("*").remove();
    misfitSel.selectAll("*").remove();
    convSel.selectAll("*").remove();
    el.kIter.textContent = "—";
    el.kRMS.textContent = "—";
    wavesSel.selectAll("*").remove();
    L.end();
  }

  // Events
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
