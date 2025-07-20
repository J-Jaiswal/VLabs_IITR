function strikeDipRakeToMomentTensor(strike, dip, rake, magnitudeMw) {
  const deg2rad = (deg) => (deg * Math.PI) / 180;

  strike = deg2rad(strike);
  dip = deg2rad(dip);
  rake = deg2rad(rake);

  const cos = Math.cos;
  const sin = Math.sin;

  const M0 = Math.pow(10, 1.5 * magnitudeMw + 9.1);

  const Mrr =
    -M0 *
    (sin(dip) * cos(rake) * Math.sin(2 * strike) +
      sin(2 * dip) * sin(rake) * Math.pow(sin(strike), 2));
  const Mtt =
    M0 *
    (sin(dip) * cos(rake) * Math.sin(2 * strike) -
      sin(2 * dip) * sin(rake) * Math.pow(cos(strike), 2));
  const Mpp = M0 * sin(2 * dip) * sin(rake);
  const Mrp =
    -M0 *
    (cos(dip) * cos(rake) * cos(strike) +
      cos(2 * dip) * sin(rake) * sin(strike));
  const Mtp =
    -M0 *
    (cos(dip) * cos(rake) * sin(strike) -
      cos(2 * dip) * sin(rake) * cos(strike));
  const Mrt =
    -M0 *
    (sin(dip) * cos(rake) * cos(2 * strike) +
      0.5 * sin(2 * dip) * sin(rake) * Math.sin(2 * strike));

  return { Mrr, Mtt, Mpp, Mrp, Mtp, Mrt };
}

let waveforms = { Z: [], R: [], T: [] };
let time = Array.from({ length: 4000 }, (_, i) => i / 4);

document.addEventListener("DOMContentLoaded", () => {
  // Function to generate and plot waveforms
  const generateWaveforms = async () => {
    const strike = parseFloat(document.getElementById("strike").value);
    const dip = parseFloat(document.getElementById("dip").value);
    const rake = parseFloat(document.getElementById("rake").value);
    const magnitude = parseFloat(document.getElementById("magnitude").value);
    const azimuth = parseFloat(document.getElementById("azimuth").value);

    const az = (azimuth * Math.PI) / 180;
    const mt = strikeDipRakeToMomentTensor(strike, dip, rake, magnitude);

    const channelMap = {
      Z: ["ZSS", "ZDD", "ZEP", "ZDS"],
      R: ["RSS", "RDD", "REP", "RDS"],
      T: ["TSS", "TDS"],
    };

    const greensData = {};
    for (const chs of Object.values(channelMap)) {
      for (const ch of chs) {
        try {
          const res = await fetch(`greens/${ch}.json`);
          const json = await res.json();
          greensData[ch] = json.data;
        } catch {
          greensData[ch] = Array(15520).fill(0);
        }
      }
    }

    const { Mrr, Mtt, Mpp, Mrp, Mtp, Mrt } = mt;

    waveforms.Z = greensData["ZSS"]
      .map(
        (_, i) =>
          Mtt *
            ((greensData["ZSS"][i] / 2) * Math.cos(2 * az) -
              greensData["ZDD"][i] / 6 +
              greensData["ZEP"][i] / 3) +
          Mpp *
            ((-greensData["ZSS"][i] / 2) * Math.cos(2 * az) -
              greensData["ZDD"][i] / 6 +
              greensData["ZEP"][i] / 3) +
          Mrr * (greensData["ZDD"][i] / 3 + greensData["ZEP"][i] / 3) +
          Mtp * (greensData["ZSS"][i] * Math.sin(2 * az)) +
          Mrt * (greensData["ZDS"][i] * Math.cos(az)) +
          Mrp * (greensData["ZDS"][i] * Math.sin(az))
      )
      .slice(0, 4000);

    waveforms.R = greensData["RSS"]
      .map(
        (_, i) =>
          Mtt *
            ((greensData["RSS"][i] / 2) * Math.cos(2 * az) -
              greensData["RDD"][i] / 6 +
              greensData["REP"][i] / 3) +
          Mpp *
            ((-greensData["RSS"][i] / 2) * Math.cos(2 * az) -
              greensData["RDD"][i] / 6 +
              greensData["REP"][i] / 3) +
          Mrr * (greensData["RDD"][i] / 3 + greensData["REP"][i] / 3) +
          Mtp * (greensData["RSS"][i] * Math.sin(2 * az)) +
          Mrt * (greensData["RDS"][i] * Math.cos(az)) +
          Mrp * (greensData["RDS"][i] * Math.sin(az))
      )
      .slice(0, 4000);

    waveforms.T = greensData["TSS"]
      .map(
        (_, i) =>
          Mtt * ((greensData["TSS"][i] / 2) * Math.sin(2 * az)) -
          Mpp * ((greensData["TSS"][i] / 2) * Math.sin(2 * az)) -
          Mtp * (greensData["TSS"][i] * Math.cos(2 * az)) +
          Mrt * (greensData["TDS"][i] * Math.sin(az)) -
          Mrp * (greensData["TDS"][i] * Math.cos(az))
      )
      .slice(0, 4000);

    Plotly.newPlot(
      "waveformPlot",
      [
        {
          x: time,
          y: waveforms.Z,
          type: "scatter",
          mode: "lines",
          name: "Z",
          line: { color: "black" },
          yaxis: "y1",
        },
        {
          x: time,
          y: waveforms.R,
          type: "scatter",
          mode: "lines",
          name: "R",
          line: { color: "black" },
          yaxis: "y2",
        },
        {
          x: time,
          y: waveforms.T,
          type: "scatter",
          mode: "lines",
          name: "T",
          line: { color: "black" },
          yaxis: "y3",
        },
      ],
      {
        height: 700,
        width: 1150,
        margin: { l: 70, r: 70, t: 20, b: 50 },
        showlegend: false,
        xaxis: {
          domain: [0, 1],
          title: "Time (s)",
          anchor: "y3",
          range: [0, 1000],
          tickfont: { size: 12 },
        },
        yaxis: {
          domain: [0.7, 1],
          title: "Amplitude",
          automargin: true,
          tickfont: { size: 12 },
          tickpadding: 10,
        },
        yaxis2: {
          domain: [0.35, 0.65],
          title: "Amplitude",
          automargin: true,
          tickfont: { size: 12 },
          tickpadding: 10,
        },
        yaxis3: {
          domain: [0.0, 0.3],
          title: "Amplitude",
          automargin: true,
          tickfont: { size: 12 },
          tickpadding: 10,
        },
        annotations: [
          {
            text: "Vertical (Z)",
            xref: "paper",
            yref: "paper",
            x: 1.0,
            y: 0.95,
            showarrow: false,
          },
          {
            text: "Radial (R)",
            xref: "paper",
            yref: "paper",
            x: 1.0,
            y: 0.57,
            showarrow: false,
          },
          {
            text: "Transverse (T)",
            xref: "paper",
            yref: "paper",
            x: 1.0,
            y: 0.2,
            showarrow: false,
          },
        ],
      }
    );
  };

  // Bind button click
  document
    .getElementById("generateBtn")
    ?.addEventListener("click", generateWaveforms);

  // ✅ Auto trigger on initial load
  generateWaveforms();
});

document.getElementById("downloadBtn").addEventListener("click", () => {
  const rows = ["time,Z,R,T"];
  for (let i = 0; i < time.length; i++) {
    rows.push(
      `${time[i]},${waveforms.Z[i]},${waveforms.R[i]},${waveforms.T[i]}`
    );
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "waveforms.csv";
  a.click();
  URL.revokeObjectURL(url);
});
