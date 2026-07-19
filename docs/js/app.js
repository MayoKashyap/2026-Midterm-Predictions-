/* ElectMap 2026 — app.js */

const FIPS_TO_STATE = {
  '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA',
  '08':'CO','09':'CT','10':'DE','12':'FL','13':'GA',
  '15':'HI','16':'ID','17':'IL','18':'IN','19':'IA',
  '20':'KS','21':'KY','22':'LA','23':'ME','24':'MD',
  '25':'MA','26':'MI','27':'MN','28':'MS','29':'MO',
  '30':'MT','31':'NE','32':'NV','33':'NH','34':'NJ',
  '35':'NM','36':'NY','37':'NC','38':'ND','39':'OH',
  '40':'OK','41':'OR','42':'PA','44':'RI','45':'SC',
  '46':'SD','47':'TN','48':'TX','49':'UT','50':'VT',
  '51':'VA','53':'WA','54':'WV','55':'WI','56':'WY'
};

const STATE_NAMES = {
  AL:'Alabama',AK:'Alaska',AZ:'Arizona',AR:'Arkansas',CA:'California',
  CO:'Colorado',CT:'Connecticut',DE:'Delaware',FL:'Florida',GA:'Georgia',
  HI:'Hawaii',ID:'Idaho',IL:'Illinois',IN:'Indiana',IA:'Iowa',
  KS:'Kansas',KY:'Kentucky',LA:'Louisiana',ME:'Maine',MD:'Maryland',
  MA:'Massachusetts',MI:'Michigan',MN:'Minnesota',MS:'Mississippi',MO:'Missouri',
  MT:'Montana',NE:'Nebraska',NV:'Nevada',NH:'New Hampshire',NJ:'New Jersey',
  NM:'New Mexico',NY:'New York',NC:'North Carolina',ND:'North Dakota',OH:'Ohio',
  OK:'Oklahoma',OR:'Oregon',PA:'Pennsylvania',RI:'Rhode Island',SC:'South Carolina',
  SD:'South Dakota',TN:'Tennessee',TX:'Texas',UT:'Utah',VT:'Vermont',
  VA:'Virginia',WA:'Washington',WV:'West Virginia',WI:'Wisconsin',WY:'Wyoming'
};

const FEATURE_LABELS = {
  partisan_lean:   'Partisan Lean (Cook PVI)',
  dem_share_prev:  'Prior Dem Vote Share',
  pres_approval:   'Presidential Approval',
  incumbent_party: 'Incumbency',
  gdp_growth_q2:   'GDP Growth (Q2)',
  unemp_oct:       'Unemployment (Oct)',
  log_gdp_growth:  'GDP Growth (log)',
  midterm_year:    'Midterm Year Penalty',
  redistricted:    'Redistricted District'
};

const RATING_ORDER = ['Safe D', 'Lean D', 'Toss-up', 'Lean R', 'Safe R'];

const RATING_CSS = {
  'Safe D':  'rating-safe-d',
  'Lean D':  'rating-lean-d',
  'Toss-up': 'rating-tossup',
  'Lean R':  'rating-lean-r',
  'Safe R':  'rating-safe-r'
};

let predictions = [];
let lookup = {};        // "AL-1" → prediction object
let history = {};       // "AL-1" → [{year, dem_share}, ...]
let geojsonLayer = null;
let map = null;

/* ── Color scale ─────────────────────────────────────────── */
const colorScale = d3.scaleDiverging(d3.interpolateRdBu).domain([0, 0.5, 1]);

function districtColor(winProb) {
  if (winProb == null) return '#cccccc';
  return colorScale(winProb);
}

/* ── Lookup helpers ──────────────────────────────────────── */
function predKey(statePo, district) {
  return `${statePo}-${district}`;
}

function predFromFeature(feature) {
  const statePo = FIPS_TO_STATE[feature.properties.STATEFP];
  const district = parseInt(feature.properties.CD119FP, 10);
  if (!statePo) return null;
  return lookup[predKey(statePo, district)] || null;
}

function districtDisplayName(feature, pred) {
  const statePo = FIPS_TO_STATE[feature.properties.STATEFP];
  if (!statePo) return feature.properties.NAMELSAD || 'Unknown';
  const district = parseInt(feature.properties.CD119FP, 10);
  const stateName = STATE_NAMES[statePo] || statePo;
  if (district === 0) return `${stateName} At-Large`;
  const suffix = ordinal(district);
  return `${stateName} ${suffix} District`;
}

function ordinal(n) {
  const s = ['th','st','nd','rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

/* ── Map init ────────────────────────────────────────────── */
function initMap() {
  map = L.map('map', {
    center: [39.5, -98.35],
    zoom: 4,
    minZoom: 3,
    maxZoom: 10,
    zoomControl: true
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
  }).addTo(map);
}

/* ── GeoJSON layer ───────────────────────────────────────── */
function addDistrictLayer(geojson) {
  geojsonLayer = L.geoJSON(geojson, {
    style: feature => {
      const pred = predFromFeature(feature);
      return {
        fillColor:   districtColor(pred?.win_prob_dem),
        fillOpacity: 0.82,
        weight:      0.5,
        color:       '#ffffff',
        opacity:     1
      };
    },
    onEachFeature: (feature, layer) => {
      const pred = predFromFeature(feature);
      const name = districtDisplayName(feature, pred);

      layer.on('mouseover', function() {
        this.setStyle({ weight: 2, color: '#333', fillOpacity: 0.95 });
        this.bringToFront();
      });

      layer.on('mouseout', function() {
        geojsonLayer.resetStyle(this);
      });

      layer.on('click', function() {
        showSidebar(pred, name);
      });
    }
  }).addTo(map);
}

/* ── Sidebar ─────────────────────────────────────────────── */
function showSidebar(pred, name) {
  const sidebar  = document.getElementById('sidebar');
  const content  = document.getElementById('sidebar-content');
  sidebar.classList.add('open');

  if (!pred) {
    content.innerHTML = `<p class="sidebar-placeholder">No prediction data for this district.</p>`;
    return;
  }

  const ratingCss = RATING_CSS[pred.rating] || 'rating-tossup';
  const demWinPct = (pred.win_prob_dem * 100).toFixed(1);
  const repWinPct = ((1 - pred.win_prob_dem) * 100).toFixed(1);
  const demShare  = (pred.pred_dem_share * 100).toFixed(1);

  const histKey = predKey(pred.state_po, pred.district);
  const distHistory = history[histKey] || [];

  const histRows = distHistory.map(h => `
    <tr>
      <td>${h.year}</td>
      <td class="td-dem">${(h.dem_share * 100).toFixed(1)}%</td>
      <td class="td-rep">${((1 - h.dem_share) * 100).toFixed(1)}%</td>
    </tr>`).join('');

  const shapHtml = buildShapHtml(pred.shap || []);

  content.innerHTML = `
    <div class="sd-name">${name}</div>
    <span class="sd-rating ${ratingCss}">${pred.rating}</span>

    <div class="sd-prob-bar-wrap" style="margin:0.75rem 0 1rem">
      <div class="sd-prob-bar-track" style="background:var(--safe-r)">
        <div class="sd-prob-bar-fill" style="width:${demWinPct}%; background:var(--safe-d);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-top:0.3rem">
        <span style="color:var(--safe-d)">Dem ${demWinPct}%</span>
        <span style="color:var(--safe-r)">Rep ${repWinPct}%</span>
      </div>
    </div>

    <table class="sd-table">
      <thead>
        <tr><th>Year</th><th class="td-dem">Dem</th><th class="td-rep">Rep</th></tr>
      </thead>
      <tbody>
        <tr class="sd-table-projected">
          <td>2026 Projected</td>
          <td class="td-dem">${demShare}%</td>
          <td class="td-rep">${(100 - parseFloat(demShare)).toFixed(1)}%</td>
        </tr>
        ${histRows}
      </tbody>
    </table>

    <div class="sd-section-head">Top Predictive Factors</div>
    ${shapHtml}
  `;
}


function buildShapHtml(shapArr) {
  if (!shapArr.length) return '<p style="font-size:.8rem;color:var(--text-muted)">No factor data.</p>';

  const maxAbs = Math.max(...shapArr.map(s => Math.abs(s.impact)), 0.001);

  return shapArr.map(s => {
    const label     = FEATURE_LABELS[s.feature] || s.feature;
    const pct       = Math.round((Math.abs(s.impact) / maxAbs) * 100);
    const direction = s.impact >= 0 ? '→ D' : '→ R';
    const barClass  = s.impact >= 0 ? 'shap-pos' : 'shap-neg';
    const valStr    = (s.impact >= 0 ? '+' : '') + s.impact.toFixed(3);

    return `
      <div class="shap-row">
        <div class="shap-label">
          <span>${label} <span style="font-size:.72rem;color:var(--text-muted)">${direction}</span></span>
          <span class="shap-impact-val">${valStr}</span>
        </div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${barClass}" style="width:${pct}%"></div>
        </div>
      </div>`;
  }).join('');
}

document.getElementById('sidebar-close').addEventListener('click', () => {
  document.getElementById('sidebar').classList.remove('open');
});

/* ── Headline + distribution bar ─────────────────────────── */
function computeHeadline(preds) {
  const counts = { 'Safe D':0, 'Lean D':0, 'Toss-up':0, 'Lean R':0, 'Safe R':0 };
  preds.forEach(p => { if (counts[p.rating] !== undefined) counts[p.rating]++; });

  const demSeats = counts['Safe D'] + counts['Lean D'];
  document.getElementById('dem-seats').textContent = demSeats;

  document.getElementById('ct-safe-d').textContent  = counts['Safe D'];
  document.getElementById('ct-lean-d').textContent  = counts['Lean D'];
  document.getElementById('ct-tossup').textContent  = counts['Toss-up'];
  document.getElementById('ct-lean-r').textContent  = counts['Lean R'];
  document.getElementById('ct-safe-r').textContent  = counts['Safe R'];

  const total = preds.length;
  document.getElementById('band-safe-d').style.width = (counts['Safe D']  / total * 100) + '%';
  document.getElementById('band-lean-d').style.width = (counts['Lean D']  / total * 100) + '%';
  document.getElementById('band-tossup').style.width = (counts['Toss-up'] / total * 100) + '%';
  document.getElementById('band-lean-r').style.width = (counts['Lean R']  / total * 100) + '%';
  document.getElementById('band-safe-r').style.width = (counts['Safe R']  / total * 100) + '%';
}

/* ── Bootstrap ───────────────────────────────────────────── */
async function main() {
  initMap();

  const [preds, geojson, hist] = await Promise.all([
    fetch('data/predictions_2026.json').then(r => r.json()),
    fetch('data/districts.geojson').then(r => r.json()),
    fetch('data/history.json').then(r => r.json())
  ]);

  predictions = preds;
  history = hist;
  preds.forEach(p => { lookup[predKey(p.state_po, p.district)] = p; });

  addDistrictLayer(geojson);
  computeHeadline(preds);
}

/* ── Theme toggle ────────────────────────────────────────── */
(function() {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;

  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    btn.textContent = t === 'dark' ? '☀' : '☾';
    localStorage.setItem('theme', t);
  }

  const stored = localStorage.getItem('theme') || 'dark';
  applyTheme(stored);

  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });
})();

main().catch(err => {
  console.error('ElectMap load error:', err);
  document.getElementById('map').innerHTML =
    '<p style="padding:2rem;color:#c00">Failed to load map data. Run a local server: python -m http.server 8000 --directory site</p>';
});
