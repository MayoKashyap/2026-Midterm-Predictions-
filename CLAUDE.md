# PRD: 2026 Midterm Election Forecaster
**Version 3 — Current working document**

---

## Project Owner
Mayur Kashyap — CMU Mathematical Sciences, rising sophomore. Python-fluent, strong math background (calc, linear algebra, proof-based courses), weaker in applied probability/stats going in. Learning ML as he builds. GitHub: github.com/MayoKashyap

---

## Purpose of This Document
This PRD is the single source of truth for the full build. Claude Code should:
- Treat each milestone as a unit of work with a clear done condition — not a calendar deadline
- Confirm the deliverable is met and the user understands the math before moving on
- Never silently substitute synthetic data if a real source fails — flag it and ask
- Explain math before writing code (see Teaching Mode below)
- Update this document when scope changes rather than diverging from it silently

---

## Teaching Mode — Standing Rule
Applies for the entire project without exception.

Before writing or using any formula, model, or statistical method:
1. **Intuition first** — what real-world problem is this solving, in plain English, before any symbols
2. **The formula** — full notation, every term named and defined individually
3. **Why it's shaped this way** — what would break if a term were removed or changed
4. **Connection to the code** — point to exactly which line(s) implement which part of the formula

Write all math explanations as markdown cells in the Jupyter notebook, placed immediately above the code cell that implements the concept. Use proper LaTeX syntax (`$$...$$` for block, `$...$` for inline) — Jupyter renders this natively.

Do not let the user copy-paste a formula or library call without understanding it. Even if the user says "just give me the code," give the math explanation first, briefly, then the code. This is a standing rule, not a one-time request.

No inline code comments. Explanations belong in notebook markdown cells only. Use `os.path.expanduser` for all file paths. Use `print(..., flush=True)` for progress output.

---

## Project Goal
Build a four-layer probabilistic forecasting system for the 2026 U.S. midterms (House + Senate) that:
1. Establishes a historical baseline prediction per race using past elections, incumbency, partisan lean, and economic conditions — backtested across multiple election cycles
2. Models how real-world events shift polling averages using an NLP pipeline trained on historical event-to-polling-movement data
3. Aggregates real polls using a Bayesian model that corrects for pollster bias, incorporates event-driven adjustments, tracks opinion drift, and corrects for the registered-voter vs. likely-voter gap using a trained turnout propensity model
4. Simulates the full election 10,000 times to produce win probabilities per race, accounting for correlated national and regional shocks
5. Ships as a real interactive website with a Leaflet + D3 map of all 435 congressional districts (and state-level Senate map) as the centerpiece
6. Releases publicly in stages as each milestone completes, with LinkedIn posts at key milestones to build an audience before November 2026

---

## Explicit Non-Goals
- Not predicting the presidency or any 2028 race
- Not modeling primaries — general election only
- Not building investor-grade infrastructure — portfolio/learning project
- A full transformer-based NLP model trained from scratch is out of scope — use pretrained models or the Claude API for event classification
- No Streamlit — the frontend is a real custom website

---

## Architecture

### Layer 1 — Fundamentals Model
XGBoost regressor predicting baseline Democrat vote share per race from historical results, incumbency, partisan lean, and economic conditions. Trained on 2010–2022 historical data. Backtested across 2018, 2020, and 2022 separately. Then run forward on 2026 feature values to generate actual 2026 predictions — this output is what the first version of the site renders.

### Layer 2 — Event Impact Model (Novel component)
NLP pipeline quantifying how much a real-world political event shifts the generic ballot polling average, and how long that shift persists. Trained on historical events matched to subsequent polling movement. Auto-ingests new events via GDELT. Output: a time-varying adjustment term fed into Layer 3.

### Layer 3 — Bayesian Poll Aggregator
PyMC model combining Layer 1 prior, Layer 2 event adjustments, likely-voter-corrected polls, and house-effect correction. Gaussian random walk for opinion drift. Output: posterior distribution over true support per race at each point in time.

### Layer 4 — Election Simulator
Monte Carlo simulation (10,000 runs) sampling from Layer 3 posteriors with correlated national and regional shock terms. Output: win probability per race, seat distribution, P(majority), tipping-point races, static JSON for frontend.

---

## Frontend Stack
- **Map library:** Leaflet + D3
- **Hosting:** TBD (Vercel, Netlify, or GitHub Pages — free static hosting)
- **Domain:** TBD — to be purchased separately
- **Data flow:** Python pipeline exports static JSON → frontend reads and renders it. Model and presentation fully decoupled. No live Python process serving the frontend.
- **Design target:** NYT/538 election page quality. Real site, not a dashboard. Must be mobile-readable.

---

## Data Sources

| Source | URL | Used For |
|---|---|---|
| MIT Election Lab | electionlab.mit.edu/data | Historical House + Senate results 1976–2022 |
| Cook PVI (119th Congress) | cookpolitical.com / GitHub mirrors | Partisan lean per district, post-2020-redistricting adjusted |
| FRED API | fred.stlouisfed.org | GDP, unemployment, approval — auto-refreshed |
| FEC candidate filings | fec.gov | 2026 incumbency data — who is running, retiring, open seats |
| Ballotpedia | ballotpedia.org | Supplementary 2026 candidate and incumbency data |
| 538 polling + generic ballot | projects.fivethirtyeight.com/polls | Poll data + 2018/20/22 forecasts for backtest |
| Redistricting Data Hub | redistrictingdatahub.org | Which districts were redrawn post-2020 Census |
| CCES | cces.gov.harvard.edu | 60k-respondent voter survey for likely-voter logistic regression |
| GDELT | gdeltproject.org | Event ingestion for Layer 2 |
| NewsAPI | newsapi.org | Supplementary headline source for Layer 2 |
| Census TIGER/Line GeoJSON | census.gov | Congressional district boundary shapes for Leaflet map |

---

## Known Data Considerations
- **Redistricting:** Cook PVI is the primary historical anchor since it's recalculated post-redistricting. Add explicit `redistricted` binary flag as a feature.
- **2026 incumbency data:** Most manual data pull in the pipeline. Pull from FEC filings + Ballotpedia. Flag any districts where incumbent status is unclear.
- **Senate data sparsity:** ~33-34 seats per cycle vs 435 House seats. Separate model, separate pipeline. Never merge House and Senate data.
- **Event training set is small:** Major political events are sparse. Regularize the Layer 2 model aggressively. Be honest in the writeup about this limitation.
- **GDELT noise:** Requires careful keyword and source filtering to isolate politically relevant events. This is a real data engineering task.

---

## Milestones

Milestones are units of work with clear done conditions, not calendar deadlines. Move on when the deliverable is met and the math is understood.

---

### M1 — House Data Pipeline ✅ COMPLETE
Clean merged district-year dataset (2010–2022) with vote share, Cook PVI, FRED economic indicators, redistricting flags, standardized district IDs. EDA notebook with sanity-check plots.

---

### M2 — Fundamentals Model + Backtest + 2026 Predictions
**Goal:** A trained, backtested XGBoost model that generates actual 2026 district-level predictions ready to render on the site.

**Tasks:**

*Model training and backtest:*
- Feature engineering: incumbency encoding (+1/0/-1), Cook PVI as numeric, log-transformed economic variables, `redistricted` flag, midterm-penalty indicator, presidential approval
- Set up FRED API auto-refresh so economic features always use current data
- Train/test splits across three cycles: train 2010–2016 → test 2018, train 2010–2018 → test 2020, train 2010–2020 → test 2022
- Download 538's published forecasts for 2018, 2020, 2022
- Compute RMSE and Brier Score for your model and 538's across all three cycles
- SHAP analysis — global feature importance + individual district explanations
- Write findings summary: where does your model beat 538, where does it underperform

*2026 predictions (required before site can launch):*
- Pull 2026 incumbency data from FEC filings and Ballotpedia — compile who is running in each of the 435 districts, flag retirements and open seats
- Pull current Cook PVI values for the 119th Congress
- Pull current FRED economic indicators via API
- Assemble a 2026 feature matrix (one row per district, all features populated)
- Run the trained model on the 2026 feature matrix — generate predicted Democrat vote share for all 435 districts
- Convert vote share predictions to win probabilities using normal CDF
- Export predictions as static JSON: district ID, predicted dem share, win probability, top 3 SHAP factors

**Done condition:** Trained model + multi-cycle backtest table (RMSE/Brier vs 538 across 2018/2020/2022) + SHAP plots + 2026 predictions JSON exported and verified sane (known safe/competitive/safe-R districts should read that way).

**Math to teach:** MSE loss function and why squared not absolute value, gradient boosting sequential tree construction and learning rate intuition, SHAP Shapley value derivation from cooperative game theory, Brier Score as a proper scoring rule, normal CDF for converting vote share to win probability.

---

### M2-SHIP — Site v1 Launch + LinkedIn Post 1
**Goal:** First public release. Fundamentals-only forecast, all 435 districts on an interactive map.

**Site requirements for v1:**
- Leaflet + D3 map rendering all 435 congressional district polygons from Census GeoJSON
- Color gradient (red → purple → blue) based on predicted Democrat win probability — not binary win/lose
- Click any district → sidebar panel showing: predicted win probability, predicted vote share, top 3 SHAP factors driving the prediction, 2022 actual result for comparison
- Headline number at top: "Democrats projected to win X seats — fundamentals model only"
- Seat distribution bar (how many safe D / lean D / toss-up / lean R / safe R)
- Clear label: "Version 1 — historical fundamentals only. Live polls and event model coming soon."
- Mobile readable

**LinkedIn Post 1:**
- Hook: "What does history say about 2026 before a single poll is counted?"
- Show the map as the visual
- 3-4 sentences on methodology (XGBoost, 12 years of data, backtested vs 538)
- Link to the site
- Frame explicitly as version 1 with more layers coming

---

### M3 — Likely-Voter Logistic Regression
**Goal:** Trained logistic regression predicting individual turnout propensity from CCES data.

**Tasks:**
- Download CCES 2018 and 2020 waves for training, 2022 for validation (Harvard Dataverse, free)
- Features: self-reported voting intention, past validated vote history, age, education, party registration, political interest, state
- Target: validated vote indicator (CCES matches respondents to actual voter files post-election)
- Train logistic regression, validate on 2022 wave
- Calibration plot: does 70% predicted propensity correspond to ~70% actual turnout in that group?
- Output: adjustment function that takes a poll's demographic breakdown and returns a likely-voter-adjusted topline

**Done condition:** Trained model + calibration plot + adjustment function that Layer 3 can call.

**Math to teach:** logistic function and why it maps to probabilities, log-odds interpretation, maximum likelihood estimation and cross-entropy loss, calibration curves and what they mean for a forecasting model.

---

### M4 — Event Impact Model
**Goal:** Trained model quantifying how much a political event shifts the generic ballot and how long that shift persists.

**Tasks:**

*Data pipeline:*
- Pull 538 historical generic ballot daily averages back to 2016
- Pull GDELT event data filtered to US political events for the same period
- For each major event, compute polling shift over 7-day and 14-day windows vs 7-day pre-event baseline
- Build labeled dataset: each row is one event, features are event encoding, target is polling shift

*Event encoding:*
- Use Claude API (claude-sonnet-4-6) to classify each event headline: type (economic/scandal/policy/international/candidate), severity (1–5), direction (favors D / favors R / neutral)

*Model:*
- Ridge regression on (event type, severity, direction, days-until-election, current partisan environment) → polling shift
- Fit exponential decay curve to post-event polling movement data
- Regularize aggressively given small training set

*Integration + auto-update:*
- Output: time-varying adjustment term — given events in past 30 days weighted by decay, what is the net estimated generic ballot shift?
- Set up GDELT or NewsAPI pull that runs on a schedule to ingest new events automatically

**Done condition:** Trained event regression + decay function + auto-ingestion pipeline + sanity check showing estimated impact of 3-4 major historical events reads directionally correct.

**LinkedIn Post 2 (after M4):**
- Hook: "I built a model that quantifies how much real-world events move election polls"
- Show a table of major 2026 events and their estimated polling impact
- Frame as the novel component of the project
- Link to updated site

**Math to teach:** why exponential decay is natural for fading effects, Ridge vs Lasso regularization and geometric intuition, handling small training sets, difference between an event shift estimate (a delta) and a probability.

---

### M5 — Bayesian Poll Aggregator
**Goal:** Working PyMC model combining all inputs, validated on 2022 before pointing at 2026.

**Tasks:**
- Acquire 2022 House poll data (538 archive) and current 2026 poll data
- Build PyMC model:
  - Prior mean = Layer 1 fundamentals prediction + Layer 2 event adjustment term
  - Gaussian random walk for opinion drift over time
  - House effects per pollster as latent variables
  - Each poll's topline passed through M3 likely-voter adjustment before being treated as observation
  - Sampling noise weighted by poll sample size
- Validate: run on 2022 polls, confirm posterior converges toward actual 2022 result
- Check convergence diagnostics via ArviZ (trace plots, R-hat near 1.0, effective sample size)
- Point validated model at live 2026 polls
- Set up periodic re-run trigger: when new polls or events arrive, re-run and update posteriors
- Export updated posteriors to static JSON for frontend

**Done condition:** Working PyMC model + 2022 validation check + posterior plots for competitive 2026 districts + updated JSON exported.

**LinkedIn Post 3 (after M5):**
- Hook: "The forecast now incorporates live polls — here's how the picture changed from fundamentals alone"
- Show before/after map comparison (fundamentals-only vs poll-adjusted)
- Link to updated site

**Math to teach:** Bayes' theorem from scratch (prior/likelihood/posterior), Gaussian random walk as smooth opinion drift, what MCMC is doing under the hood, R-hat convergence diagnostic, why house effects are latent variables not directly observed.

---

### M6 — House Simulator + Site v2
**Goal:** Full Monte Carlo simulator live, site updated with complete House forecast.

**Tasks:**
- Write simulation loop: draw national shock + regional shock terms, apply to all districts, add local noise from posterior variance, count seats above 50%, repeat 10,000 times
- Validate: distribution shape sanity check, tipping-point districts surface correctly
- Export full simulation output to static JSON: win probabilities, seat distribution, P(majority), tipping-point races
- Update site:
  - Map now shows poll-informed win probabilities (not just fundamentals)
  - District click-through shows full detail: SHAP breakdown, poll history, event impact, posterior time series
  - Seat distribution histogram
  - Backtest panel: multi-cycle accuracy vs 538
  - What-if slider: shift national environment ± N points, map updates client-side
  - Live events panel: recent events ingested by Layer 2 and their estimated impact

**Done condition:** Live public site with full House forecast, working district drill-down, backtest panel, and what-if slider.

**LinkedIn Post 4 (after M6):**
- Biggest drop — announce the full forecast properly
- Show the interactive map
- Include the backtest accuracy number vs 538

**Math to teach:** why correlated simulation differs from independent probability multiplication, national + regional shock terms and what they represent, law of large numbers and why 10,000 runs is sufficient.

---

### M7 — Senate Pipeline + Models
**Goal:** Adapt the full M1–M5 pipeline to Senate races. Separate data, separate models, same architecture.

**Tasks:**
- Pull Senate historical results from MIT Election Lab
- State-level features: Cook Senate Race Ratings, incumbent margin last race, years in office, candidate-specific signals
- Train Senate fundamentals model (separate XGBoost, backtest 2018/2020/2022)
- Apply M3 likely-voter adjustment to Senate polls
- Apply M4 event model (same national adjustment term)
- Build Senate PyMC aggregator (same structure as M5)
- Build Senate simulator (state-level, correlated shocks, tipping-point states surfaced explicitly)
- Export Senate JSON in same format as House

**Done condition:** Senate fundamentals model + backtest + aggregator + simulator + JSON exported.

---

### M8 — Combined Site, Polish, Writeup, Deploy
**Goal:** Final public deliverable before November.

**Tasks:**
- Merge House and Senate into one site with chamber toggle
- Polish visual design — clean, mobile-first, reads as a real published site
- Write README: motivation, four-layer architecture, data sources, multi-cycle backtest results, known limitations, what November will tell us
- Write ~500 word findings post for LinkedIn/Substack
- Confirm public link works end to end on desktop and mobile

**LinkedIn Post 6 (M8 final):**
- Full project writeup post
- Architecture overview, key findings from backtest, what the model says about 2026

---

### POST-ELECTION — November 2026 (Most Important Milestone)
When results come in:
- Run full post-election accuracy analysis: compare district-level predictions to actual results across all 435 House races and all Senate races
- Compute final RMSE and Brier Score vs 538's published final accuracy
- Analyze where the event model helped vs hurt
- Write detailed post-mortem: what worked, what didn't, what would change next cycle

**LinkedIn Post 7 (post-election):**
- Most important post of the entire project
- Real accuracy vs 538 on actual results
- Honest analysis of where the model succeeded and failed
- This is what actually builds credibility

---

## LinkedIn Release Schedule

| Post | Trigger | Hook |
|---|---|---|
| 1 | M2-SHIP | "What does history say about 2026 before a single poll?" |
| 2 | M4 complete | "I built a model that quantifies how much events move polls" |
| 3 | M5 complete | "The forecast now incorporates live polls — here's what changed" |
| 4 | M6 complete | "Full 2026 House forecast is live" — biggest post |
| 5 | M7 complete | Senate added — only post if something interesting |
| 6 | M8 complete | Full project writeup |
| 7 | November 2026 | Post-election accuracy analysis — most important |

Post on Tuesday or Wednesday mornings for highest LinkedIn engagement. Under 200 words per post. One strong visual per post. Don't over-explain the math — make people curious enough to click through to the site.

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Data cleaning | pandas |
| Fundamentals model | XGBoost, scikit-learn |
| Model explanation | SHAP |
| Likely-voter model | scikit-learn LogisticRegression |
| Event classification | Claude API (claude-sonnet-4-6) |
| Event impact regression | scikit-learn Ridge |
| Bayesian inference | PyMC |
| Posterior diagnostics | ArviZ |
| Simulation | NumPy |
| Model output | Static JSON exported from Python |
| Map | Leaflet + D3 |
| Frontend | Custom HTML/CSS/JS or lightweight framework |
| Hosting | Vercel, Netlify, or GitHub Pages |
| Economic data refresh | FRED API (fredapi Python package) |
| Event ingestion | GDELT Python client + NewsAPI |
| 2026 candidate data | FEC API + Ballotpedia |

---

## Project Rating
**Current (M1 complete): 6.5/10**

**Target (all milestones complete + November post-mortem): 8.5/10**

The event impact model is the only genuinely original component. The multi-cycle backtest is what makes the fundamentals model credible. The November post-election analysis is what turns this from a portfolio project into something with a real track record.

Gap to 9+: event training data is sparse, Senate is compressed, no finding that would surprise a professional in the space. The November post-mortem is the most realistic path to closing that gap.
