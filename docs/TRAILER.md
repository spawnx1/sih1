# MULE CONNECTION — 60-second Cinematic Product Trailer
### Full production package (concept · storyboard · timeline · VO · sound · VFX · asset list · AI prompts)

> Product working name: **Mule Connection** (engine: the SIH26184 Cash-out Forecast Console).
> Every UI beat below is a **real, implemented** feature of the project — nothing fabricated.
> The only stylised (non-literal-UI) beat is the 0–8s abstract intro, which **match-cuts into the real Mule Network graph**.

---

## 0. What the film is about (one idea)

> **"Behind every transaction is a connection. Some connections hide a network."**

Emotional arc: *ordinary → uneasy → revealed → mastered.* A single ₹ transaction multiplies into chaos, the system imposes order, and the hidden mule network becomes a clean, navigable intelligence map. Apple-launch restraint throughout: black, charcoal, white, cool cyan; **red only for risk**.

**Tagline options (pick one):**
1. *See the connections. Stop the network.*
2. *Don't just detect the transaction. Understand the network.*
3. *The network was always there. Now you can see it.*  ← recommended

---

## 1. Real features used (and where each appears)

| # | Real feature (file) | Trailer role |
|---|---|---|
| A | **Mule Network** — force-directed codenamed entity graph (`/network`, `web/index.html` `showNetwork`) | The hero. The "hidden network" reveal (10s, 45–52s). |
| B | **Live money-flow map** — victim→mules→ATM red waypoint lines, ATM markers (`drawFlow`, Leaflet) | "Follow the money" (30–36s). |
| C | **Money-mule network (per case)** — victim(blue)→mules→red holding node (`muleGraphSVG`) | "Who is holding the money" (26–30s). |
| D | **Mule Profile** — one codenamed mule → 11 victims/cases, chain, history (`/account/{id}`, `openMule`) | The gut-punch reveal: *one account, many victims* (38–43s). |
| E | **Cinematic Replay** — letterbox, timecode, act labels, "⚠ CASH-OUT WINDOW" (`togglePlay`) | Tension climax (36–40s). |
| F | **Command Centre** — metro risk map + KPIs (`renderOverview`) | The "system" establishing shot (18–24s). |
| G | **Cash-out forecast** — risk %, when/where/how, hazard (`/predict`, XGBoost M1/M2/M3) | "The intelligence" (32–36s). |
| H | **Risk tiers + freeze** — RED/AMBER/GREY, freeze recommendation | Red accent language throughout. |

Behaviour colours in the graph are meaningful: **amber = cash-out specialist, teal = fan-out distributor, violet = collector**. Use them; don't recolour.

---

## 2. The 60-second concept (beat map)

| Beat | Time | Title | What the viewer feels |
|---|---|---|---|
| 1 | 0–8s | The trace | quiet, ordinary |
| 2 | 8–18s | The hidden network | unease — it's not isolated |
| 3 | 18–26s | Meet the system | authority, arrival |
| 4 | 26–36s | The intelligence | sophistication |
| 5 | 36–43s | One account. Many victims. | the reveal (D + E) |
| 6 | 43–52s | Understand the network | mastery, calm |
| 7 | 52–60s | Logo | resolution |

---

## 3. Shot-by-shot storyboard

Each shot: **timestamp · shot · camera · composition · on-screen (real feature) · text · transition · sound · VO.**

### SHOT 1 — 0:00–0:03 · Black → first trace
- **Shot:** Pure black. A single hairline of light draws left-to-right; a tiny label resolves: `₹2,500 · A → B`.
- **Camera:** Locked. Micro slow push-in (2%).
- **Composition:** One line, dead-centre, vast negative space.
- **On-screen:** *Stylised* (motion graphics) — represents a real transaction edge from `drawFlow`.
- **Text:** none.
- **Transition:** in from black.
- **Sound:** near-silence; one soft sub-bass swell; a single tiny UI *tick* as the label resolves.
- **VO:** —

### SHOT 2 — 0:03–0:08 · Traces multiply
- **Shot:** A second line. A third. Then dozens, fast but elegant, each a faint `₹ A→B`. Camera pulls back; the lines begin to knot into a lattice.
- **Camera:** Slow, deliberate pull-out.
- **Composition:** Symmetrical bloom from centre; shallow DOF, edges soft.
- **On-screen:** stylised edges (same visual grammar as the real graph — thin cyan lines, small nodes).
- **Text:** **"Every transaction leaves a trace."** (large, thin, lower third, fades in/out).
- **Transition:** continuous.
- **Sound:** ticks accumulate into a soft rhythmic bed; low pulse rises.
- **VO (optional):** *"Every transaction tells a story."*

### SHOT 3 — 0:08–0:14 · Into the lattice
- **Shot:** Camera flies *through* the transaction lattice; nodes rush past in shallow focus; connections form ahead.
- **Camera:** Smooth dolly-through; slight parallax.
- **Composition:** Depth layers of nodes; a few resolve sharp, most bokeh.
- **On-screen:** stylised → begins matching the **real Mule Network** node/edge style (colours still muted).
- **Text:** —
- **Transition:** —
- **Sound:** airy movement; sub-bass drone; a rising filtered tension note.

### SHOT 4 — 0:14–0:18 · First red
- **Shot:** Deep in the lattice, three nodes and their links **flush red**, then a small cluster. Everything else desaturates to charcoal.
- **Camera:** Slows to a near-stop, locks on the red cluster.
- **Composition:** Red cluster off-centre (rule of thirds); rest recedes.
- **On-screen:** stylised suspicious sub-graph (mirrors a real "cell" of shared-case mules).
- **Text:** **"Some connections aren't accidental."**
- **Transition:** hard cut to black on the last word (0.2s) → **match cut** to real UI.
- **Sound:** the bed drops out; one clean, restrained **bass impact** on the cut to black.

### SHOT 5 — 0:18–0:24 · System reveal (Command Centre)
- **Shot:** From black, the **Command Centre** fades up — the metro risk map glowing, KPI tiles crisp. A slow push-in across the dashboard.
- **Camera:** Smooth push-in, slight downward tilt, like revealing a device.
- **Composition:** UI floating on black, generous margins; reflection/vignette.
- **On-screen (REAL):** feature **F** — `renderOverview` (map + KPIs). Capture at 4K, dark, sidebar collapsed.
- **Text:** **"Meet Mule Connection."** (Apple-reveal typography, centred, held 1.5s).
- **Transition:** cross-dissolve from black.
- **Sound:** a single warm, resonant tone (the "product on" chord); soft UI hum.
- **VO (optional):** *"But some stories are hidden in the connections."*

### SHOT 6 — 0:24–0:30 · The money moves (map + case graph)
- **Shot:** Cut to a **case**: the **live money-flow map** with the red waypoint line drawing victim → mules → ATM; then a quick, elegant push to the **money-mule network** (blue victim → mules → red holding node).
- **Camera:** Lateral drift across the map, then push-in on the red holding node.
- **Composition:** Red line leads the eye diagonally; holding node lands on a thirds point.
- **On-screen (REAL):** features **B** + **C**.
- **Text:** **"Follow the money."** (small, lower third).
- **Transition:** map line → graph edge **match cut** (a red line becomes a graph link).
- **Sound:** a soft "draw" whoosh as the line extends (restrained); ticking as nodes light.

### SHOT 7 — 0:30–0:36 · The intelligence (forecast)
- **Shot:** The forecast panel: risk **%** counts up, the hazard curve rises, "when / where / how" resolve; a tier flips to **RED**. Overlay a subtle "noise → pattern → detection" motion on the graph behind.
- **Camera:** Slow push-in on the risk figure; rack focus from curve to the RED tier.
- **Composition:** Big number left, curve right; red tier badge as the punctuation.
- **On-screen (REAL):** feature **G** (`/predict` outputs — p_cashout %, hazard curve, location, channel).
- **Text:** **"Detect the pattern."**
- **Transition:** —
- **Sound:** rising tension; a clean rising synth line resolves on the RED flip with one bass note.
- **VO (optional):** *"Mule Connection reveals the network behind the transaction."*

### SHOT 8 — 0:36–0:40 · Cash-out window (Replay climax)
- **Shot:** The **cinematic Replay** takes over full-frame: letterbox bars slide in, timecode ticks `T +… min`, act label reads **"⚠ CASH-OUT WINDOW — FREEZE"**, the flow pulses red toward the ATM.
- **Camera:** Locked, letterboxed; the UI's own cinematic motion carries it.
- **Composition:** Cinemascope; red converging lines.
- **On-screen (REAL):** feature **E** (`togglePlay` replay at its climax).
- **Text:** —
- **Transition:** —
- **Sound:** heartbeat-like sub pulses accelerate; then a held breath (near-silence) on the freeze frame.

### SHOT 9 — 0:40–0:43 · One account. Many victims. (Profile)
- **Shot:** Snap to the **Mule Profile** — codename **NOMAD-2405** at the top; the "LINKED CASES — the same mule across **11** complaints" table fills downward; the **VICTIMS: 11** chip pulses.
- **Camera:** Quick push-in on the "11" then a slow reveal-scroll of the case list.
- **Composition:** Codename hero; the growing list = the weight of it.
- **On-screen (REAL):** feature **D** (`/account/ACC_02405`).
- **Text:** **"One account. Eleven victims."** (or keep silent and let the list speak).
- **Transition:** hard, confident cut in.
- **Sound:** a low, resonant impact as the codename lands; soft ticks per case row.

### SHOT 10 — 0:43–0:52 · Understand the network (hero pull-back)
- **Shot:** Pull back from the single profile into the **full Mule Network** — the codenamed graph settling, clusters forming, the amber/teal/violet behaviour lobes resolving. The camera keeps pulling until the whole web is one clean map on black.
- **Camera:** Long, smooth pull-out (the signature move); ends locked, centred.
- **Composition:** The entire network centred, breathing gently; labels legible on the hubs.
- **On-screen (REAL):** feature **A** — the hero frame you captured.
- **Text:** **"Don't just detect the transaction."** … (0.6s pause) … **"Understand the network."**
- **Transition:** everything else dissolves away, leaving the graph.
- **Sound:** the tension resolves into a single sustained, warm chord; ticking fades to one slow pulse.
- **VO (optional):** *"Detect the pattern. Understand the network."*

### SHOT 11 — 0:52–0:60 · Logo
- **Shot:** The network gently collapses to a single point of light, then the wordmark resolves.
- **Camera:** Locked.
- **Composition:** Centred wordmark **Mule Connection**; smaller line beneath.
- **On-screen:** logo (see §11 for a quick wordmark spec).
- **Text:** **Mule Connection** → (1s) → small: **"See the connections. Stop the network."**
- **Transition:** fade to black.
- **Sound:** one final, soft, expensive sub-bass *thump*; then silence.

---

## 4. Exact timeline (cue sheet)

```
0:00  black · first trace · tick
0:03  traces multiply · TEXT "Every transaction leaves a trace."
0:08  fly through lattice
0:14  first RED cluster · TEXT "Some connections aren't accidental." → cut to black @0:18
0:18  Command Centre reveal · TEXT "Meet Mule Connection." · product chord
0:24  money-flow map + mule graph · TEXT "Follow the money."
0:30  forecast: risk % + hazard + RED tier · TEXT "Detect the pattern."
0:36  Replay climax "⚠ CASH-OUT WINDOW"
0:40  Mule Profile · "11 complaints" · TEXT "One account. Eleven victims."
0:43  pull back to full Mule Network · TEXT "Understand the network."
0:52  network → point of light
0:54  wordmark "Mule Connection"
0:57  sub: "See the connections. Stop the network."
0:60  black · final thump · silence
```

---

## 5. Voiceover script (optional — the film works silent)

Calm, deep, measured; ≤ 22 words total; large gaps.

```
(0:04)  Every transaction tells a story.
(0:20)  But some stories are hidden in the connections.
(0:32)  Mule Connection reveals the network behind the transaction.
(0:45)  Detect the pattern. Understand the network.
```

If silent: keep only the on-screen text (§6) + sound design.

---

## 6. On-screen text (all of it)

1. Every transaction leaves a trace.
2. Some connections aren't accidental.
3. Meet Mule Connection.
4. Follow the money.
5. Detect the pattern.
6. One account. Eleven victims.
7. Don't just detect the transaction.
8. Understand the network.
9. **Mule Connection**
10. See the connections. Stop the network.

**Type spec:** one weight, tight tracking, generous scale. Suggested: *Inter / SF Pro Display* (Medium), white `#F5F7FA`, on black; fade 0.4s in / 0.4s out; never more than one line on screen; align centre (statements) or lower-third (labels).

---

## 7. Sound design spec

- **Bed:** a single sustained sub-bass drone (~40–55 Hz), barely there, that swells at each beat change.
- **Texture:** soft digital *ticks* (real UI click samples), used as rhythm as traces multiply and case rows appear.
- **Tension:** a filtered rising note from 0:30 that resolves on the RED tier flip (0:35) and again on the pull-back (0:43).
- **Impacts:** at most **3** restrained bass hits — the cut to black (0:18), the codename land (0:40), the final logo (0:60).
- **Product chord:** one warm, resonant tone on "Meet Mule Connection" (0:18).
- **Silence is a tool:** near-silence at 0:00–0:03 and on the freeze-frame (0:39).
- **Avoid:** cyberpunk arps, dubstep, whooshes stacked, Hollywood risers. Reference: Apple "Privacy. That's iPhone." / M-series launch films.

---

## 8. Camera & VFX direction (global)

- **Moves:** only push-ins, pull-outs, lateral drifts, and one fly-through. Ease-in-out on everything; no snap zooms (except the two intentional hard cuts at 0:18 and 0:40).
- **DOF:** shallow; foreground/background bokeh on the graph beats.
- **Grade:** deep blacks, charcoal mids, cool cyan lift in highlights; **red only** on risk. Subtle film grain (very low), gentle vignette, faint bloom on glowing nodes.
- **UI treatment:** composite the real screen recordings on black "glass" with a soft reflection and 2–4° perspective tilt on reveals (Shots 5, 9) so the dashboard reads as a *product*, not a screenshot.
- **VFX minimalism:** the transaction-to-network animation (Shots 1–4) is the only heavy motion-graphics; everything else is real UI + camera + grade.

---

## 9. Real-feature → shot map (quick reference)

| Shot | Time | Real feature |
|---|---|---|
| 5 | 0:18 | **F** Command Centre (map + KPIs) |
| 6 | 0:24 | **B** money-flow map + **C** mule graph |
| 7 | 0:30 | **G** forecast (risk %, hazard, tier) |
| 8 | 0:36 | **E** cinematic Replay |
| 9 | 0:40 | **D** Mule Profile (11 cases) |
| 10 | 0:43 | **A** Mule Network (hero) |

Shots 1–4 = stylised intro (motion graphics), Shot 11 = logo.

---

## 10. Production / editing instructions

**Tooling (any one):** DaVinci Resolve (free, best grade) or After Effects. Edit at **3840×2160, 24fps** (cinematic), export H.264/H.265.

**Pipeline:**
1. **Capture the real UI** as clean screen recordings (see §11) at 4K, dark theme, sidebar collapsed, no cursor. Record each beat 8–10s so you have handles.
2. **Build the intro (0–18s)** in AE/Resolve Fusion or a generative tool (see §12) — a particle/line network that adopts the real graph's colours (cyan lines, small nodes, amber/teal/violet accents), ending on a frame that matches the real Mule Network so the 0:18 cut lands.
3. **Composite UI on black glass:** place each recording on a black background, add a soft floor reflection, 2–4° tilt on the reveal shots, subtle drop shadow, gentle bloom.
4. **Animate camera** with keyframed scale/position (push-in/pull-out) on the UI recordings — don't rely on the app's own motion except for the Replay (Shot 8), which is already cinematic.
5. **Grade** to the palette in §8; ensure red only appears on risk elements.
6. **Type pass** — add the §6 text with the spec's fades.
7. **Sound pass** — lay the bed, ticks, tension, 3 impacts, product chord (§7). Mix quiet; leave headroom; let silence breathe.
8. **Final:** 60.0s exactly; fade last frame to black; one final thump; 1s of black.

**Do:** keep it one coherent film, deliberate pacing, lots of black. **Don't:** fill silence, stack transitions, or add stock footage.

---

## 11. Assets to capture (from the running project)

Run the app (`uvicorn api.main:app --port 8000`), dark theme, **Advanced mode**, sidebar collapsed, browser at 4K/zoomed for crisp text. Capture (screen-record 8–10s each, no cursor):

1. **Mule Network** (hero) — `Mule Network` tab; let it settle; slow manual zoom-out. *(You already have the hero still.)*
2. **Command Centre** — landing view; metro map + KPI tiles.
3. **Case money-flow map** — open a RED case; the red waypoint line victim→mules→ATM; slow pan.
4. **Money-mule network (per case)** — the "who is holding the money" graph; hover a red node.
5. **Forecast panel** — the risk %, hazard curve, RED tier, when/where/how.
6. **Cinematic Replay** — press *Replay simulation*; capture through the "⚠ CASH-OUT WINDOW" climax (this is your most cinematic real asset).
7. **Mule Profile** — click **NOMAD-2405** (ACC_02405); the "11 complaints" list; slow scroll.
8. **Logo plate** — see wordmark below.

**Wordmark (quick):** set **Mule Connection** in Inter/SF Pro Display Medium, white on black, letter-spacing +2%; optional tiny cyan dot or a single node-glyph as the "connection." Animate: fade in + 1px cyan underline draw. (If you want, I can generate a clean SVG/PNG wordmark plate.)

---

## 12. Exact prompts for AI-generated visuals (intro only, 0–18s)

Use these for a tool like Runway / Kling / Sora / Pika, or as a look-reference for AE particle work. Keep them consistent with the real palette.

**Prompt A (0:00–0:08 — traces multiply):**
> "Extreme minimal cinematic macro, pure black background. A single thin luminous cyan line draws between two tiny glowing points, like a financial transaction. Then more thin cyan lines appear one by one and multiply into an elegant interconnected lattice of small glowing nodes. Slow deliberate camera pull-back, shallow depth of field, soft bokeh, deep blacks, charcoal, subtle cool cyan accents, no text, no people, premium Apple-commercial aesthetic, controlled lighting, gentle bloom, 4k, 24fps."

**Prompt B (0:08–0:14 — fly-through):**
> "Cinematic slow camera flight through a dark 3D network of small glowing cyan nodes connected by thin luminous lines, financial data network, shallow depth of field, nodes rushing past in soft focus, deep black environment, minimal, elegant, no text, Apple product film look, subtle particles, 4k."

**Prompt C (0:14–0:18 — first red):**
> "Dark cinematic network of glowing cyan nodes; three connected nodes and their links suddenly turn deep red, spreading to a small red cluster while everything else desaturates to charcoal. Camera slows and locks on the red cluster, shallow depth of field, ominous restraint, minimal, premium, no text, 4k, 24fps."

**Negative / avoid (all):** "no green matrix code, no dense hacker text, no neon cyberpunk, no lens flares stacked, no busy backgrounds, no logos, no watermarks, no people, no fast strobing."

> After the AI intro, **cut to the real recordings** — do not AI-generate the dashboard; the product's real UI is the point.

---

## 13. One decision for you
- **Product name on screen:** I used **"Mule Connection"** (your words). The engine's internal name is "Cash-out Forecast Console (SIH26184)." Tell me which should appear in Shots 5 & 11 — I'd keep **Mule Connection** as the product, and optionally a tiny subtitle *"Cash-out intelligence · I4C"*.

*All UI beats reference real, implemented features. Silent-first; VO optional. 60.0s. — generated for SIH26184 / Mule Connection.*
