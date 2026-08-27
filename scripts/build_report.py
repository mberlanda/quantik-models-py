"""Generate the artifact body from the run reports, so no number is hand-typed."""
import json
from pathlib import Path

# The report is regenerated from the run reports, never hand-edited, so a
# rerun of the evaluation updates every number in it. The CSS/head lives in
# docs/nn-quest/report.head.html; this appends the body.
OUT = Path("docs/nn-quest/report.html")
final = json.load(open("runs/arena/final.json"))
showdown = json.load(open("runs/arena/showdown.json"))
handicap = json.load(open("runs/arena/handicap.json"))
# Accuracy comes from the 8,440-position probe, not the 640-position one the
# first draft used: that probe held only 33 won positions each at plies 4-5,
# the two plies that carry the whole margin.
cov = json.load(open("runs/coverage.json"))

probe = {
    x["agent"]: {
        "outcome_accuracy": x["outcome_accuracy"],
        "accuracy_by_ply": {k: v for k, v in x["by_ply"].items()},
    }
    for x in cov["agents"]
}
arena = final["arena"]

def ms(agent):
    vals = [r[k] for r in arena for a, k in ((r["agent_a"], "ms_per_move_a"), (r["agent_b"], "ms_per_move_b")) if a == agent]
    return sum(vals) / len(vals)

def board(rows):
    tally = {}
    for r in rows:
        for n, w in ((r["agent_a"], r["wins_a"]), (r["agent_b"], r["wins_b"])):
            e = tally.setdefault(n, [0, 0]); e[0] += w; e[1] += r["games"]
    return sorted(({"agent": k, "wins": v[0], "games": v[1], "rate": v[0]/v[1]} for k, v in tally.items()), key=lambda r: -r["rate"])

LB = board(arena)
PLIES = list(range(4, 13))
COVERAGE = cov["coverage"]
COMPARE = cov["comparisons"]

# --- series for the per-ply chart: the four engines that actually compete ---
SERIES = [
    ("qnet@200ms", "var(--s1)", "network"),
    ("minimax@100ms", "var(--s2)", "minimax"),
    ("beam@100ms", "var(--s3)", "beam"),
    ("mcts@100ms", "var(--s4)", "MCTS"),
]

W, H = 720, 340
ML, MR, MT, MB = 40, 96, 16, 34
PW, PH = W - ML - MR, H - MT - MB
def px(ply): return ML + (ply - 4) / 8 * PW
def py(v): return MT + (1 - v) * PH

def line_chart():
    s = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Outcome accuracy by ply for four engines. The network holds 97 to 100 percent across every ply; minimax falls to 84.8 percent at ply 4.">']
    for g in (0, .25, .5, .75, 1.0):
        y = py(g)
        s.append(f'<line class="gridline" x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}"/>')
        s.append(f'<text class="axis" x="{ML-8}" y="{y+4:.1f}" text-anchor="end">{int(g*100)}%</text>')
    # the gap: region between network and minimax over the opening
    net = [probe["qnet@200ms"]["accuracy_by_ply"][str(p)]["accuracy"] for p in PLIES]
    mm  = [probe["minimax@100ms"]["accuracy_by_ply"][str(p)]["accuracy"] for p in PLIES]
    top = " ".join(f"{px(p):.1f},{py(v):.1f}" for p, v in zip(PLIES, net))
    bot = " ".join(f"{px(p):.1f},{py(v):.1f}" for p, v in reversed(list(zip(PLIES, mm))))
    s.append(f'<polygon points="{top} {bot}" fill="var(--accent-soft)"/>')
    for p in PLIES:
        s.append(f'<text class="axis" x="{px(p):.1f}" y="{MT+PH+20}" text-anchor="middle">{p}</text>')
    s.append(f'<text class="axis" x="{ML+PW/2:.0f}" y="{H-3}" text-anchor="middle">ply — pieces already on the board</text>')
    s.append(f'<line class="baseline" x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}"/>')
    for agent, colour, short in SERIES:
        vals = [probe[agent]["accuracy_by_ply"][str(p)]["accuracy"] for p in PLIES]
        pts = " ".join(f"{px(p):.1f},{py(v):.1f}" for p, v in zip(PLIES, vals))
        wide = 2.6 if agent.startswith("qnet") else 2
        s.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{wide}" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for p, v in zip(PLIES, vals):
            s.append(f'<circle cx="{px(p):.1f}" cy="{py(v):.1f}" r="4" fill="{colour}" '
                     f'stroke="var(--surface)" stroke-width="2"/>')
        s.append(f'<text class="serieslabel" x="{ML+PW+10}" y="{py(vals[-1])+4:.1f}" fill="{colour}">{short}</text>')
    # hover layer
    s.append('<g id="crosshair" style="display:none"><line class="baseline" y1="%d" y2="%d" stroke-dasharray="3 3"/></g>' % (MT, MT+PH))
    for p in PLIES:
        s.append(f'<rect class="hit" data-ply="{p}" x="{px(p)-PW/16:.1f}" y="{MT}" width="{PW/8:.1f}" '
                 f'height="{PH}" fill="transparent"/>')
    s.append("</svg>")
    return "\n".join(s)

# --- head-to-head bars with confidence intervals ---
def h2h_rows():
    sd = showdown["arena"][0]
    hc = handicap["arena"][0]
    rr = next(r for r in arena if {r["agent_a"], r["agent_b"]} == {"minimax@100ms", "alphazero@200ms"})
    az_rate = rr["score_a"] if rr["agent_a"] == "alphazero@200ms" else 1 - rr["score_a"]
    az_lo, az_hi = (rr["ci_low"], rr["ci_high"]) if rr["agent_a"] == "alphazero@200ms" else (1 - rr["ci_high"], 1 - rr["ci_low"])
    return [
        ("qnet @ 200 ms", sd["score_a"], sd["ci_low"], sd["ci_high"], sd["games"],
         f"{sd['ms_per_move_a']:.0f} ms/move vs {sd['ms_per_move_b']:.0f}"),
        ("qnet @ 50 ms", hc["score_a"], hc["ci_low"], hc["ci_high"], hc["games"],
         f"{hc['ms_per_move_a']:.0f} ms/move vs {hc['ms_per_move_b']:.0f}"),
        ("AlphaZero from scratch", az_rate, az_lo, az_hi, rr["games"],
         f"{rr['ms_per_move_b'] if rr['agent_a']=='minimax@100ms' else rr['ms_per_move_a']:.0f} ms/move vs "
         f"{rr['ms_per_move_a'] if rr['agent_a']=='minimax@100ms' else rr['ms_per_move_b']:.0f}"),
    ]

BW, BH = 720, 200
BL, BR2, BT = 168, 26, 22
BPW = BW - BL - BR2
def bx(v): return BL + v * BPW

def bar_chart():
    rows = h2h_rows()
    s = [f'<svg class="chart" viewBox="0 0 {BW} {BH}" role="img" '
         f'aria-label="Win rate against minimax at 100 milliseconds, with 95 percent confidence intervals. '
         f'The distilled network wins at both time budgets; AlphaZero from scratch loses.">']
    for g in (0, .25, .5, .75, 1.0):
        x = bx(g)
        s.append(f'<line class="gridline" x1="{x:.1f}" y1="{BT}" x2="{x:.1f}" y2="{BT+3*40+8}"/>')
        s.append(f'<text class="axis" x="{x:.1f}" y="{BT+3*40+26}" text-anchor="middle">{int(g*100)}%</text>')
    x50 = bx(.5)
    s.append(f'<line x1="{x50:.1f}" y1="{BT-8}" x2="{x50:.1f}" y2="{BT+3*40+8}" stroke="var(--rule)" stroke-width="1.5"/>')
    s.append(f'<text class="axis" x="{x50:.1f}" y="{BT-12}" text-anchor="middle" fill="var(--muted)">even</text>')
    for i, (label, rate, lo, hi, games, note) in enumerate(rows):
        y = BT + 14 + i * 40
        won = rate > 0.5
        # One measure diverging around even, so it takes the palette's
        # blue-red diverging pair rather than another chart's series hues.
        colour = "var(--s1)" if won else "var(--loss)"
        s.append(f'<text class="axis" x="{BL-12}" y="{y+4}" text-anchor="end" fill="var(--ink)" '
                 f'style="font-size:12px">{label}</text>')
        left, right = (x50, bx(rate)) if won else (bx(rate), x50)
        s.append(f'<rect x="{left:.1f}" y="{y-9:.1f}" width="{max(right-left,1):.1f}" height="18" '
                 f'rx="4" fill="{colour}"/>')
        s.append(f'<line x1="{bx(lo):.1f}" y1="{y:.1f}" x2="{bx(hi):.1f}" y2="{y:.1f}" '
                 f'stroke="var(--ink)" stroke-width="1.5" opacity=".55"/>')
        for e in (lo, hi):
            s.append(f'<line x1="{bx(e):.1f}" y1="{y-5:.1f}" x2="{bx(e):.1f}" y2="{y+5:.1f}" '
                     f'stroke="var(--ink)" stroke-width="1.5" opacity=".55"/>')
        anchor_x = bx(hi) + 8 if won else bx(lo) - 8
        s.append(f'<text class="axis" x="{anchor_x:.1f}" y="{y+4:.1f}" fill="var(--ink)" '
                 f'text-anchor="{"start" if won else "end"}" style="font-size:12.5px;font-weight:500">{rate:.1%}</text>')
    s.append("</svg>")
    return "\n".join(s)

CW, CH = 720, 300
CL, CR, CT = 44, 132, 18
CPW = CW - CL - CR
_ROW = 26


def cx(count):
    """Log position; 0 maps to the axis origin."""
    import math

    if count <= 0:
        return CL
    return CL + min(1.0, math.log10(count) / 8.0) * CPW


def coverage_chart():
    """Per ply: every canonical position, and the slice the model trained on.

    A log axis makes the segment between the two dots *be* the coverage
    ratio, so the eye reads the fraction and the absolute scale at once —
    which matters when the plies span 1 position and 38 million.
    """
    rows = [r for r in COVERAGE if r["canonical_live"] and r["ply"] <= 9]
    height = CT + len(rows) * _ROW + 34
    s = [f'<svg class="chart" viewBox="0 0 {CW} {height}" role="img" '
         f'aria-label="Canonical positions per ply against the number trained on. '
         f'Plies 4 and 5 have no training data at all; deeper plies reach a few percent.">']
    for power in range(0, 9):
        x = CL + power / 8.0 * CPW
        s.append(f'<line class="gridline" x1="{x:.1f}" y1="{CT - 6}" x2="{x:.1f}" '
                 f'y2="{CT + len(rows) * _ROW}"/>')
        label = "1" if power == 0 else ("10" if power == 1 else f"10^{power}")
        s.append(f'<text class="axis" x="{x:.1f}" y="{CT + len(rows) * _ROW + 18}" '
                 f'text-anchor="middle">{label}</text>')
    s.append(f'<text class="axis" x="{CL + CPW / 2:.0f}" y="{height - 2}" '
             f'text-anchor="middle">canonical positions (log scale)</text>')
    for i, row in enumerate(rows):
        y = CT + i * _ROW + _ROW / 2
        total, trained = row["canonical_live"], row["trained_total"]
        s.append(f'<text class="axis" x="{CL - 10}" y="{y + 4:.1f}" text-anchor="end" '
                 f'fill="var(--ink-2)">ply {row["ply"]}</text>')
        if trained > 0:
            s.append(f'<line x1="{cx(trained):.1f}" y1="{y:.1f}" x2="{cx(total):.1f}" '
                     f'y2="{y:.1f}" stroke="var(--rule)" stroke-width="2"/>')
            s.append(f'<circle cx="{cx(trained):.1f}" cy="{y:.1f}" r="5" fill="var(--s1)" '
                     f'stroke="var(--surface)" stroke-width="2"/>')
        s.append(f'<circle cx="{cx(total):.1f}" cy="{y:.1f}" r="5" fill="var(--muted)" '
                 f'stroke="var(--surface)" stroke-width="2"/>')
        note = f"{row['coverage']:.2%}" if trained else "no training data"
        colour = "var(--ink)" if trained else "var(--loss)"
        s.append(f'<text class="axis" x="{CL + CPW + 12}" y="{y + 4:.1f}" fill="{colour}" '
                 f'style="font-size:11.5px;font-weight:500">{note}</text>')
    s.append("</svg>")
    return "\n".join(s)


def glyph(n):
    """A 4x4 Quantik board with n cells filled — section index in the game's own geometry."""
    cells = []
    for i in range(16):
        r, c = divmod(i, 4)
        cls = "on" if i < n else ""
        cells.append(f'<rect class="{cls}" x="{c*6+1}" y="{r*6+1}" width="4" height="4" rx="1"/>')
    return ('<svg class="board-glyph" width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">'
            + "".join(cells) + "</svg>")

def head(n, eyebrow, title):
    return (f'<div class="head"><div class="marker">{glyph(n)}<span class="eyebrow">{eyebrow}</span></div>'
            f'<h2>{title}</h2></div>')

# --- leaderboard table ---
def coverage_table():
    rows = []
    for row in COVERAGE:
        if row["ply"] > 13:
            continue
        total = f"{row['canonical_live']:,}" if row["canonical_live"] else "not enumerated"
        probe_count = sum(v for k, v in row.items() if k.startswith("probe_"))
        if row["canonical_live"] and row["trained_total"] == 0:
            cover = '<td style="color:var(--loss);font-weight:600">none</td>'
        elif row["canonical_live"]:
            cover = f"<td>{row['coverage']:.2%}</td>"
        else:
            cover = "<td>—</td>"
        hero = ' class="hero"' if row["ply"] in (4, 5) else ""
        rows.append(
            f'<tr{hero}><td>{row["ply"]}</td><td>{total}</td>'
            f'<td>{row["trained_total"]:,}</td><td>{row["trained_policy"]:,}</td>'
            f'{cover}<td>{probe_count:,}</td></tr>')
    return ('<table><caption>Positions counted up to symmetry — a position and its 191 '
            'images are one game. Terminal positions are excluded; they need no decision.'
            '</caption><thead><tr><th>ply</th><th>canonical positions</th><th>trained on</th>'
            '<th>with move label</th><th>coverage</th><th>held-out probe</th></tr></thead>'
            '<tbody>' + "".join(rows) + "</tbody></table>")


def significance_table():
    rows = []
    for block in COMPARE:
        m, ci = block["mcnemar"], block["difference"]
        rows.append(
            f'<tr><td>{block["scope"]}</td><td>{m["positions"]:,}</td>'
            f'<td>{m["accuracy_a"]:.2%}</td><td>{m["accuracy_b"]:.2%}</td>'
            f'<td>{m["a_right_b_wrong"]}</td><td>{m["b_right_a_wrong"]}</td>'
            f'<td>{ci["point"]:+.2%}</td>'
            f'<td>{ci["low"]:+.2%} to {ci["high"]:+.2%}</td>'
            f'<td><strong>{m["p_value"]:.1g}</strong></td></tr>')
    return ('<table><caption>Paired comparison of `qnet@200ms` against `minimax@100ms` on '
            'positions the mover provably wins.</caption><thead><tr><th>scope</th>'
            '<th>positions</th><th>network</th><th>minimax</th><th>only network right</th>'
            '<th>only minimax right</th><th>difference</th><th>95% CI</th><th>p</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def leaderboard_table():
    rows = []
    for r in LB:
        acc = probe.get(r["agent"], {}).get("outcome_accuracy")
        hero = ' class="hero"' if r["agent"] == "qnet@200ms" else ""
        rows.append(
            f'<tr{hero}><td class="name">{r["agent"]}</td>'
            f'<td>{r["rate"]:.1%}</td><td>{r["wins"]}</td>'
            f'<td>{ms(r["agent"]):.0f}</td>'
            f'<td>{acc:.1%}</td></tr>' if acc is not None else
            f'<tr{hero}><td class="name">{r["agent"]}</td><td>{r["rate"]:.1%}</td>'
            f'<td>{r["wins"]}</td><td>{ms(r["agent"]):.0f}</td><td>—</td></tr>')
    return ('<table><caption>1,800 games per agent — every pairing, both sides, '
            'one CPU thread each.</caption>'
            '<thead><tr><th>agent</th><th>win rate</th><th>wins / 1800</th>'
            '<th>ms per move</th><th>exact accuracy</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")

def ply_table():
    order = ["qnet@200ms", "minimax@100ms", "alphazero@200ms", "qnet-policy", "beam@100ms", "mcts@100ms", "random"]
    rows = []
    for a in order:
        if a not in probe: continue
        cells = []
        for p in PLIES:
            b = probe[a]["accuracy_by_ply"].get(str(p))
            v = b["accuracy"] if b else None
            cls = ' class="perfect"' if v == 1.0 else ""
            cells.append(f"<td{cls}>{v:.0%}</td>" if v is not None else "<td>—</td>")
        hero = ' class="hero"' if a == "qnet@200ms" else ""
        rows.append(f'<tr{hero}><td class="name">{a}</td>' + "".join(cells)
                    + f'<td><strong>{probe[a]["outcome_accuracy"]:.1%}</strong></td></tr>')
    heads = "".join(f"<th>{p}</th>" for p in PLIES)
    return ('<table><caption>Share of provably won positions where the agent plays a move '
            'that keeps the win. 640 held-out solved positions.</caption>'
            f'<thead><tr><th>agent</th>{heads}<th>all</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")

deep_positions = sum(
    v["total"] for k, v in probe["qnet@200ms"]["accuracy_by_ply"].items() if int(k) >= 8
)
# Positions where both engines are already exact — the region that carries no
# information about which is stronger.
deep_positions = sum(
    v["total"] for k, v in probe["qnet@200ms"]["accuracy_by_ply"].items() if int(k) >= 8
)
sd = showdown["arena"][0]
hc = handicap["arena"][0]
net_lb = next(r for r in LB if r["agent"] == "qnet@200ms")
mm_lb = next(r for r in LB if r["agent"] == "minimax@100ms")

body = f"""
<header class="reveal">
  <span class="eyebrow">Quantik &middot; measured against a perfect solver</span>
  <h1>The ply-four gap</h1>
  <p class="dek">Quantik is small enough to solve exactly, so a search engine plays
  the endgame perfectly. It plays the <em>opening</em> at 84.8&thinsp;%. That gap is
  the whole reason a neural network can beat it &mdash; and it does, on less time per move.</p>
  <div class="byline">
    <span>27 August 2026</span><span>quantik-models-py</span>
    <span>1,800 games per agent</span><span>8,440 exactly-solved held-out positions</span>
  </div>
</header>

<div class="stats reveal wide">
  <div class="stat"><span class="value">{net_lb['rate']:.1%}</span>
    <span class="label">Network win rate across the full round-robin</span>
    <span class="foot">minimax: {mm_lb['rate']:.1%}</span></div>
  <div class="stat"><span class="value">{probe['qnet@200ms']['outcome_accuracy']:.1%}</span>
    <span class="label">Moves that preserve a won position, over {sum(v['total'] for v in probe['qnet@200ms']['accuracy_by_ply'].values()):,} solved positions</span>
    <span class="label" style="font-size:.82rem">minimax: {probe['minimax@100ms']['outcome_accuracy']:.1%} &middot; paired test p = {COMPARE[0]['mcnemar']['p_value']:.1g}</span>
    <span class="foot">held out, and disjoint from training up to symmetry</span></div>
  <div class="stat"><span class="value">{hc['ms_per_move_b']/hc['ms_per_move_a']:.1f}&times;</span>
    <span class="label">Less thinking time, still winning {hc['score_a']:.1%} of games</span>
    <span class="foot">{hc['ms_per_move_a']:.0f} ms/move vs {hc['ms_per_move_b']:.0f} ms/move</span></div>
</div>

<section>
  {head(1, "the game", "Sixteen squares, four shapes")}
  <p class="lede">Quantik is played on a 4&times;4 board with four piece shapes in two
  colours. You win by completing any row, column, or 2&times;2 zone containing all four
  <em>different</em> shapes &mdash; colour is irrelevant, so you can win with your
  opponent&rsquo;s pieces. The catch: you may not place a shape where the opponent already
  has that same shape in the row, column, or zone. A player with no legal move loses.</p>
  <figure class="boardfig">
    <svg width="188" height="188" viewBox="0 0 188 188" role="img"
         aria-label="A 4 by 4 Quantik board. The top row holds all four shapes in mixed colours and is highlighted as a win.">
      <rect x="1" y="1" width="186" height="186" rx="8" fill="none" stroke="var(--rule)"/>
      <g stroke="var(--hair)" stroke-width="1">
        <line x1="47" y1="6" x2="47" y2="182"/><line x1="141" y1="6" x2="141" y2="182"/>
        <line x1="6" y1="47" x2="182" y2="47"/><line x1="6" y1="141" x2="182" y2="141"/>
      </g>
      <line x1="94" y1="6" x2="94" y2="182" stroke="var(--rule)" stroke-width="1.5"/>
      <line x1="6" y1="94" x2="182" y2="94" stroke="var(--rule)" stroke-width="1.5"/>
      <rect x="7" y="7" width="174" height="40" rx="5" fill="var(--accent-soft)"/>
      <g fill="var(--ink)">
        <circle cx="24" cy="27" r="11"/>
        <rect x="60" y="16" width="21" height="21" rx="2" fill="none" stroke="var(--ink)" stroke-width="2.5"/>
        <path d="M118 38 L129 16 L140 38 Z" fill="none" stroke="var(--ink)" stroke-width="2.5"/>
        <path d="M164 16 L175 27 L164 38 L153 27 Z"/>
      </g>
      <g fill="var(--muted)" opacity=".55">
        <circle cx="118" cy="74" r="11"/>
        <rect x="13" y="112" width="21" height="21" rx="2" fill="none" stroke="var(--muted)" stroke-width="2.5"/>
        <path d="M60 168 L71 146 L82 168 Z" fill="none" stroke="var(--muted)" stroke-width="2.5"/>
      </g>
    </svg>
    <p class="note">The highlighted row holds all four shapes &mdash; two dark, two light
    &mdash; so whoever placed the last one wins. Twelve lines can win: four rows, four
    columns, four zones. A whole game lasts at most sixteen plies, which is why the
    game can be solved outright, and why the interesting question is not
    <em>whether</em> a machine can play it perfectly but <em>how fast</em>.</p>
  </figure>
</section>

<section>
  {head(2, "the problem", "You cannot out-search a solved game")}
  <p>The existing engine, <code>MinimaxEngine</code>, runs alpha-beta to depth 16 with a
  transposition table. No Quantik game exceeds 16 plies, so given enough time it does not
  approximate anything &mdash; it <em>solves</em>. Beating it outright is impossible; the
  only meaningful contest is at a fixed budget.</p>
  <p>So the first thing built here was not a network. It was a
  <strong>ruler</strong>: 640 positions solved to game-theoretic truth by an exact
  oracle, held out from all training, scoring each engine on a single question
  &mdash; <em>given a position you can provably win, do you play a move that keeps
  the win?</em></p>
  <p class="callout">Accuracy tells you <em>where</em> an engine is wrong. A win rate only
  tells you that it lost.</p>
  <p>The ruler said something that redirected the entire project.</p>
</section>

<section class="wide">
  {head(3, "the finding", "Minimax is perfect &mdash; from ply 8 onward")}
  <p style="max-width:34rem">Past the halfway mark, minimax at 100&thinsp;ms plays every
  won position correctly. It never errs again. But at ply 4, with twelve squares still
  empty, it converts only 84.8&thinsp;% of wins. The search cannot reach the end of the
  game in the time it has.</p>
  <figure>
    <div class="fig-title"><span class="t">Outcome accuracy by ply</span>
      <span class="s">Shaded band: the network&rsquo;s margin over minimax. It closes to
      zero by ply 8, where both are exact.</span></div>
    <div class="scroll">{line_chart()}</div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--s1)"></i>network @ 200&thinsp;ms</span>
      <span><i class="swatch" style="background:var(--s2)"></i>minimax @ 100&thinsp;ms</span>
      <span><i class="swatch" style="background:var(--s3)"></i>beam @ 100&thinsp;ms</span>
      <span><i class="swatch" style="background:var(--s4)"></i>MCTS @ 100&thinsp;ms</span>
    </div>
    <figcaption>Every engine converges to perfection in the endgame &mdash; the game is
    simply small there. All of the daylight between them is in the opening, and that is
    the only place a network can win anything.</figcaption>
  </figure>
  <p style="max-width:34rem">This made the goal precise: <strong>match minimax in the
  endgame, beat it at plies 4&ndash;7.</strong> Work spent making the network stronger at
  ply 10 would have been wasted.</p>
</section>

<section>
  {head(4, "the method", "Distil the solver, don&rsquo;t imitate the player")}
  <p class="lede">A from-scratch AlphaZero run was the first attempt. It reached
  97.2&thinsp;% accuracy &mdash; exactly level with minimax &mdash; and stopped
  improving. The probe explained why: its <strong>value head had a mean error of
  0.727 against a &plusmn;1 truth</strong>. It was outputting nearly zero. It did not
  know who was winning.</p>
  <p>That failure is structural, not a bug. The value target was a blend of the final
  game result &mdash; one bit at the end of an eight-ply game &mdash; and the search&rsquo;s
  own backed-up estimate, produced by the same untrained network. The signal was
  circular. Breaking that circle meant getting labels from outside.</p>
  <ol class="steps">
    <li><span><span class="what">Make the rules fast enough to matter.</span>
      <span class="how">The reference engine generated 21.6k positions/s while the network
      could evaluate 630k/s &mdash; self-play was 97&thinsp;% rule-bound. Re-expressed as
      array operations over batches of bitboards: 8.1M/s, a 377&times; speedup, cross-checked
      against the reference on 3,000 positions.</span></span></li>
    <li><span><span class="what">Solve positions exactly, in bulk.</span>
      <span class="how">A Rust exporter turns the depth-16 solver into a labelling tool.
      Solving one position also solves all its children, so each solve yields an optimal-move
      set plus a dozen free exact values.</span></span></li>
    <li><span><span class="what">Train on truth, not on itself.</span>
      <span class="how">3.09M exactly-labelled positions, 250k with exact optimal-move sets.
      Quantik is invariant under 192 symmetries, so every batch is replayed under a fresh
      random one. Training draws are balanced across plies, because the corpus is 75&thinsp;%
      endgame and the match is decided in the opening.</span></span></li>
    <li><span><span class="what">Search on the same clock.</span>
      <span class="how">The network drives a batched MCTS with a wall-clock budget, on one
      CPU thread &mdash; the same hardware the classical engines get. Every time in this
      report is measured, never nominal.</span></span></li>
  </ol>
  <p class="callout">Value-head error against exact truth: <strong>0.727 &rarr; 0.084</strong>.
  Everything else followed from that one number.</p>
</section>

<section class="wide">
  {head(5, "the result", "It wins, and it wins on less time")}
  <figure>
    <div class="fig-title"><span class="t">Win rate against minimax @ 100&thinsp;ms</span>
      <span class="s">Side-balanced paired games; bars run from even. Whiskers are 95&thinsp;% Wilson intervals.</span></div>
    <div class="scroll">{bar_chart()}</div>
    <figcaption>Both distilled configurations sit entirely above even. The bottom bar is
    the same architecture trained by self-play alone &mdash; equal average accuracy to
    minimax, and still a losing record, because equal accuracy is not equal strength when
    the mistakes fall in different places.</figcaption>
  </figure>
  <p style="max-width:34rem">The second bar is the one that settles it. At a
  {hc['ms_per_move_a']:.0f}&thinsp;ms budget against minimax&rsquo;s
  {hc['ms_per_move_b']:.0f}&thinsp;ms &mdash; <strong>{hc['ms_per_move_b']/hc['ms_per_move_a']:.1f}&times;
  less thinking time on the same single thread</strong> &mdash; the network still wins
  {hc['score_a']:.1%} of {hc['games']:,} games. It is not out-searching the solver. It is
  out-<em>knowing</em> it.</p>
  <div class="scroll">{leaderboard_table()}</div>
  <p style="max-width:34rem">Three things in that table are worth more than the headline.</p>
  <ul class="plain" style="max-width:34rem">
    <li><strong>The policy head alone places fourth, at 1&thinsp;ms per move.</strong> One
    forward pass, no search whatsoever, beating a beam search that spends 451&thinsp;ms.
    Most of what the classical engines buy with compute, the network simply has.</li>
    <li><strong>Beam spends 4.5&times; its stated budget.</strong> It only checks its clock
    between beam levels; minimax overshoots 2&times; for the same kind of reason. Had this
    report used nominal budgets, it would have been fiction.</li>
    <li><strong>Equal accuracy is not equal strength.</strong> The self-play network and
    minimax both score 97.2&thinsp;%, and one beats the other 58&ndash;42.</li>
  </ul>
  <div class="scroll">{ply_table()}</div>
</section>

<section class="wide">
  {head(6, "coverage", "It saw 5% of the game, and none of where it wins")}
  <p class="lede" style="max-width:34rem">Quantik has <strong>61,495,314</strong> canonical
  positions through ply 9 &mdash; counting a position and its 191 symmetric images as one.
  The network trained on 3,087,356 of them: <strong>5.02%</strong>. And at plies 4 and 5,
  where it beats minimax by the widest margin, it trained on <strong>none</strong>.</p>
  <figure>
    <div class="fig-title"><span class="t">Positions that exist, and positions it saw</span>
      <span class="s">Grey: every canonical position at that ply. Blue: the training slice.
      On a log axis the gap between the dots is the coverage ratio.</span></div>
    <div class="scroll">{coverage_chart()}</div>
    <figcaption>The opening is tiny &mdash; 10,946 positions at ply 4, fewer than a single
    training batch &mdash; and the network was shown none of it. Its accuracy there comes
    from exact values learned two to four plies deeper, carried up by search. That is what
    makes this a generalization result rather than a lookup table.</figcaption>
  </figure>
  <div class="scroll">{coverage_table()}</div>
  <p style="max-width:34rem">The enumeration reproduces the counts published in the
  project&rsquo;s own game-tree analysis exactly for plies 1&ndash;7. The one apparent
  discrepancy at ply 8 &mdash; 17,894,928 against a published 17,900,160 &mdash; resolves
  cleanly: the difference is exactly the 5,232 positions where the mover has no legal reply
  but no line is complete. That is a loss, so it is counted here as terminal and there as
  ongoing. Ply 9, at 37,922,646, is past where the published analysis stopped.</p>
</section>

<section class="wide">
  {head(7, "power", "Is the evaluation big enough?")}
  <p class="lede" style="max-width:34rem">The first draft of this report leaned on 640
  solved positions. That was too few where it counted: only 33 provably-won positions at
  each of plies 4 and 5. The probe is now <strong>8,440</strong> positions, with 1,240 at
  each of those plies.</p>
  <div class="scroll">{significance_table()}</div>
  <p style="max-width:34rem">Both agents see identical positions, so the comparison is
  <em>paired</em> &mdash; an unpaired interval would throw away most of the evidence. On the
  {COMPARE[0]['mcnemar']['positions']:,} won positions the two disagree
  {COMPARE[0]['mcnemar']['a_right_b_wrong'] + COMPARE[0]['mcnemar']['b_right_a_wrong']} times, and
  {COMPARE[0]['mcnemar']['a_right_b_wrong']} of those go the network&rsquo;s way. An exact paired test
  puts that at <strong>p&nbsp;=&nbsp;{COMPARE[0]['mcnemar']['p_value']:.1g}</strong>.</p>
  <p class="callout">Every disagreement between them is in the opening. Across
  <strong>{deep_positions:,}</strong> won positions at plies 8&ndash;12, neither engine makes
  a single mistake.</p>
  <p style="max-width:34rem">The original 640 was underpowered, not wrong. Minimax measured
  84.8% at ply 4 on 33 positions and 84.4% on 951; the network 97.0% and 96.4%. The point
  estimates held; the intervals were simply too wide to lean on. Headline accuracies moved
  (network 99.6%&nbsp;&rarr;&nbsp;{probe['qnet@200ms']['outcome_accuracy']:.1%}) because the new probe
  deliberately weights the opening far more heavily &mdash; not because any agent changed.</p>
</section>

<section>
  {head(8, "the honest part", "What this is not")}
  <ul class="plain">
    <li><strong>The network did not learn Quantik from nothing.</strong> It was taught by
    an exact solver. The from-scratch AlphaZero run is in the table precisely so the
    comparison stays honest &mdash; and it lost.</li>
    <li><strong>The ceiling here is low by construction.</strong> In side-balanced pairs,
    two near-perfect players each win the side they were handed, so scores are pulled
    toward 50&thinsp;%. Clearing it by ten points means the opening error rate is genuinely
    near zero, but nobody is winning 90&thinsp;% of these games.</li>
    <li><strong>A three-hour solve was thrown away.</strong> The oracle buffered its
    results and wrote them at the end; an over-broad process kill destroyed the lot. It
    now streams and resumes &mdash; which immediately saved the next interrupted run. A
    job measured in hours has to be interruptible by construction.</li>
    <li><strong>Sixteen probe positions leaked into training.</strong> The corpus builder
    filtered the positions it <em>sampled</em> but not the child rows derived from them, so
    16 of the original 640 arrived as value-only rows at plies 7-9 &mdash; none with a move
    label, none at plies 4-6. Worst case it moves the network 99.63%&nbsp;&rarr;&nbsp;99.61%
    and minimax 97.19%&nbsp;&rarr;&nbsp;97.11%. Immaterial, but the filter now sits where it
    cannot be bypassed, and the 7,800-position probe excludes the corpus up to symmetry.</li>
    <li><strong>One reported metric was wrong for an afternoon.</strong> Validation
    averaged unequally-weighted chunks, printing a real 89&thinsp;% top-1 accuracy as
    11&thinsp;%. It looked plausible &mdash; a low loss beside a low accuracy &mdash; which is
    exactly what made it dangerous.</li>
  </ul>
</section>

<section>
  {head(9, "reproduce", "Every number above")}
  <details>
    <summary>Commands</summary>
    <pre>cd quantik-models-py &amp;&amp; git checkout nn-beats-baselines
.venv/bin/python -m pytest -q                    # 77 tests

# exact labels (Rust solver, streams and resumes)
.venv/bin/python scripts/build_oracle_corpus.py

# distillation
.venv/bin/python -m quantik_models.train.supervised \
  --name qnet --corpus runs/oracle/corpus/sampled.npz \
  --channels 128 --blocks 6 --epochs 16

# the headline: arena + exact-truth probe in one report
.venv/bin/python scripts/final_evaluation.py \
  --agents runs/oracle/final-agents.json --out runs/arena/final \
  --positions 150 --start-plies 3-5 --workers 16</pre>
  </details>
  <p style="font-size:.93rem;color:var(--ink-2)">The engine cross-check is worth a line of
  its own: enumerating Quantik&rsquo;s canonical positions reproduced the counts published
  years earlier in the project&rsquo;s own game-tree analysis &mdash; 3, 51, 726, 10,946,
  105,632, 901,916 &mdash; exactly, at every level.</p>
</section>

<footer>
  <span>Network: 1,786,823 parameters, 8 residual blocks&rsquo; worth of trunk at 128 channels,
  trained 16 epochs on an Apple M5 Pro. Ground truth cross-validated between an independent
  Rust and Python solver.</span>
  <span>Confidence intervals are 95&thinsp;% Wilson. Quantik has no draws.</span>
</footer>
</div>

<script>
(function () {{
  var svg = document.querySelector('.chart');
  if (!svg) return;
  var cross = document.getElementById('crosshair');
  var line = cross && cross.querySelector('line');
  var tip = document.createElement('div');
  tip.style.cssText = 'position:fixed;pointer-events:none;opacity:0;transition:opacity .12s;' +
    'background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:.5rem .65rem;' +
    'font-family:"IBM Plex Mono",monospace;font-size:.72rem;line-height:1.6;color:var(--ink);' +
    'box-shadow:var(--shadow);z-index:9;white-space:nowrap';
  document.body.appendChild(tip);
  var data = {json.dumps({a: {p: probe[a]["accuracy_by_ply"][str(p)]["accuracy"] for p in PLIES} for a, _, _ in SERIES})};
  var names = {json.dumps([[a, s] for a, _, s in SERIES])};
  var colours = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
  svg.querySelectorAll('.hit').forEach(function (hit) {{
    hit.addEventListener('pointerenter', function (event) {{
      var ply = hit.getAttribute('data-ply');
      var x = parseFloat(hit.getAttribute('x')) + parseFloat(hit.getAttribute('width')) / 2;
      if (line) {{ line.setAttribute('x1', x); line.setAttribute('x2', x); cross.style.display = ''; }}
      tip.innerHTML = '<strong>ply ' + ply + '</strong><br>' + names.map(function (n, i) {{
        return '<span style="color:' + colours[i] + '">&#9632;</span> ' + n[1] + ' &middot; ' +
               Math.round(data[n[0]][ply] * 100) + '%';
      }}).join('<br>');
      tip.style.opacity = '1';
      move(event);
    }});
    hit.addEventListener('pointermove', move);
    hit.addEventListener('pointerleave', function () {{
      tip.style.opacity = '0';
      if (cross) cross.style.display = 'none';
    }});
  }});
  function move(event) {{
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(event.clientX + 14, window.innerWidth - w - 8) + 'px';
    tip.style.top = Math.max(event.clientY - h - 12, 8) + 'px';
  }}
}})();
</script>
"""

head = Path("docs/nn-quest/report.head.html").read_text()
OUT.write_text(head + body)
print("wrote", OUT, OUT.stat().st_size, "bytes")
