# polymarket_jev

A Polymarket scanner built on **TypeSafe's Jev model**. It looks for markets where
a published fact has *already* satisfied the market's written resolution rules but
the price has not caught up yet.

It is alert-only. There is no wallet, no signing code and no order path anywhere in
this repository.

```
python run.py --demo        # offline walkthrough, no API key, no network
python run.py --dry-run     # live markets + news, but nothing sent to Jev
python run.py --scan        # the real thing (needs TYPESAFE_API_KEY)
python run.py --backtest    # score logged judgments against what actually happened
```

---

## The one thing to understand first

Jev is not a forecaster. It returns typed judgments about text you give it, with
calibrated probabilities. A Noul answer is **P(the answer to your question is yes,
given the text you supplied)** — not P(the event happens).

Ask it "will Bitcoin be above $100k in December?" and you get a semantic prior
dressed up as a forecast. That number is worthless for trading, and it will look
exactly as confident as a good one.

So this project never asks about the future. It asks:

> Does this article report that the thing `resolution_rules` requires has **already
> happened**?

That is a reading-comprehension question about two pieces of text, which is what
the model is actually good at. The edge, when there is one, is **resolution lag**:
the fact is public, the rules are plainly satisfied, and the order book has not
moved yet. Most of the time there is no such market, and the scanner says so —
`NO_VIEW` is the correct and most common answer.

The model's own documented weaknesses drove the rest of the design. Jev is *"not a
calculator"*, treats dates as text rather than ordered quantities, and loses
accuracy when state is padded with irrelevant context. So all arithmetic, every
price and deadline comparison, and all position sizing happen in `pmjev/scoring.py`,
and each deep read sees exactly one article.

---

## Install

```bash
cd polymarket_jev
pip install -r requirements.txt          # requests + PyYAML
cp config.example.yaml config.yaml       # optional; defaults work
```

Put your key in `.env` or the environment — never in `config.yaml`:

```
TYPESAFE_API_KEY=sk-...
```

Get one at <https://console.typesafe.ai/>. Polymarket's read APIs need no key.

---

## What a run does

```
discover  Gamma /markets -> filter (open, liquid, priced 0.05-0.95, resolves
          within 120 days, has rules text, not a sports or price-feed market)
prices    CLOB /prices -> overwrite Gamma's book with live bid/ask
evidence  Google News RSS per market -> up to 12 candidate headlines
screen    ONE Jev request per market: one Noul per headline. Junk drops out here.
judge     ONE Jev request per surviving headline: six independent judgments
score     pure code: haircut, shrink, effective cost, edge, Kelly, gate
report    ranked table + Markdown + append-only JSONL log
```

### The six judgments

Sent as one batched request per article. Batching independent questions is roughly
12x cheaper and 10x faster than asking them one at a time.

| id | type | what it asks |
|---|---|---|
| `condition_met` | Noul | Does the evidence report the YES condition as **already** satisfied, read literally? |
| `condition_precluded` | Noul | Does the evidence report something that makes YES impossible? |
| `stance` | Choice | supports_yes / supports_no / mixed / **not_relevant** |
| `directness` | Score 0–3 | passing mention → states the named event as accomplished fact |
| `rule_ambiguity` | Score 0–3 | no judgment call → two careful readers would plainly disagree |
| `source_authority` | Score 0–3 | anonymous → the primary source the rules name |

Both Nouls are asked because a Noul is answered independently: `P(met)` and
`P(precluded)` need not sum to 1. When both come back high, that is an incoherent
reading of the rules, and the scanner refuses the market rather than averaging.

### From judgments to a position

```
decisive = max(condition_met, condition_precluded)
  < 0.80                       -> NO_VIEW   (a forecasting question; no edge here)
  both nouls >= 0.35           -> AVOID     (contradictory reading)
  rule_ambiguity >= 0.75 norm  -> AVOID     (dispute risk dominates any edge)

haircut  = weighted(ambiguity, 1-authority, 1-directness) * 0.30
           ...with any Score the model was unsure of pushed toward its worst case
p_model  = 0.50 + (decisive - 0.50) * (1 - haircut)      # shrink toward no-information

cost     = ask + slippage            (YES)
         = (1 - bid) + slippage      (NO -- the spread is really paid on this side)
edge     = p_model - cost
stake    = clamp((p - cost)/(1 - cost) * 0.25, 0, 5% of bankroll)   # quarter Kelly

edge >= 0.05 and evidence fresher than 72h  -> TRADE
otherwise                                    -> WATCH
```

Every threshold is in `config.yaml`. Re-tune them from backtest results, not from
how the console output feels.

---

## Before you risk anything

`--backtest` replays the JSONL log against realised resolutions and prints:

- **Brier gap** = model Brier − market-price Brier on the same resolved markets.
  Negative means the model was closer to the truth than the price. Positive means
  the price already knew.
- Calibration buckets: predicted vs. observed frequency.
- Paper P&L and hit rate on the `TRADE`-gated signals only.

Typed output guarantees the interface, not the truth. Jev's calibration is measured
on judgment tasks — not on your markets, your news source, or your thresholds. Until
the Brier gap is clearly negative across a few dozen resolutions, this has not earned
real money. The scanner logs the exact state and questions with every judgment so a
bad call can be re-read rather than argued about from memory.

---

## Cost

Jev bills input tokens only, at $42 per billion. A 40-market scan is roughly 150k
input tokens — well under a cent. `budget_usd_per_run` (default $0.50) aborts the
scan rather than overspending, and `--dry-run` prices a scan before you commit.

---

## Layout

```
run.py                  CLI
config.example.yaml     every threshold that decides whether money moves
pmjev/
  questions.py          every word Jev is ever asked  <- the design surface
  scoring.py            all arithmetic, Kelly, gating (pure, no I/O)
  judge.py              two-stage screen -> deep read cascade
  polymarket.py         Gamma + CLOB read-only clients
  evidence.py           Google News / publisher RSS / file providers
  typesafe.py           System One HTTP client, retries, budget meter
  backtest.py           Brier, calibration, paper P&L
  store.py              append-only JSONL judgment log
  report.py             console + Markdown output
  fixtures/             recorded markets + scripted answers for --demo
tools/make_fixtures.py  regenerate the demo fixtures
tests/                  109 tests, no network
```

```bash
python -m pytest tests/ -q
```

---

## Working from another device

```bash
git clone https://github.com/swang666/polymarket-jev.git
cd polymarket-jev
pip install -r requirements.txt
cp .env.example .env          # then paste your TYPESAFE_API_KEY in
python run.py --demo          # verifies the install; needs no key and no network
python -m pytest tests/ -q    # 109 tests, no network
```

Python 3.9+. The only dependencies are `requests` and `PyYAML`.

**The judgment log travels with the repo.** `data/judgments.jsonl` is committed on
purpose: it is the accumulated record of whether this strategy works, and a
backtest run against half your history is worse than no backtest. `.gitattributes`
merges it with git's `union` driver, so two devices appending on the same day
combine instead of conflicting, and `backtest.py` keys on market id and keeps the
earliest view of each market, so duplicate rows are collapsed at read time.

The habit that keeps this honest:

```bash
git pull                      # before scanning
python run.py --scan
git add data/judgments.jsonl && git commit -m "scan $(date -u +%FT%TZ)" && git push
```

`.env` and `config.yaml` are gitignored, so your key never leaves the machine and
each device can carry its own thresholds. If you want your tuned thresholds to
follow you too, commit a copy under a different name (`config.mine.yaml`) and
point at it with `--config config.mine.yaml`.

---

## Known limits

- **Evidence is headline-only by default.** Google News RSS article links are
  JavaScript redirects, so body text cannot be fetched from it. For "X was sworn
  in" the headline *is* the evidence, but it is a real ceiling. Use
  `provider: rss` with publisher feeds (they carry summaries), `provider: file`
  to supply your own, or drop in a keyed news API behind `EvidenceProvider`.
- **The screen is only as good as the search query.** `Market.keywords()` is
  deliberately crude; the Noul screen is what separates signal from noise. Watch
  what `--dry-run` returns for your markets before trusting a scan.
- **Sports and crypto-price markets are excluded by default.** They resolve off a
  scoreboard or a price feed, so reading the news buys you nothing, and dates and
  arithmetic are Jev's documented weak spot.
- **Resolution-lag opportunities are rare and short-lived.** If the scanner
  frequently finds large edges on liquid markets, suspect your evidence source or
  your thresholds before believing it.
- **`--demo` bypasses the discovery filters** so it can show all four gate
  outcomes; one fixture market resolves in 2028 and a live scan would skip it.
