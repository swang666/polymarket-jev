# Design notes

Why this is built the way it is, including the parts that were deliberately not
built. `README.md` covers how to run it.

---

## 1. The problem with the obvious version

The obvious way to point an AI model at a prediction market is to show it a market
and ask for a probability, then trade the gap against the price. With a model that
returns calibrated probabilities, this is very tempting.

It does not work, and the failure is not subtle once you name it.

A Noul answer is **P(the answer to your question is yes | the text you supplied)**.
TypeSafe's calibration claim is that those probabilities match observed label
frequencies across prediction groups on judgment tasks. It is not a claim that the
model's probability for a future world event matches that event's frequency.

Ask "will the ceasefire hold through March?" with no state and you get a number
built from how the sentence reads, not from anything about March. It arrives typed,
bounded and confident-looking — which makes it worse than a visibly bad answer,
because the pipeline downstream cannot tell the difference. Worse, on any liquid
market the price is an aggregate of people with money at stake; a text model with
no private information has no reason to beat it.

So the question became: **is there a question about a prediction market that is a
text-judgment question rather than a forecasting question?**

There is one, and it is the whole project.

## 2. Resolution lag

Polymarket markets resolve on written rules. The rules are prose, often pedantic,
and they name specific triggers, actors, sources and deadlines:

> This market will resolve to "Yes" if the Digital Asset Market Clarity Act of 2025
> (H.R.3633) is passed by both chambers of the U.S. Congress and signed into law by
> December 31, 2026, 11:59 PM ET.

Between the moment a qualifying fact is published and the moment the order book
reflects it, there is a window. Deciding whether a given article satisfies that
paragraph is exactly a reading-comprehension task over two pieces of text — the
thing Jev is built for, and a thing that scales badly for a human watching hundreds
of markets.

That reframing gives the project a claim it can actually defend:

> **This tool does not predict events. It notices when a published fact has already
> settled a market's rules and the price has not moved.**

Everything downstream follows from it. `condition_met` is phrased in the past
tense with an explicit "Do not consider whether it is likely to happen later."
`decisive_threshold` exists so that markets where no article settles anything —
the overwhelming majority — return `NO_VIEW` instead of a number. A scanner that
finds an opportunity in every market is broken, not lucky.

## 3. Reading the model's weaknesses as a spec

TypeSafe publishes a jaggedness page for Jev 1.13. Three of its entries map
directly onto things a trading tool naively does:

| Documented weakness | What it rules out | What this project does instead |
|---|---|---|
| *"Jev is not a calculator"*; unreliable counting | Asking the model about prices, spreads, expected value | All arithmetic in `scoring.py`, unit-tested against hand-worked values |
| Dates are treated as text, not ordered quantities | Asking "is this before the deadline?" | `days_to_resolution` and staleness computed in code |
| Context rot: accuracy falls with large irrelevant context | Stuffing twelve articles into one state | One article per deep read; rules truncated to 4k chars |
| Reads instructions literally, misses implied conditions | Vague instructions and abstract rubrics | Literal reading is the *feature* here — the rules are meant to be read literally. Score levels describe concrete situations, never "moderately severe" |
| `P(noul) != 1 - P(not noul)` | Deriving one direction from the other | Both `condition_met` and `condition_precluded` asked; both high means AVOID |

That last row is the one worth dwelling on. The natural instinct on seeing
`condition_met = 0.85` and `condition_precluded = 0.60` is to normalise them. But
they are independent judgments over the same text, and their disagreement is
information: it means the rules do not cleanly apply to this evidence. Averaging
would destroy exactly the signal that should stop the trade.

## 4. The cascade

Two stages, for two different reasons.

**Stage 1 — screen.** A keyword news search returns mostly noise; the live dry-run
for "Will the Iranian regime fall before 2027?" returned LNG demand and a
Switzerland sanctions guide. One request per market carries one Noul per headline.
The questions are independent, so they batch — the docs measure ~12x cheaper and
~10x faster than asking separately, because the rules text is sent once instead of
twelve times.

**Stage 2 — deep read.** Survivors get the six-question set, one article at a time.
Splitting per-article is more requests, not fewer, and that is the point: it buys
per-article attribution for the log, and it keeps each state small enough to dodge
context rot. At $42 per billion input tokens the extra requests are free in any
sense that matters.

Choosing between multiple qualifying articles uses *decisiveness* —
`max(condition_met, condition_precluded)` — rather than the screen score. An
article that settles the question in either direction is worth more than three
that hedge.

## 5. Judgments to money

Four decisions, all in code, all configurable, all tested.

**Shrink toward 0.5, not toward 0.** A decisive reading with weak sourcing should
move toward "no information", not toward the opposite conclusion:
`p = 0.5 + (decisive - 0.5) * (1 - haircut)`.

**Low confidence pushes a rating to its worst case.** Confidence measures how
concentrated a distribution is, not whether the answer is right. So it is never
treated as evidence of quality — only as a reason to stop leaning on the number.
At confidence 0, a Score input is treated as fully bad.

**The NO side pays the spread on the other side.** Buying NO costs `1 - bid`, not
`1 - mid`. Using the midpoint would manufacture roughly half a spread of edge that
does not exist — on a 2-cent spread against a 5-cent minimum edge, that alone
would have turned refusals into trades.

**Kelly, quartered and capped.** For a contract paying 1 at cost q with win
probability p, the Kelly fraction reduces to `(p - q) / (1 - q)`. Quarter Kelly,
then a 5% per-trade cap. Full Kelly on a model whose calibration in this domain is
unverified is not a considered risk, it is a guess.

**Staleness kills a trade.** The strategy's entire premise is that the price has
not caught up. A 200-hour-old headline has had time to reach the order book, so the
premise is gone and `TRADE` degrades to `WATCH`. This is why prices are refreshed
from CLOB rather than trusting Gamma's cached book.

## 6. What was deliberately left out

**Order placement.** Not a scope decision — a correctness one. There is no evidence
yet that the Brier gap is negative, so there is nothing to automate. The repository
contains no wallet and no signing code, which makes that non-negotiable rather than
a matter of remembering not to.

**A composite "confidence score" across all six judgments.** Tempting, and it would
collapse the gate logic into one number. But "any serious violation" rules do not
compose into weighted sums: an ambiguous resolution clause should veto a trade, not
be outvoted by three good scores. So ambiguity and contradiction are separate hard
gates, and only the three quality dimensions are weighted.

**Multi-outcome and neg-risk markets.** Every Gamma row is already binary Yes/No,
so the scanner treats each independently and ignores the neg-risk relationship
between sibling markets. That leaves a real arbitrage on the table — sibling
probabilities summing past 1 — but it is a pricing strategy, not a text-judgment
one, and mixing the two would blur what the backtest is measuring.

**Asking Jev to summarise or explain.** It does not generate text. The reasons in
the report are assembled in code from typed answers, which is why they are always
traceable to a number.

## 7. How this gets falsified

The honest failure modes, in rough order of likelihood:

1. **No lag exists.** Liquid markets may reprice faster than a news-RSS poll can
   observe. Shows up as: almost no `TRADE` gates, or `TRADE` gates whose edge has
   vanished by the time you look.
2. **Headlines are too thin.** Without body text the model over-reads a confident
   headline. Shows up as: good Brier on `NO_VIEW`/`WATCH` markets, bad Brier on
   `TRADE` ones. Fix is a body-text evidence provider, not a threshold tweak.
3. **The remaining edges are the ones nobody wants.** If a fact is public and the
   price has not moved, sometimes the reason is that resolution is genuinely
   disputed — precisely what `rule_ambiguity` is there to catch. Shows up as: high
   hit rate on `AVOID`-adjacent markets you overrode.
4. **The thresholds are fitted to nothing.** `decisive_threshold = 0.80` is a
   starting guess, not a result. It should move once there are resolved markets to
   move it with.

`--backtest` is built to surface 1 and 2 directly. The number that matters is the
Brier gap against the market price on the same markets — not the model's absolute
Brier, and not paper P&L, both of which can look fine while adding nothing to what
the price already knew.
