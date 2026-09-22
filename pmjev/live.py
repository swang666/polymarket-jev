"""Live execution adapter -- deliberately not wired up.

This file exists so the remaining work is visible and small, not so it can be
switched on today. It refuses to place an order, and the refusal is the point.

WHY IT IS OFF
-------------
As of the last scan there were zero resolved outcomes in the backtest, so the
Brier gap -- the only number that says whether any of this beats the price -- is
undefined. Forecast mode's signals failed an elementary coherence check that the
order book passed: two mutually exclusive, exhaustive markets summed to 0.69 on
the model against 1.00 on the book. Resolution-lag mode has found nothing at all
across three scans.

Automating execution against that is not a trading system. It is a scheduled
task that loses money in a reproducible way.

WHAT TURNING IT ON ACTUALLY REQUIRES
------------------------------------
1. `pip install py-clob-client` (0.34.6 at time of writing).
2. A funded Polygon wallet (chain id 137) holding USDC.e, and the private key
   available to an unattended process. That is the real cost of automation: a
   key that can move your funds sits on disk where a scheduled task can reach
   it. It is a different risk class from the TypeSafe API key entirely -- that
   one can only spend a few cents of inference.
3. CLOB API credentials derived from the key (api key, secret, passphrase),
   used for L2 HMAC request signing.
4. The right `signature_type` for how you hold funds: 0 for an EOA, 1 for a
   Polymarket proxy wallet, 2 for a Safe. Getting this wrong means orders that
   sign fine and never fill.
5. `funder` set to the address that actually holds the USDC, which for a proxy
   wallet is NOT the signer address.

The sketch, for when that day comes:

    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=PRIVATE_KEY,            # from the environment, never a config file
        chain_id=137,
        signature_type=1,           # proxy wallet
        funder=FUNDER_ADDRESS,
    )
    client.set_api_creds(client.create_or_derive_api_creds())

    signed = client.create_order(OrderArgs(
        token_id=market.yes_token_id,
        price=limit_price,          # a LIMIT, never a market order
        size=contracts,
        side="BUY",
    ))
    client.post_order(signed, OrderType.GTC)

Three things that sketch already gets right and any real version must keep:

* **Limit orders only.** A market order on a thin prediction-market book is how
  a 3-cent edge becomes a 12-cent loss on the fill alone.
* **The key comes from the environment**, never from config.yaml, which is
  gitignored only by convention and sits in a repo that is public.
* **The paper broker's portfolio limits apply unchanged.** Live execution
  changes where fills come from, nothing else.

THE GATE
--------
`--execute --live` stays refused until all of these hold:

* `--backtest` reports a Brier gap that is clearly negative,
* across at least ~30 resolved markets,
* in whichever mode you intend to trade, read from the mode comparison table,
* with the coherence guard quiet on that mode.

Those are not arbitrary. They are the minimum that distinguishes an edge from a
run of luck, and every one of them is already measured by code in this repo.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .broker import PaperBroker, Position
from .models import Signal


class LiveExecutionNotEnabled(RuntimeError):
    """Raised on any attempt to place a real order."""


class LiveBroker(PaperBroker):
    """Placeholder with the same interface as the paper broker.

    Subclassing PaperBroker is deliberate: when this is implemented, position
    keeping, settlement and every portfolio limit are inherited unchanged, and
    only the fill source differs.
    """

    venue = "live"

    def execute(self, signals: Sequence[Signal], run_id: str = "",
                now=None) -> Tuple[List[Position], List[Tuple[Signal, str]]]:
        raise LiveExecutionNotEnabled(
            "Live execution is not implemented, and switching it on now would "
            "be trading an unvalidated edge with an unattended private key.\n\n"
            "The backtest currently has no resolved outcomes, so there is no "
            "evidence either mode beats the price. Run 'python run.py "
            "--backtest' once the logged markets settle; if the Brier gap is "
            "clearly negative over ~30 resolutions, the adapter in pmjev/live.py "
            "is a small piece of work and the portfolio limits already exist.\n\n"
            "Until then use --execute on its own, which records the same "
            "positions on paper at real book prices."
        )
