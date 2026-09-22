#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    import os

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda
    from langchain_deepseek import ChatDeepSeek

    system_prompt = (
        "You are a meticulous assistant that extracts payment figures from Hong Kong "
        "supermarket receipt photos.\n"
        "OUTPUT FORMAT: reply with STRICT JSON only: "
        '{"final_payment": <number>, "subtotal": <number>, "discounts": [<number>, ...]} '
        "with every amount in HKD using exactly 2 decimal places. No markdown fences, "
        "no explanation, no extra keys.\n"
        "FIELD DEFINITIONS:\n"
        '1. "subtotal" = the SUBTOTAL / 小計 line of the receipt (after all discounts '
        "have been applied, BEFORE the ROUNDING line).\n"
        '2. "final_payment" = the amount actually paid by the customer: the payment line '
        "printed immediately AFTER the ROUNDING line (e.g. OCTOPUS / 八達通扣賬金額 / "
        "VISA / CASH / 扣除金額). If the receipt has no ROUNDING line, take the first "
        "payment line after SUBTOTAL. Ignore duplicate payment-network summary lines "
        "further down (e.g. GP.VISA), CHANGE / 找續, card balance / 餘額 / remaining "
        "value lines, and points earned.\n"
        '3. "discounts" = a LIST of every negative amount (-$X.XX) printed BEFORE '
        "the SUBTOTAL line, each added back as a POSITIVE number. List every single "
        "discount line separately, in order. This includes: promotions (e.g. "
        "Buy 2 Save $x, Buy 3 Save $x), percentage discounts (e.g. 5% OFF, "
        "MB $200get 5%off), coupon lines, member/app upgrades (e.g. App upgrade, "
        "App Upgrades#390-$20_C), and packaging-damage rebates (包裝變形). Discount "
        "lines can be embedded between item lines and may use odd or truncated labels; "
        "list EVERY line whose amount is printed as negative before SUBTOTAL, even if "
        "the label is hard to read. Do NOT include ROUNDING (printed after SUBTOTAL; "
        "it is a rounding adjustment, not a discount). A coupon printed as $0.00 is not "
        "a discount and should be omitted.\n"
        "CRITICAL — AMOUNT COLUMN IS AUTHORITATIVE: every discount line has a "
        "promotional LABEL (e.g. 'Buy 2 Save $5') and a right-hand AMOUNT COLUMN "
        "(e.g. -$6.00). The label can DISAGREE with the amount actually deducted "
        "(labels may show a per-unit or stale figure). ALWAYS use the value printed "
        "in the right-hand amount column of the line, never the number inside the "
        "label text.\n"
        "QUALITY: read every number digit-by-digit, especially the ones in the middle "
        "of a line that may be blurry, tilted, faded, or partly covered by a finger. "
        "Never invent numbers; copy each amount exactly as printed."
    )

    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0,
        timeout=120,
        max_retries=2,
    )

    human_text = (
        "Read this receipt and reply with ONLY the strict JSON object "
        '{"final_payment": ..., "subtotal": ..., "discounts": [...]}.'
    )

    def _to_messages(inputs: dict) -> list:
        human_content = [
            {"type": "image_url", "image_url": {"url": inputs["image_url"]}},
            {"type": "text", "text": human_text},
        ]
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
        note = inputs.get("note")
        if note:
            messages.append(HumanMessage(content=note))
        return messages

    chain = RunnableLambda(_to_messages) | llm | StrOutputParser()
    return chain


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    import json
    import re
    from decimal import Decimal, InvalidOperation

    RETRY_NOTE = (
        "Your previous reply could not be parsed or was inconsistent. Look at the "
        "receipt image again very carefully and reply with ONLY the strict JSON object "
        '{"final_payment": ..., "subtotal": ..., "discounts": [...]} '
        "(2 decimal places each). Remember: subtotal = the 小計/SUBTOTAL line; "
        "final_payment = the payment line immediately after ROUNDING (or the first "
        "payment line after SUBTOTAL if there is no ROUNDING); discounts = every "
        "negative line printed BEFORE the SUBTOTAL line as a separate positive number "
        "in the list, excluding ROUNDING. IMPORTANT: for each discount line, use the "
        "right-hand amount column value (e.g. -$6.00), NOT the number in the "
        "promotional label (e.g. 'Buy 2 Save $5') — they can disagree."
    )

    def _extract_json(text: str):
        """Parse the amount fields out of a model reply; None if unusable."""
        if not isinstance(text, str):
            text = str(text)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        else:
            brace = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            candidate = brace.group(0) if brace else cleaned
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None

        def _dec(value):
            raw = str(value).replace(",", "").replace("$", "").strip()
            try:
                return Decimal(raw)
            except (InvalidOperation, ValueError):
                return None

        parsed = {}
        for key in ("final_payment", "subtotal"):
            if key not in data:
                return None
            value = _dec(data[key])
            if value is None:
                return None
            parsed[key] = value
        if isinstance(data.get("discounts"), list):
            total = Decimal("0")
            for item in data["discounts"]:
                value = _dec(item)
                if value is None:
                    return None
                total += -value if value < 0 else value
            parsed["discount_total"] = total
        elif "discount_total" in data:  # tolerate the flat-sum variant
            value = _dec(data["discount_total"])
            if value is None:
                return None
            parsed["discount_total"] = abs(value)
        else:
            return None
        return parsed

    def _plausible(data) -> bool:
        """Basic sanity checks; wildly wrong fields trigger one soft retry."""
        for value in data.values():
            if value != value or abs(value) > Decimal("1000000"):  # NaN / absurd
                return False
        if data["subtotal"] <= 0 or data["discount_total"] < 0:
            return False
        if data["final_payment"] <= 0:
            return False
        # paid amount should be within ~1 HKD of subtotal (rounding only)
        return abs(data["subtotal"] - data["final_payment"]) <= Decimal("1.00")

    def _fallback_from_text(text: str):
        """Last-resort: grab the last three plain money numbers in the reply."""
        if not isinstance(text, str):
            return None
        numbers = re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL))
        if len(numbers) < 3:
            return None
        try:
            return {
                "final_payment": Decimal(numbers[-3]),
                "subtotal": Decimal(numbers[-2]),
                "discount_total": Decimal(numbers[-1]),
            }
        except (InvalidOperation, ValueError):
            return None

    inputs = [{"image_url": image_data_url(path)} for path in images]

    def _run_batch(batch_inputs):
        try:
            return chain.batch(batch_inputs, config={"max_concurrency": 4})
        except Exception as exc:  # never crash: fall back to sequential invokes
            print(f"batch failed ({exc!r}); retrying sequentially")
            outputs = []
            for item in batch_inputs:
                try:
                    outputs.append(chain.invoke(item))
                except Exception as inner_exc:
                    print(f"invoke failed for one image: {inner_exc!r}")
                    outputs.append(None)
            return outputs

    def _majority(parsed):
        """Pick the value agreed by at least two votes, else the first plausible one."""
        valid = [d for d in parsed if d is not None]
        if not valid:
            return None
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                if valid[i] == valid[j]:
                    return valid[i]
        for d in valid:
            if _plausible(d):
                return d
        return valid[0]

    # pass 1: two independent votes per receipt to cancel out vision misreads
    double_inputs = []
    for item in inputs:
        double_inputs.extend([dict(item), dict(item)])
    double_outputs = _run_batch(double_inputs)
    per_receipt: list = []
    for idx, path in enumerate(images):
        votes = double_outputs[idx * 2:(idx + 1) * 2]
        parsed = [_extract_json(v) if v is not None else None for v in votes]
        chosen = _majority(parsed)
        per_receipt.append([path, votes, chosen])
        if chosen is not None:
            print(
                f"  {path.name}: paid={chosen['final_payment']}, "
                f"subtotal={chosen['subtotal']}, discounts={chosen['discount_total']}"
                f"{' [votes disagreed]' if len({str(v) for v in parsed if v is not None}) > 1 else ''}"
            )

    # pass 2: tie-breaker third vote for receipts whose two votes disagreed
    tie_pending = []
    for idx, entry in enumerate(per_receipt):
        votes, chosen = entry[1], entry[2]
        parsed = [_extract_json(v) if v is not None else None for v in votes]
        valid = [d for d in parsed if d is not None]
        agreed = any(
            valid[i] == valid[j]
            for i in range(len(valid))
            for j in range(i + 1, len(valid))
        )
        if chosen is not None and agreed:
            continue
        # no agreement (or too few usable votes): take a third vote
        tie_pending.append((idx, entry))
    if tie_pending:
        print(f"tie-breaking {len(tie_pending)} receipt(s) with a third vote")
        tie_outputs = _run_batch(
            [{"image_url": image_data_url(per_receipt[idx][0])} for idx, _ in tie_pending]
        )
        for (idx, entry), extra in zip(tie_pending, tie_outputs):
            entry[1].append(extra)
            entry[2] = _majority([_extract_json(v) if v is not None else None for v in entry[1]])

    # pass 3: corrective re-asks for receipts that are still unusable
    for _round in range(2):
        pending = []
        for entry in per_receipt:
            path, raw, data = entry
            if data is not None and _plausible(data):
                continue
            pending.append(entry)
        if not pending:
            break
        print(f"retrying {len(pending)} receipt(s), round {_round + 1}")
        for entry in pending:
            try:
                entry[1].append(
                    chain.invoke({"image_url": image_data_url(entry[0]), "note": RETRY_NOTE})
                )
                entry[2] = _majority([_extract_json(v) if v is not None else None for v in entry[1]])
            except Exception as exc:
                print(f"retry failed for {entry[0].name}: {exc!r}")

    # final fallback so the run always completes with a results.csv
    totals = {"final_payment": Decimal("0"), "subtotal": Decimal("0"), "discount_total": Decimal("0")}
    for path, raw_votes, data in per_receipt:
        if data is None:
            for raw in raw_votes:
                data = _fallback_from_text(raw)
                if data is not None and _plausible(data):
                    break
            else:
                data = None
        if data is None:
            print(f"WARNING: could not reliably extract {path.name}; counting it as 0")
            continue
        # parsed values are accepted even if implausible (better than zeroing)
        totals["final_payment"] += data["final_payment"]
        totals["subtotal"] += data["subtotal"]
        totals["discount_total"] += data["discount_total"]

    total_paid = totals["final_payment"].quantize(Decimal("0.01"))
    total_without_discount = (
        (totals["subtotal"] + totals["discount_total"]).quantize(Decimal("0.01"))
    )
    print(f"Per-question totals: paid={total_paid}, without_discount={total_without_discount}")

    return {
        QUERY_1: f"HK${total_paid}",
        QUERY_2: f"HK${total_without_discount}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
