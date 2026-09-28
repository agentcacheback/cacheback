"""Record a real three-hop state relay and measured latency for the browser demo."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from rcc import Delivery, SenderState, transfer_sync
from rcc.selectors import cacheback

DOCUMENTS = [
    {
        "name": "The booking",
        "text": (
            "FRIDAY COLLECTION CONFIRMATION\n\n"
            "Maya Chen has booked collection of her replacement building pass for Friday. "
            "Her collection reference is J7. The original confirmation lists Central Desk "
            "as the collection point. The pass gives access to her new workplace from Monday; "
            "it does not need activating at the office after collection. Maya will collect "
            "the pass herself. She wants to check the correct destination and accepted ID "
            "before leaving, rather than make a second trip.\n\n"
            "Bring the confirmation reference and a photo ID. A driving licence or passport "
            "is normally accepted at Central Desk. The name on the ID must match the booking. "
            "Payment was completed "
            "online, so no payment step is needed at the desk. A colleague cannot collect "
            "this pass on Maya's behalf. Her old pass stops working at the end of Sunday. "
            "If she misses Friday, the next collection day is Monday.\n\n"
            "The reception team also manages visitor badges and equipment returns. Visitor "
            "badges are issued on arrival; equipment returns use the service entrance. "
            "Neither queue handles replacement building passes. Printed confirmations are "
            "optional because the desk can look up a booking by reference.\n\n"
            "This confirmation was issued on Wednesday. Collection locations may change "
            "during building works. The Friday operations notice takes precedence over "
            "the location printed here. Check the notice using the collection reference, "
            "then consult the ID rules for the actual destination. A moved booking follows "
            "the receiving desk's rules, not the original desk's general advice.\n\n"
            "OTHER BOOKINGS IN THE SAME CONFIRMATION BATCH\n\n"
            "The reception team sent several kinds of appointment confirmation on Wednesday. "
            "Noor Patel holds collection reference J5 for a replacement building pass. "
            "Elias Grant holds collection reference J9 for the same service. Neither booking "
            "is linked to Maya's application. People travelling together must still check "
            "their own references, because being in the same confirmation batch does not "
            "mean their passes will be held at the same desk. A forwarded email from a "
            "colleague is not a change to the recipient's own collection arrangement.\n\n"
            "Visitor appointments use a different reference series. Lena Ortiz has a V3 "
            "appointment to meet the facilities team, and her host will meet her at visitor "
            "reception. Owen Brooks has an E2 equipment-return appointment for a borrowed "
            "monitor. Equipment-return receipts confirm that an item was received; they do "
            "not authorize collection of a building pass. The service printed at the top "
            "of each confirmation determines which instructions belong to that booking. "
            "A person attending two services should keep both confirmations available.\n\n"
            "CHECKING AND CHANGING A BOOKING\n\n"
            "The portal shows whether an application is approved, awaiting a photograph, "
            "or cancelled. An approved application is ready for the collection process. "
            "A photograph request means the pass has not yet been printed. Cancellation "
            "removes the appointment from the collection list and generates a separate "
            "email. Maya's application is approved and has not been cancelled. Her photo "
            "and spelling of her name have already been checked by the reception team. "
            "The portal's application status does not identify a desk after a temporary "
            "building closure; location changes are published in the operations notice.\n\n"
            "Booking holders can update a contact telephone number without creating a new "
            "application. A correction to the printed name must be reviewed by staff before "
            "collection. Rescheduling creates a replacement appointment and sends a new "
            "confirmation; simply opening the rescheduling page does not change anything. "
            "Automated reminder emails repeat the original booking details and may arrive "
            "after an operations notice. Keep the confirmation reference available when "
            "contacting reception so staff can find the correct application. Marketing "
            "messages about workplace events are unrelated to the collection service.\n\n"
            "If an email cannot be opened on arrival, reception can look up the "
            "application after confirming the holder's details. A screenshot of the "
            "booking is useful for finding the reference but does not replace the "
            "identity document. Please keep confirmation emails separate from calendar "
            "invitations sent by colleagues. Adding an appointment to a shared calendar "
            "does not authorize another person to collect the pass. The collection "
            "record remains attached to the named applicant."
        ),
    },
    {
        "name": "The change",
        "text": (
            "FRIDAY OPERATIONS NOTICE\n\n"
            "Building works have closed the public entrance at Central Desk for Friday. "
            "Existing collection bookings remain valid, but some have moved. This notice "
            "replaces locations in earlier confirmations. A desk name on a Wednesday "
            "confirmation is not a guarantee that the desk is operating on Friday.\n\n"
            "Collection references J4, J5 and J6 have moved to North Annex. Reference J7 "
            "has moved to Harbour Hub. References J8 and J9 have moved to Garden Desk. "
            "Do not go to Central Desk for these bookings. Staff at the old entrance can "
            "give directions but cannot hand out passes or reserve a later collection.\n\n"
            "Moved bookings keep their original collection reference. The replacement "
            "pass will be waiting at the assigned desk, so there is no need to submit "
            "another application or pay again. The destination is determined by the "
            "reference on the confirmation, not by the visitor's home address or the "
            "desk they used on a previous visit.\n\n"
            "The shuttle stop outside the workplace remains open. Use the destination "
            "on the front of the vehicle, because all three services use the same stop. "
            "Tickets are included with a collection booking. Bicycles cannot be carried, "
            "and luggage must fit below the seat. These transport arrangements do not "
            "change which desk holds a booking.\n\n"
            "This notice assigns destinations; it does not list accepted identity "
            "documents. Each receiving desk sets its own ID requirements. Consult the "
            "collection desk rules for the destination assigned to the booking reference.\n\n"
            "OTHER FRIDAY SERVICES\n\n"
            "References K1 and K2 are locker-key appointments and will be handled at "
            "North Annex. References K3 and K4 are locker-key appointments assigned to "
            "Garden Desk. These keys are stored separately from building passes. The "
            "letter in the reference is part of the identifier: a locker appointment "
            "does not inherit the arrangements for a pass booking with the same number. "
            "The locker team cannot hand out passes, even when both services share a "
            "reception area. Anyone attending both services needs both booking references.\n\n"
            "Equipment returns with E references use the loading entrance at the "
            "workplace. Couriers should report to the goods-in bell rather than the "
            "public reception queue. Visitors with V references should follow the "
            "instructions from their named host; their meeting may be in a different "
            "building. The Friday relocation of replacement pass collections does not "
            "move those meetings. The reception phone line can explain the service "
            "categories, but the operator cannot move a printed pass between desks "
            "during the collection day.\n\n"
            "ACCESS TO THE TEMPORARY DESKS\n\n"
            "North Annex visitors should use the courtyard entrance beside the bicycle "
            "stands. The main staircase is reserved for contractors, and an accessible "
            "lift is available from the courtyard lobby. Garden Desk is inside the "
            "library foyer; the garden gate is an exit and has no reception staff. "
            "Harbour Hub uses the signed reception entrance facing the bus stop. Its "
            "delivery entrance is for supplies only. These directions describe how to "
            "reach each desk and do not determine which desk holds an individual's pass.\n\n"
            "Shuttle drivers display the destination before passengers board. A change "
            "of vehicle does not change a passenger's booking or permit collection at "
            "another desk. Staff can arrange accessible transport when requested through "
            "reception. Personal vehicles must use public parking; loading bays remain "
            "reserved for deliveries. Keep pedestrian routes clear while contractors "
            "move equipment. The service team will publish a separate notice before "
            "normal arrangements resume. Archived notices for previous closures remain "
            "in the portal for reference, but their temporary routes do not apply to "
            "this Friday. A new notice will be dated and will state which service "
            "and collection day it replaces.\n\n"
            "Queue stewards will direct arrivals to the appropriate service counter. "
            "Lost property can be reported to any reception, but found items stay "
            "with the building where they were handed in. A report at another desk "
            "does not arrange a transfer. Emergency exits must remain clear, including "
            "when a queue extends into the foyer. Visitors needing a quiet waiting "
            "space should speak to a steward. These arrangements apply across all "
            "three temporary reception areas."
        ),
    },
    {
        "name": "The ID rule",
        "text": (
            "COLLECTION DESK RULES\n\n"
            "Replacement passes require an in-person identity check followed by handover. "
            "Bring the original identity document; photographs and photocopies are not "
            "accepted. The name must match the booking. A booking reference locates the "
            "pass but is not proof of identity. Online payment does not remove the "
            "identity check.\n\n"
            "Central Desk normally accepts a passport or a driving licence, but its "
            "Friday entrance is closed during building works. North Annex accepts a "
            "passport or a driving licence. Garden Desk also accepts either document. "
            "Use the rule for the desk where the collection actually takes place.\n\n"
            "Harbour Hub requires a passport for replacement building pass collections. "
            "A driving licence is not accepted at Harbour Hub. This passport requirement "
            "also applies to bookings moved from another desk, even if the original "
            "confirmation listed a driving licence as acceptable.\n\n"
            "A booking guarantees that the pass is ready, not that identity checks can "
            "be waived. Desk staff cannot accept a colleague's ID or leave a pass at "
            "the security gate. Anyone without the required ID must return with it "
            "or reschedule. Staff cannot transfer a booking to another desk on arrival.\n\n"
            "The public waiting area has drinking water and accessible seating. "
            "Security staff can give directions but cannot issue replacement passes. "
            "Lost property and visitor registration are separate services with their "
            "own procedures. Their ID policies do not apply to replacement building "
            "pass collections.\n\n"
            "RULES FOR OTHER SERVICES\n\n"
            "Locker-key collections require the original booking confirmation and the "
            "identity check specified by the locker team. A staff building pass can be "
            "used for that service because the holder already has active access. This "
            "does not make an old or replacement building pass an accepted identity "
            "document for collecting a new one. Each service keeps its own checklist. "
            "A receipt from an equipment return identifies the returned item and is "
            "not a substitute for a personal identity document at any collection desk.\n\n"
            "Visitor registration is arranged by the host. A visitor badge is valid "
            "only for the named visit and must be returned before leaving. Hosts may "
            "meet visitors in reception, but they cannot override the identity checks "
            "for replacement pass collections. Contractor access uses a separate "
            "approval list maintained by the building manager. A contractor's company "
            "letter confirms the purpose of the visit; it does not change the rules "
            "for a personal building pass. An appointment in one queue does not give "
            "priority in another queue.\n\n"
            "AT THE COLLECTION COUNTER\n\n"
            "Staff first locate the booking, compare the person and identity document, "
            "and check the printed name before handing over the pass. If the name is "
            "incorrect, the pass returns to the issuing team for correction. Staff "
            "will explain how to arrange a new collection once the correction is "
            "complete. A companion can help with communication, and an interpreter "
            "can be arranged through reception, but the booking holder must attend "
            "the check. Accessibility support changes how the appointment is handled, "
            "not which identity documents the destination accepts.\n\n"
            "After a successful check, staff provide a receipt and explain how to "
            "report a faulty pass. The receipt is proof of handover and should be "
            "kept until the holder has used the pass successfully. A damaged holder "
            "or lanyard can be replaced without issuing another pass. If the pass "
            "does not work at the workplace, contact the access team using the "
            "details on the receipt. Do not post a photograph of the pass or its "
            "access number in a public support forum. Returned old passes go into "
            "the secure disposal box at reception. Questions about printing, access "
            "permissions and lost cards are handled separately from the identity "
            "requirements for an appointment that has not yet been completed.\n\n"
            "Identity documents remain with their owners after the check. Staff do "
            "not retain a passport or driving licence as a deposit for the pass. "
            "Do not leave documents unattended on a public counter while joining "
            "another queue. If an item is left behind, contact the desk's reception "
            "team directly so it can be placed in secure storage. Collecting a "
            "forgotten item is a separate visit and does not complete an unfinished "
            "building pass appointment."
        ),
    },
]
QUESTIONS = [
    "Where should Maya collect her pass on Friday, and what ID should she bring?",
    "Can Maya collect her pass on Friday using only her driving licence?",
]
MAX_ANSWER_TOKENS = 1024
ANSWER_INSTRUCTION = (
    "Solve the question by combining the supplied evidence. Account for corrections and "
    "exceptions. Give a concise answer and the evidence that supports it. "
    "If a required fact is missing, say unknown. Do not invent facts."
)


def embed(model: Any, tokenizer: Any, text: str) -> torch.Tensor:
    """Embed prompt text using the same table as the sender and receiver."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return model.get_input_embeddings()(ids.to(model.device))[0]


def timestamp(model: Any) -> float:
    """Read wall time after any queued GPU work has finished."""
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    return perf_counter()


def decode(model: Any, tokenizer: Any, rows: torch.Tensor, limit: int) -> dict[str, Any]:
    """Greedily decode a bounded continuation, preserving exact transmitted token IDs."""
    output = model(inputs_embeds=rows[None], use_cache=True)
    ids: list[int] = []
    stop = "limit"
    for step in range(limit):
        token = output.logits[:, -1:].argmax(-1)
        if int(token.item()) == tokenizer.eos_token_id:
            stop = "eos"
            break
        ids.append(int(token.item()))
        if step + 1 < limit:
            output = model(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
    return {"text": tokenizer.decode(ids), "token_ids": ids, "stop": stop}


def respond(
    model: Any,
    tokenizer: Any,
    question: str,
    evidence: torch.Tensor,
    instruction: str,
    limit: int,
) -> dict[str, Any]:
    """Continue over actual input rows using the checkpoint's native chat framing."""
    marker = "<<HANDOFF>>"
    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Evidence:\n{marker}\nQuestion: {question}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before, after = chat.split(marker)
    rows = torch.cat((embed(model, tokenizer, before), evidence, embed(model, tokenizer, after)))
    return decode(model, tokenizer, rows, limit)


def relay(
    model: Any,
    tokenizer: Any,
    sources: list[dict[str, Any]],
    question: str,
    ratio: int,
    method: str,
    span_size: int,
) -> dict[str, Any]:
    """Pass each actual message to the next agent, which adds only its own new source."""
    weight = model.get_input_embeddings().weight
    incoming = weight.new_empty((0, weight.shape[1]))
    origins: list[list[int]] = []
    selected: list[int] = []

    def trace_selection(state: SenderState, ids: torch.Tensor, budget: int) -> list[int]:
        selected[:] = sorted(cacheback(state, ids, budget, span_size=span_size))
        return selected

    hops, cap = [], 0
    started = timestamp(model)
    for index, source in enumerate(sources):
        hop_start = timestamp(model)
        ids = torch.tensor(source["token_ids"], dtype=torch.long, device=model.device)
        rows = torch.cat((incoming, model.get_input_embeddings()(ids)))
        cap += (len(source["token_ids"]) + ratio - 1) // ratio
        if method == "text":
            message = respond(
                model,
                tokenizer,
                question,
                rows,
                "You are passing evidence to the next agent in a team. Combine the incoming "
                "notes with the new source. Preserve facts, numbers, exceptions and corrections "
                f"needed to answer the question. Your message is limited to {cap} tokens. "
                "Do not invent missing facts or assume you have seen the remaining sources.",
                cap,
            )
            ids = torch.tensor(message["token_ids"], dtype=torch.long, device=model.device)
            incoming = model.get_input_embeddings()(ids)
            hop = {"message": message}
        else:
            originals = origins + [[index, i] for i in range(len(source["token_ids"]))]
            output = model.model(inputs_embeds=rows[None], use_cache=True)
            state = SenderState(
                model,
                output.past_key_values,
                rows,
                tokenizer,
                inherited_positions=incoming.shape[0],
            )
            inbox: list[Delivery] = []
            transfer_sync(state, inbox.append, question, ratio=ratio, selector=trace_selection)
            incoming = inbox[0].messages[0].materialize(weight)
            assert torch.equal(incoming, rows[selected])
            assert state.past_key_values.get_seq_length() == rows.shape[0]
            origins = [originals[i] for i in selected]
            assert incoming.shape[0] == cap
            hop = {"origins": origins, "selected_indices": selected.copy()}
        assert incoming.shape[0] <= cap
        hops.append(
            {
                **hop,
                "positions": incoming.shape[0],
                "cap": cap,
                "seconds": timestamp(model) - hop_start,
            }
        )
        logging.info(
            "%s hop %s: %s/%s positions, %.3fs",
            method,
            index + 1,
            incoming.shape[0],
            cap,
            hops[-1]["seconds"],
        )
    answer_start = timestamp(model)
    answer = respond(
        model,
        tokenizer,
        question,
        incoming,
        ANSWER_INSTRUCTION,
        MAX_ANSWER_TOKENS,
    )
    answer_seconds = timestamp(model) - answer_start
    total_seconds = timestamp(model) - started
    logging.info("%s answer (%.3fs): %s", method, answer_seconds, answer["text"])
    return {
        "hops": hops,
        "answer": answer,
        "answer_seconds": answer_seconds,
        "seconds": total_seconds,
    }


@torch.inference_mode()
def record(
    model: Any,
    tokenizer: Any,
    documents: list[dict[str, str]],
    questions: list[str],
    ratio: int = 4,
    span_size: int = 4,
    output: Path | None = None,
) -> dict[str, Any]:
    """Check full context, then run both relays with alternating method order."""
    if type(ratio) is not int or ratio < 1:
        raise ValueError("ratio must be a positive integer")
    if type(span_size) is not int or span_size < 1:
        raise ValueError("span_size must be a positive integer")
    sources = []
    for document in documents:
        encoded = tokenizer(document["text"], add_special_tokens=False, return_offsets_mapping=True)
        sources.append(
            {**document, "token_ids": encoded["input_ids"], "offsets": encoded["offset_mapping"]}
        )
    decode(model, tokenizer, embed(model, tokenizer, "A short warmup."), 4)
    cases: list[dict[str, Any]] = []
    trace = {
        "schema": 2,
        "ratio": ratio,
        "span_size": span_size,
        "expected_cases": len(questions),
        "complete": False,
        "sources": sources,
        "cases": cases,
        "run": {
            "model": model.config._name_or_path or type(model).__name__,
            "revision": getattr(model.config, "_commit_hash", None),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "device": str(model.device),
            "dtype": str(model.dtype),
            "gpu": torch.cuda.get_device_name(model.device)
            if model.device.type == "cuda"
            else None,
            "platform": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cpu_threads": torch.get_num_threads(),
            "latent_steps": 0,
            "representation": "embeddings",
            "decoding": "greedy",
            "thinking": False,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
            "timing": "One warm run per method and question, sequential; model loading, "
            "tokenization and warmup excluded. CUDA is synchronized at timing boundaries. "
            "Hops include prefill and handoff; total also includes final receiver generation. "
            "These are local observations, not a benchmark or speedup claim.",
        },
    }
    for index, question in enumerate(questions):
        logging.info("Recording: %s", question)
        evidence = torch.cat([embed(model, tokenizer, source["text"]) for source in sources])
        baseline_start = timestamp(model)
        baseline = respond(
            model, tokenizer, question, evidence, ANSWER_INSTRUCTION, MAX_ANSWER_TOKENS
        )
        full_context = {
            "answer": baseline,
            "seconds": timestamp(model) - baseline_start,
            "positions": evidence.shape[0],
        }
        logging.info("full_context answer (%.3fs): %s", full_context["seconds"], baseline["text"])
        order = ("text", "rclc") if index % 2 else ("rclc", "text")
        results = {
            method: relay(model, tokenizer, sources, question, ratio, method, span_size)
            for method in order
        }
        cases.append(
            {"question": question, "full_context": full_context, "order": order, **results}
        )
        trace["complete"] = len(cases) == len(questions)
        if output is not None:
            save(trace, output)
    return trace


def save(trace: dict[str, Any], path: Path) -> None:
    """Write readable JSON with numeric arrays wrapped at a reviewable line width."""
    text = json.dumps(trace, ensure_ascii=True, indent=2)
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|\[[\d\s,\[\]]*\]',
        lambda m: (
            m[0]
            if m[0].startswith('"')
            else textwrap.fill(json.dumps(json.loads(m[0])), width=160, subsequent_indent="      ")
        ),
        text,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Record the demo on CPU or one CUDA GPU, with no serving dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--span-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trace-w4.json"))
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(4)
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but no CUDA GPU is available")
    if options.dtype == "auto":
        options.dtype = (
            "float32"
            if options.device == "cpu"
            else ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(options.model, revision=options.revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            options.model,
            revision=options.revision,
            dtype=getattr(torch, options.dtype),
            attn_implementation="sdpa",
        )
        .to(options.device)
        .eval()
    )
    record(model, tokenizer, DOCUMENTS, QUESTIONS, options.ratio, options.span_size, options.output)
    logging.info("Saved %s", options.output)


if __name__ == "__main__":
    main()
