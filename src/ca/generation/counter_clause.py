
"""Deterministic counter-clause drafting for Contract Autopsy Agent.

Drafts are negotiation starting points, not legal advice.
Original risk findings and evidence are never modified.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.ca.schemas import RiskFinding


class CounterClauseDraft(BaseModel):
    """A proposed clause linked to its source risk finding."""

    finding_id: str
    category: str
    original_clause_id: str | None = None
    original_rating: str
    original_evidence_status: str
    original_evidence_quotes: list[str] = Field(default_factory=list)
    proposed_clause: str = Field(min_length=1)
    negotiation_points: list[str] = Field(default_factory=list)
    generated_by: str = "template"
    applicability_note: str
    disclaimer: str = (
        "Draft for negotiation only. Not legal advice or a legally validated "
        "clause. Have a qualified lawyer review it before use."
    )


_CATEGORY_DRAFTS: dict[str, str] = {
    "payment": (
        "The client shall pay all undisputed invoices within 30 calendar days "
        "of receipt. Any disputed amount shall be identified in writing with "
        "reasons, and the parties shall work in good faith to resolve the "
        "dispute. Undisputed amounts shall remain payable when due."
    ),
    "ip": (
        "Ownership and permitted use of deliverables shall be expressly agreed "
        "in writing. Any transfer of intellectual property rights shall take "
        "effect only on payment in full of the agreed fees. Each party shall "
        "retain ownership of its pre-existing materials, tools, and know-how."
    ),
    "termination": (
        "Either party may terminate this agreement by giving 30 calendar days' "
        "written notice. The client shall pay for work completed and approved "
        "expenses incurred up to the effective termination date. Any immediate "
        "termination rights shall be expressly specified in this agreement."
    ),
    "scope": (
        "The services, deliverables, milestones, and acceptance criteria shall "
        "be documented in writing. Any material change in scope, timeline, or "
        "fees shall require written agreement by both parties before the "
        "additional work begins."
    ),
    "liability": (
        "Each party's liability shall be subject to clearly stated limits and "
        "agreed exceptions. The parties shall expressly identify the types of "
        "loss covered, applicable caps, and any exclusions. The final limits "
        "shall be agreed in writing before this agreement is signed."
    ),
    "confidentiality": (
        "Each party shall protect the other party's confidential information "
        "using reasonable safeguards and use it only for purposes of this "
        "agreement. Confidentiality obligations shall not apply to information "
        "that is public through no breach, independently developed, or lawfully "
        "received without a duty of confidence. Any required disclosure by law "
        "shall be handled as permitted by applicable law."
    ),
}


_CATEGORY_NEGOTIATION_POINTS: dict[str, list[str]] = {
    "payment": [
        "Agree on a clear invoice due date.",
        "Specify how payment disputes must be raised.",
        "Confirm whether late-payment consequences should be included.",
    ],
    "ip": [
        "Identify pre-existing intellectual property.",
        "Clarify when ownership transfers and what the fees cover.",
        "Specify any continuing licence or portfolio-use rights.",
    ],
    "termination": [
        "Agree on a reasonable notice period.",
        "Specify payment for completed work and approved expenses.",
        "Clarify any immediate termination conditions.",
    ],
    "scope": [
        "List deliverables and acceptance criteria.",
        "Agree on a written change-request process.",
        "Clarify how additional work affects fees and deadlines.",
    ],
    "liability": [
        "Discuss a proportionate liability cap.",
        "Define covered losses and any exceptions.",
        "Have a lawyer review the proposed allocation of liability.",
    ],
    "confidentiality": [
        "Define confidential information clearly.",
        "Agree on reasonable exclusions and permitted disclosures.",
        "Specify the duration of confidentiality obligations.",
    ],
}


def _applicability_note(finding: RiskFinding) -> str:
    """Explain the limits of a template using existing finding metadata."""

    if finding.rating == "UNRATED":
        return (
            "This finding is UNRATED. The rule engine has not established a "
            "specific risk rating. This generic template is not evidence that "
            "the original clause is defective or that this wording is needed."
        )

    if finding.evidence_status in {"absent", "insufficient"}:
        return (
            "The finding reports absent or insufficient evidence. This template "
            "is a discussion starting point, not proof that the proposed wording "
            "addresses the contract's actual needs."
        )

    if finding.evidence_status in {"ambiguous", "conflicting"}:
        return (
            "The finding reports ambiguous or conflicting evidence. Review the "
            "full contract and resolve the uncertainty before deciding whether "
            "to use this template."
        )

    if finding.evidence_status == "present":
        return (
            "Evidence is attached to this finding. This generic category-level "
            "template has not been tailored to the original clause and may "
            "duplicate, change, or weaken protections already present. Compare "
            "both texts before considering any replacement."
        )

    return (
        "This is a generic category-level template. Check it against the full "
        "contract before considering whether it is appropriate."
    )


def generate_counter_clause(finding: RiskFinding) -> CounterClauseDraft:
    """Create a conservative template without changing the original finding.

    This function does not call an AI provider and does not claim that the
    proposed wording resolves the identified issue.
    """

    category = finding.category

    if category not in _CATEGORY_DRAFTS:
        raise ValueError(
            f"No counter-clause template is available for category: {category}"
        )

    evidence_quotes = [
        evidence.quote for evidence in finding.evidence if evidence.quote
    ]

    return CounterClauseDraft(
        finding_id=finding.finding_id,
        category=category,
        original_clause_id=finding.clause_id,
        original_rating=finding.rating,
        original_evidence_status=finding.evidence_status,
        original_evidence_quotes=evidence_quotes,
        proposed_clause=_CATEGORY_DRAFTS[category],
        negotiation_points=list(_CATEGORY_NEGOTIATION_POINTS[category]),
        generated_by="template",
        applicability_note=_applicability_note(finding),
    )
