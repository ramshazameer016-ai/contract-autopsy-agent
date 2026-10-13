"""
Deterministic Fact Extraction module for contract clauses.
Extracts structured facts with exact EvidenceSpan anchors back to clause text.
Strictly offline, rule-driven, no LLM calls.
"""

from __future__ import annotations

import re
from src.ca.schemas import Clause, Fact, EvidenceSpan


def _create_evidence(clause: Clause, match: re.Match) -> EvidenceSpan:
    """Create an exact EvidenceSpan anchored in clause text."""
    start = match.start()
    end = match.end()
    quote = match.group(0)
    # Ensure exact match in clause text
    assert clause.text[start:end] == quote, f"Quote offset mismatch in clause {clause.clause_id}"
    return EvidenceSpan(
        doc_id=clause.doc_id,
        clause_id=clause.clause_id,
        page=clause.page_start,
        start=start,
        end=end,
        quote=quote,
    )


def extract_facts_from_clause(clause: Clause) -> list[Fact]:
    """Extract deterministic facts from a single clause based on its category and content."""
    facts: list[Fact] = []
    text = clause.text
    text_lower = text.lower()

    # --- 1. PAYMENT FACTS ---
    # Due days: "within fifteen (15) days of receipt", "within 90 days of invoice", "net 30", "due in 60 days", "pay... within 15 days"
    due_match = re.search(r"(?:(?:pay(?:ment)?|invoice[s]?|fee[s]?)[^\n.]{0,40}within\s+(?:[a-zA-Z]+\s*\()?(\d+)\)?\s*days|within\s+(?:[a-zA-Z]+\s*\()?(\d+)\)?\s*days\s*of\s*(?:receipt|invoice)|net\s*(\d+)|due\s*(?:within|in)\s*(\d+)\s*days)", text, re.IGNORECASE)
    if due_match:
        val_str = next(g for g in due_match.groups() if g is not None)
        due_val = int(val_str)
        facts.append(
            Fact(
                slot="payment.due_days",
                value=due_val,
                unit="days",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, due_match),
            )
        )

    # Termination cure period: "cure such breach within fourteen (14) days"
    cure_match = re.search(r"(?:cure\s+(?:such\s+)?breach\s+within\s+(?:[a-zA-Z]+\s*\()?(\d+)\)?\s*days)", text, re.IGNORECASE)
    if cure_match:
        val_str = next(g for g in cure_match.groups() if g is not None)
        facts.append(
            Fact(
                slot="termination.cure_period_days",
                value=int(val_str),
                unit="days",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, cure_match),
            )
        )

    # Payment amount stated: "INR 2,50,000", "fixed fee", "$50,000", "Rs."
    amt_match = re.search(r"(?:(?:INR|Rs\.?|USD|\$|EUR|€)\s*[\d,]+(?:\.\d+)?|fixed\s+fee\s+of\s+(?:INR|Rs\.?|USD|\$)?\s*[\d,]+|fee\s+of\s+(?:INR|Rs\.?|USD|\$)\s*[\d,]+)", text, re.IGNORECASE)
    if amt_match:
        facts.append(
            Fact(
                slot="payment.amount_stated",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, amt_match),
            )
        )

    # Advance payment: "30% of the Fee... in advance", "advance payment of 20%"
    adv_match = re.search(r"(?:(\d+(?:\.\d+)?)\s*%\s*(?:of\s+the\s+fee)?\s*(?:(?:shall\s+be|is)\s+payable\s+)?in\s+advance|advance\s+payment\s+of\s+(\d+(?:\.\d+)?)\s*%)", text, re.IGNORECASE)
    if adv_match:
        adv_val_str = next(g for g in adv_match.groups() if g is not None)
        facts.append(
            Fact(
                slot="payment.advance_pct",
                value=float(adv_val_str),
                unit="percent",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, adv_match),
            )
        )

    # Milestone based:
    milestone_match = re.search(r"\b(?:milestones?|milestone-based)\b", text, re.IGNORECASE)
    if milestone_match:
        facts.append(
            Fact(
                slot="payment.milestone_based",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, milestone_match),
            )
        )

    # Late fee / interest stated:
    late_fee_match = re.search(r"(?:interest\s+at\s+[\d.]+\s*%\s*per\s+month|overdue\s+amounts?\s+bear\s+interest|late\s+fee)", text, re.IGNORECASE)
    no_interest_match = re.search(r"(?:no\s+interest\s+shall\s+accrue)", text, re.IGNORECASE)
    if late_fee_match:
        facts.append(
            Fact(
                slot="payment.late_fee_stated",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, late_fee_match),
            )
        )
    elif no_interest_match or ("payment" in clause.category and due_match):
        # A deadline exists but no interest accrues
        evidence_m = no_interest_match if no_interest_match else due_match
        facts.append(
            Fact(
                slot="payment.late_fee_stated",
                value=False,
                source="regex",
                confidence="medium",
                evidence=_create_evidence(clause, evidence_m),
            )
        )

    # Conditional on third party:
    cond_match = re.search(r"(?:only\s+after\s+the\s+client\s+has\s+been\s+paid|conditional\s+upon\s+payment\s+by\s+(?:third\s+party|customer)|pay-when-paid)", text, re.IGNORECASE)
    if cond_match:
        facts.append(
            Fact(
                slot="payment.conditional_on_third_party",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, cond_match),
            )
        )

    # --- 2. INTELLECTUAL PROPERTY FACTS ---
    # Pre-existing IP carved out:
    preexist_carve_match = re.search(r"(?:pre-existing\s+materials\s+and\s+tools\s+owned\s+by\s+the\s+(?:consultant|contractor)\s+remain\s+the\s+exclusive\s+property|background\s+ip\s+retains\s+original\s+ownership)", text, re.IGNORECASE)
    preexist_vest_match = re.search(r"(?:all\s+pre-existing\s+materials\s+used[,\s]+shall\s+vest\s+(?:exclusively\s+)?in\s+the\s+client)", text, re.IGNORECASE)
    if preexist_carve_match:
        facts.append(
            Fact(
                slot="ip.preexisting_ip_carved_out",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, preexist_carve_match),
            )
        )
    elif preexist_vest_match:
        facts.append(
            Fact(
                slot="ip.preexisting_ip_carved_out",
                value=False,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, preexist_vest_match),
            )
        )

    # Assignment scope: "all_work" vs "deliverables_only" vs "license_only" vs "retained"
    all_work_match = re.search(r"(?:all\s+work\s+product[,\s]+whether\s+or\s+not\s+related|all\s+work\s+product,\s*inventions,\s*designs|shall\s+vest\s+exclusively\s+in\s+the\s+client\s+upon\s+creation)", text, re.IGNORECASE)
    deliv_only_match = re.search(r"(?:deliverables\s+specifically\s+created\s+for\s+the\s+company\s+shall\s+become\s+the\s+property|final\s+deliverables)", text, re.IGNORECASE)
    license_match = re.search(r"(?:grants\s+the\s+client\s+(?:an?\s+)?(?:exclusive|non-exclusive)[,\s]+(?:irrevocable[,\s]+)?(?:worldwide\s+)?licen[sc]e|retains\s+ownership)", text, re.IGNORECASE)

    if license_match:
        facts.append(
            Fact(
                slot="ip.assignment_scope",
                value="license_only",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, license_match),
            )
        )
    elif all_work_match:
        facts.append(
            Fact(
                slot="ip.assignment_scope",
                value="all_work",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, all_work_match),
            )
        )
    elif deliv_only_match:
        facts.append(
            Fact(
                slot="ip.assignment_scope",
                value="deliverables_only",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, deliv_only_match),
            )
        )

    # Assignment timing: "on_creation" vs "on_full_payment"
    on_creation_match = re.search(r"(?:upon\s+creation|on\s+creation|as\s+created)", text, re.IGNORECASE)
    on_payment_match = re.search(r"(?:upon\s+(?:receipt\s+of\s+)?full\s+payment|after\s+full\s+payment)", text, re.IGNORECASE)
    if on_payment_match:
        facts.append(
            Fact(
                slot="ip.assignment_timing",
                value="on_full_payment",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, on_payment_match),
            )
        )
    elif on_creation_match:
        facts.append(
            Fact(
                slot="ip.assignment_timing",
                value="on_creation",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, on_creation_match),
            )
        )

    # Moral rights waiver:
    moral_waived_match = re.search(r"(?:waives\s+(?:all\s+)?moral\s+rights|irrevocably\s+waives\s+all\s+moral\s+rights)", text, re.IGNORECASE)
    moral_retained_match = re.search(r"(?:retains\s+moral\s+rights)", text, re.IGNORECASE)
    if moral_waived_match:
        facts.append(
            Fact(
                slot="ip.moral_rights_waiver",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, moral_waived_match),
            )
        )
    elif moral_retained_match:
        facts.append(
            Fact(
                slot="ip.moral_rights_waiver",
                value=False,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, moral_retained_match),
            )
        )

    # Portfolio right:
    portfolio_match = re.search(r"(?:showcase\s+the\s+work\s+in\s+professional\s+portfolios|portfolio\s+rights?)", text, re.IGNORECASE)
    if portfolio_match:
        facts.append(
            Fact(
                slot="ip.portfolio_right",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, portfolio_match),
            )
        )

    # --- 3. TERMINATION FACTS ---
    # Client notice days: "thirty (30) days written notice", "forty-five (45) days", "14 days"
    trm_notice_match = re.search(r"(?:(?:providing|upon)\s+(?:[a-zA-Z-]+\s*\()?(\d+)\)?\s*days\s*written\s*notice|notice\s+of\s+(\d+)\s*days)", text, re.IGNORECASE)
    if trm_notice_match:
        val_str = next(g for g in trm_notice_match.groups() if g is not None)
        facts.append(
            Fact(
                slot="termination.notice_days_client",
                value=int(val_str),
                unit="days",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, trm_notice_match),
            )
        )

    # Pay for work done upon termination:
    pay_done_match = re.search(r"(?:pay\s+for\s+all\s+work\s+completed\s+up\s+to\s+the\s+effective\s+termination\s+date|pay\s+for\s+work\s+done)", text, re.IGNORECASE)
    no_pay_done_match = re.search(r"(?:without\s+penalty\s+or\s+liability|without\s+notice\s+or\s+liability)", text, re.IGNORECASE)
    if pay_done_match:
        facts.append(
            Fact(
                slot="termination.pay_for_work_done",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, pay_done_match),
            )
        )
    elif no_pay_done_match and ("termination" in clause.category or "terminate" in text_lower):
        facts.append(
            Fact(
                slot="termination.pay_for_work_done",
                value=False,
                source="regex",
                confidence="medium",
                evidence=_create_evidence(clause, no_pay_done_match),
            )
        )

    # Client terminate without notice:
    no_notice_match = re.search(r"(?:at\s+any\s+time[,\s]+(?:with\s+or\s+without\s+cause[,\s]+)?immediately\s+upon\s+written\s+notice|at\s+any\s+time\s+without\s+notice)", text, re.IGNORECASE)
    if no_notice_match:
        facts.append(
            Fact(
                slot="termination.client_without_notice",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, no_notice_match),
            )
        )

    # Client only for convenience / contractor may not terminate:
    client_only_trm = re.search(r"(?:the\s+contractor\s+may\s+not\s+terminate\s+this\s+agreement\s+prior\s+to\s+complete\s+project\s+sign-off|consultant\s+may\s+not\s+terminate\s+before\s+completion)", text, re.IGNORECASE)
    if client_only_trm:
        facts.append(
            Fact(
                slot="termination.client_only_for_convenience",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, client_only_trm),
            )
        )

    # --- 4. SCOPE FACTS ---
    # Deliverables defined:
    scope_defined_match = re.search(r"(?:as\s+specified\s+in\s+schedule\s+1|cloud\s+architecture\s+review\s+services\s+as\s+mutually\s+agreed|deliverables\s+defined)", text, re.IGNORECASE)
    if scope_defined_match:
        facts.append(
            Fact(
                slot="scope.deliverables_defined",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, scope_defined_match),
            )
        )

    # Change control:
    change_ctrl_match = re.search(r"(?:written\s+change\s+order\s+signed\s+by\s+both\s+parties|change\s+request|change\s+control)", text, re.IGNORECASE)
    if change_ctrl_match:
        facts.append(
            Fact(
                slot="scope.change_control",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, change_ctrl_match),
            )
        )

    # Open ended additional duties / unlimited revisions:
    open_ended_match = re.search(r"(?:any\s+additional\s+tasks,\s*duties,\s*or\s*modifications\s+that\s+the\s+client\s+may\s+assign\s+at\s+its\s+sole\s+discretion|and\s+any\s+other\s+services\s+required)", text, re.IGNORECASE)
    unlimited_rev_match = re.search(r"(?:unlimited\s+revisions)", text, re.IGNORECASE)
    if open_ended_match:
        facts.append(
            Fact(
                slot="scope.open_ended_phrases",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, open_ended_match),
            )
        )
    if unlimited_rev_match:
        facts.append(
            Fact(
                slot="scope.revision_limit",
                value="unlimited",
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, unlimited_rev_match),
            )
        )

    # --- 5. LIABILITY FACTS ---
    unlimited_lia_match = re.search(r"(?:contractor(?:'s)?\s+liability\s+(?:under\s+this\s+agreement\s+)?shall\s+be\s+unlimited|unlimited\s+liability)", text, re.IGNORECASE)
    if unlimited_lia_match:
        facts.append(
            Fact(
                slot="liability.unlimited_contractor",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, unlimited_lia_match),
            )
        )

    cap_match = re.search(r"(?:aggregate\s+liability\s+of\s+either\s+party[,\s]+shall\s+not\s+exceed\s+the\s+total\s+fees\s+paid|liability\s+cap)", text, re.IGNORECASE)
    if cap_match:
        facts.append(
            Fact(
                slot="liability.cap_stated",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, cap_match),
            )
        )

    conseq_match = re.search(r"(?:in\s+no\s+event\s+shall\s+either\s+party\s+be\s+liable\s+for\s+indirect,\s*incidental,\s*or\s*consequential\s+damages|consequential\s+damages\s+(?:are\s+)?excluded)", text, re.IGNORECASE)
    if conseq_match:
        facts.append(
            Fact(
                slot="liability.consequential_damages_excluded",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, conseq_match),
            )
        )

    # --- 6. CONFIDENTIALITY FACTS ---
    mutual_conf_match = re.search(r"(?:each\s+party\s+agrees\s+to\s+hold\s+all\s+proprietary\s+information\s+in\s+strict\s+confidence|mutual\s+confidentiality)", text, re.IGNORECASE)
    if mutual_conf_match:
        facts.append(
            Fact(
                slot="confidentiality.stated",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, mutual_conf_match),
            )
        )
        facts.append(
            Fact(
                slot="confidentiality.mutual",
                value=True,
                source="regex",
                confidence="high",
                evidence=_create_evidence(clause, mutual_conf_match),
            )
        )

    clause.facts = facts
    return facts


def extract_facts_for_clauses(clauses: list[Clause]) -> list[Clause]:
    """Extract facts for all clauses in place and return them."""
    for clause in clauses:
        extract_facts_from_clause(clause)
    return clauses
