"""
Cross-clause consistency checker for Contract Autopsy Agent.
Flags conflicting values for the same slot across different clauses in the same document.
Implements Golden Test G8 requirements (sets evidence_status='conflicting', rating at least YELLOW).
"""

from __future__ import annotations

from src.ca.schemas import Clause, RiskFinding, EvidenceSpan


def check_consistency(
    findings: list[RiskFinding],
    clauses: list[Clause],
) -> list[RiskFinding]:
    """
    Examine clauses and findings for cross-clause fact conflicts.
    If different clauses state conflicting values for the same slot:
    - Mark both findings with evidence_status = "conflicting"
    - Elevate rating to at least "YELLOW"
    - Append cross-referencing evidence quotes and conflict notes
    """
    # Map slot -> list of (clause_id, fact_value, evidence_span)
    slot_map: dict[str, list[tuple[str, object, EvidenceSpan | None]]] = {}

    for clause in clauses:
        for f in clause.facts:
            if f.value is not None:
                if f.slot not in slot_map:
                    slot_map[f.slot] = []
                slot_map[f.slot].append((clause.clause_id, f.value, f.evidence))

    # Identify slots with multiple distinct values
    conflicting_slots: dict[str, list[tuple[str, object, EvidenceSpan | None]]] = {}
    for slot, occurrences in slot_map.items():
        distinct_vals = {str(val) for _, val, _ in occurrences}
        if len(distinct_vals) > 1:
            conflicting_slots[slot] = occurrences

    if not conflicting_slots:
        return findings

    # Update findings matching the conflicting clauses
    for slot, conflicts in conflicting_slots.items():
        clause_ids = [cid for cid, _, _ in conflicts]
        for finding in findings:
            if finding.clause_id in clause_ids:
                other_entries = [c for c in conflicts if c[0] != finding.clause_id]
                other_cids = [c[0] for c in other_entries]

                finding.evidence_status = "conflicting"
                finding.conflicts_with = list(set(finding.conflicts_with + other_cids))

                # Elevate rating to at least YELLOW
                if finding.rating in ("GREEN", "UNRATED"):
                    finding.rating = "YELLOW"
                    finding.score = max(finding.score or 0, 50)

                # Add explanation
                conflict_details = []
                for o_cid, o_val, o_ev in other_entries:
                    quote_txt = f" (\"{o_ev.quote}\")" if o_ev else ""
                    conflict_details.append(f"clause {o_cid} stating {slot}={o_val}{quote_txt}")
                    if o_ev and o_ev not in finding.evidence:
                        finding.evidence.append(o_ev)

                finding.plain_reason += (
                    f" CONFLICT DETECTED: This term conflicts with {', '.join(conflict_details)}."
                )

    return findings
