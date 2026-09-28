"""Offline adversarial fixtures and the blind Astra 1.2 validation plan."""
from typing import Literal

from pydantic import Field

from adaptive_llm_gateway.models.schemas import DomainModel

from .judge import SemanticJudgeRequest


class SemanticBoundaryFixture(DomainModel):
    fixture_id: str = Field(min_length=1)
    phenomenon: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    candidate_summary: str = Field(min_length=1)
    expected: Literal["supported", "unsupported"]
    expected_error_field: Literal[
        "none", "unsupported_claims", "contradictions", "causal_claim_errors",
        "attribution_errors", "quantity_or_time_errors", "certainty_distortions",
    ]


ADVERSARIAL_SEMANTIC_FIXTURES = (
    SemanticBoundaryFixture(fixture_id="supported-lexical-01", phenomenon="lexical_paraphrase",
        source_text="The board postponed the vote until Monday.",
        candidate_summary="The board delayed its vote to Monday.", expected="supported",
        expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-compression-01", phenomenon="semantic_compression",
        source_text="Northbound trains stopped at 08:10 and resumed at 08:25.",
        candidate_summary="Northbound service paused for 15 minutes.", expected="supported",
        expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-description-01", phenomenon="descriptive_characterization",
        source_text="A routing rule directed payment requests to a service that was not accepting traffic.",
        candidate_summary="A faulty routing rule sent payment traffic to an unavailable service.",
        expected="supported", expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-temporal-01", phenomenon="temporal_compression",
        source_text="Engineers identified the defect Tuesday. They deployed a patch Wednesday.",
        candidate_summary="Engineers found the defect Tuesday and patched it Wednesday.",
        expected="supported", expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-attribution-01", phenomenon="attribution_preservation",
        source_text="The city auditor said the contract review was incomplete.",
        candidate_summary="According to the city auditor, the contract review remained unfinished.",
        expected="supported", expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-quantity-01", phenomenon="equivalent_quantity",
        source_text="The shipment contained 1,500 kilograms of rice.",
        candidate_summary="The shipment carried 1.5 metric tons of rice.", expected="supported",
        expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-certainty-01", phenomenon="equivalent_certainty",
        source_text="The update could reduce startup time.",
        candidate_summary="The update may shorten startup time.", expected="supported",
        expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-borderline-01", phenomenon="borderline_compression",
        source_text="The clinic received no vaccine deliveries for three days and canceled appointments.",
        candidate_summary="A three-day delivery gap disrupted the clinic's vaccination schedule.",
        expected="supported", expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-omission-01", phenomenon="nonrequired_omission",
        source_text="The library reopened Friday at 9 a.m. after a two-day closure. The mayor attended.",
        candidate_summary="The library reopened Friday morning after a two-day closure.",
        expected="supported", expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-sequence-01", phenomenon="sequence_without_causation",
        source_text="The alert sounded at 10:02. Staff evacuated at 10:04.",
        candidate_summary="The alert sounded before staff evacuated.", expected="supported",
        expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-event-01", phenomenon="event_compression",
        source_text="The agency opened registration and received 400 applications on the first day.",
        candidate_summary="First-day registration drew 400 applications.", expected="supported",
        expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="supported-description-02", phenomenon="ordinary_characterization",
        source_text="The same invoice was entered twice, and the duplicate entry was removed.",
        candidate_summary="Staff removed a duplicated invoice entry.", expected="supported",
        expected_error_field="none"),
    SemanticBoundaryFixture(fixture_id="unsupported-cause-01", phenomenon="invented_cause",
        source_text="A patch was installed Monday. Error rates fell Tuesday.",
        candidate_summary="Monday's patch caused error rates to fall Tuesday.", expected="unsupported",
        expected_error_field="causal_claim_errors"),
    SemanticBoundaryFixture(fixture_id="unsupported-motive-01", phenomenon="invented_motive",
        source_text="The director canceled the meeting after reading the report.",
        candidate_summary="The director canceled the meeting to conceal the report's findings.",
        expected="unsupported", expected_error_field="unsupported_claims"),
    SemanticBoundaryFixture(fixture_id="unsupported-diagnosis-01", phenomenon="invented_diagnosis",
        source_text="The server restarted repeatedly during the outage.",
        candidate_summary="A memory leak made the server restart repeatedly.", expected="unsupported",
        expected_error_field="unsupported_claims"),
    SemanticBoundaryFixture(fixture_id="unsupported-actor-01", phenomenon="invented_actor",
        source_text="The waiver was approved by the review panel.",
        candidate_summary="The chief counsel approved the waiver.", expected="unsupported",
        expected_error_field="attribution_errors"),
    SemanticBoundaryFixture(fixture_id="unsupported-event-01", phenomenon="invented_event",
        source_text="The store closed early on Saturday.",
        candidate_summary="The store closed early Saturday and refunded all customers.",
        expected="unsupported", expected_error_field="unsupported_claims"),
    SemanticBoundaryFixture(fixture_id="unsupported-quantity-01", phenomenon="changed_quantity",
        source_text="Nine of the twelve sensors passed inspection.",
        candidate_summary="Ten of twelve sensors passed inspection.", expected="unsupported",
        expected_error_field="quantity_or_time_errors"),
    SemanticBoundaryFixture(fixture_id="unsupported-time-01", phenomenon="changed_time",
        source_text="The restriction ends at noon on June 4.",
        candidate_summary="The restriction ends at noon on June 5.", expected="unsupported",
        expected_error_field="quantity_or_time_errors"),
    SemanticBoundaryFixture(fixture_id="unsupported-attribution-01", phenomenon="attribution_swap",
        source_text="Mina proposed the amendment; the committee approved it.",
        candidate_summary="The committee proposed and approved the amendment.", expected="unsupported",
        expected_error_field="attribution_errors"),
    SemanticBoundaryFixture(fixture_id="unsupported-certainty-01", phenomenon="certainty_strengthening",
        source_text="Analysts said the plan might prevent delays.",
        candidate_summary="Analysts said the plan will prevent delays.", expected="unsupported",
        expected_error_field="certainty_distortions"),
    SemanticBoundaryFixture(fixture_id="unsupported-contradiction-01", phenomenon="contradiction",
        source_text="The east entrance remained closed throughout the event.",
        candidate_summary="The east entrance stayed open during the event.", expected="unsupported",
        expected_error_field="contradictions"),
    SemanticBoundaryFixture(fixture_id="unsupported-borderline-01", phenomenon="interpretive_diagnosis",
        source_text="The queue grew while two agents handled requests.",
        candidate_summary="Understaffing caused the queue to grow.", expected="unsupported",
        expected_error_field="causal_claim_errors"),
    SemanticBoundaryFixture(fixture_id="unsupported-order-01", phenomenon="changed_order",
        source_text="The inspection ended before the permit was issued.",
        candidate_summary="The permit was issued before the inspection ended.", expected="unsupported",
        expected_error_field="quantity_or_time_errors"),
)


def _request(source: str, candidate: str) -> SemanticJudgeRequest:
    return SemanticJudgeRequest(
        source_text=source,
        candidate_summary=candidate,
        semantic_requirements=("Preserve the source's material facts and attribution.",),
        output_constraints={"max_words": 30},
    )


# Expected labels are deliberately stored outside these blind requests.
ASTRA_1_2_VALIDATION_REQUESTS: dict[str, SemanticJudgeRequest] = {
    "live-01": _request(
        "A routing policy sent account updates to a backend that was not serving requests.",
        "A faulty routing policy directed account updates to an unavailable backend."),
    "live-02": _request(
        "The ferry stopped running at 14:00 and resumed at 14:40.",
        "Ferry service paused for 40 minutes."),
    "live-03": _request(
        "The safety office reported that the west stairwell inspection was incomplete.",
        "According to the safety office, the west stairwell inspection remained unfinished."),
    "live-04": _request(
        "A fault appeared Thursday. Technicians replaced the controller Friday.",
        "The fault appeared Thursday, followed by a controller replacement Friday."),
    "live-05": _request(
        "The configuration was updated Monday. Login failures declined Tuesday.",
        "Monday's configuration update caused Tuesday's decline in login failures."),
    "live-06": _request(
        "The analyst left the review after receiving the revised figures.",
        "The analyst left the review because the revised figures revealed fraud."),
    "live-07": _request(
        "Seven of ten sites opened by 09:30 on April 8.",
        "Eight of ten sites opened by 09:00 on April 8."),
    "live-08": _request(
        "The inspector said the bridge may reopen this week; the city made no forecast.",
        "The city said the bridge will reopen this week."),
}

ASTRA_1_2_VALIDATION_EXPECTATIONS: dict[str, Literal["supported", "unsupported"]] = {
    "live-01": "supported", "live-02": "supported", "live-03": "supported",
    "live-04": "supported", "live-05": "unsupported", "live-06": "unsupported",
    "live-07": "unsupported", "live-08": "unsupported",
}
