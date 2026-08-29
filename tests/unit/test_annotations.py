"""Tests for accentedge_benchmark.annotations — entity handling, validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from accentedge_benchmark.schemas import CriticalEntity, EntityType, PronunciationToken, SourceStatus
from accentedge_benchmark.annotations.entities import (
    normalize_entity,
    normalize_money,
    normalize_date,
    match_entities,
    compute_entity_error_rate,
    compute_correction_damage,
)
from accentedge_benchmark.annotations.validation import (
    validate_entities,
    validate_pronunciation_tokens,
)


class TestNormalizeMoney:
    def test_dollar_sign(self):
        assert normalize_money("$50") == "USD:50"

    def test_dollars_word(self):
        assert normalize_money("50 dollars") == "USD:50.00"

    def test_cents_word(self):
        assert normalize_money("25 cents") == "USD:0.25"

    def test_mixed(self):
        assert normalize_money("pay $50.25 or 100 dollars") == "pay USD:50.25 or USD:100.00"

    def test_no_dollar_sign_unchanged(self):
        assert normalize_money("50") == "50"


class TestNormalizeDate:
    def test_january(self):
        assert normalize_date("January 5") == "01/5"

    def test_february(self):
        assert normalize_date("February 29") == "02/29"

    def test_december(self):
        assert normalize_date("December 25") == "12/25"

    def test_no_date_unchanged(self):
        assert normalize_date("hello world") == "hello world"

    def test_case_insensitive(self):
        assert normalize_date("january 5") == "01/5"


class TestNormalizeEntity:
    def test_money_type(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=3,
        )
        assert normalize_entity(entity) == "USD:50.00"

    def test_date_type(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.DATE,
            surface="January 5", normalized="01/5",
            start_char=0, end_char=10,
        )
        assert normalize_date(entity.normalized) is not None

    def test_number_type(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.NUMBER,
            surface="42", normalized="42",
            start_char=0, end_char=2,
        )
        assert normalize_entity(entity) == "42"

    def test_person_name_type(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.PERSON_NAME,
            surface="John", normalized="John",
            start_char=0, end_char=4,
        )
        assert normalize_entity(entity) == "John"

    def test_alphanumeric_type(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.ALPHANUMERIC,
            surface="ABC123", normalized="ABC123",
            start_char=0, end_char=6,
        )
        assert normalize_entity(entity) == "ABC123"


class TestMatchEntities:
    def test_exact_match(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=3,
        )
        matches = match_entities([entity], "$50")
        assert len(matches) >= 1
        assert any(m.correct for m in matches)

    def test_no_match(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=3,
        )
        matches = match_entities([entity], "hello world")
        assert len(matches) == 1
        assert not matches[0].correct

    def test_compute_error_rate(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=3,
        )
        matches = match_entities([entity], "$50")
        stats = compute_entity_error_rate(matches)
        assert stats["accuracy"] == 1.0
        assert stats["error_rate"] == 0.0


class TestValidateEntities:
    def test_valid_entities(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=3,
        )
        result = validate_entities([entity], "$50 is the amount")
        assert result.valid is True
        assert result.issues == []

    def test_span_outside_transcript(self):
        entity = CriticalEntity(
            entity_id="e1", utterance_id="u1",
            entity_type=EntityType.MONEY,
            surface="$50", normalized="USD:50.00",
            start_char=0, end_char=100,
        )
        result = validate_entities([entity], "$50 is the amount")
        assert result.valid is False
        assert any("span outside" in issue for issue in result.issues)

    def test_end_before_start(self):
        with pytest.raises(ValidationError):
            CriticalEntity(
                entity_id="e1", utterance_id="u1",
                entity_type=EntityType.MONEY,
                surface="$50", normalized="USD:50.00",
                start_char=5, end_char=2,
            )

    def test_multiple_entities_all_valid(self):
        entities = [
            CriticalEntity(
                entity_id=f"e{i}", utterance_id="u1",
                entity_type=EntityType.MONEY,
                surface="$50", normalized="USD:50.00",
                start_char=0, end_char=3,
            )
            for i in range(3)
        ]
        result = validate_entities(entities, "$50 is the amount here")
        assert result.valid is True


class TestValidatePronunciationTokens:
    def test_valid_tokens(self):
        tok = PronunciationToken(
            token_id="t1", utterance_id="u1", word="hello",
            feature="h-vowel", phone_label="HH AH L OW",
            start_ms=0.0, end_ms=500.0,
            source_status=SourceStatus.DEVIANT,
        )
        result = validate_pronunciation_tokens([tok], 1000.0)
        assert result.valid is True
        assert result.issues == []

    def test_negative_start_raises(self):
        with pytest.raises(ValidationError):
            tok = PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=-10.0, end_ms=500.0,
                source_status=SourceStatus.DEVIANT,
            )

    def test_end_beyond_duration(self):
        tok = PronunciationToken(
            token_id="t1", utterance_id="u1", word="hello",
            feature="h-vowel", phone_label="HH AH L OW",
            start_ms=0.0, end_ms=2000.0,
            source_status=SourceStatus.DEVIANT,
        )
        result = validate_pronunciation_tokens([tok], 1000.0)
        assert result.valid is False
        assert any("beyond audio duration" in issue for issue in result.issues)

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="end_ms must be >= start_ms"):
            PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=500.0, end_ms=100.0,
                source_status=SourceStatus.DEVIANT,
            )

    def test_all_statuses_valid(self):
        for status in SourceStatus:
            tok = PronunciationToken(
                token_id="t1", utterance_id="u1", word="hello",
                feature="h-vowel", phone_label="HH AH L OW",
                start_ms=0.0, end_ms=500.0,
                source_status=status,
            )
            assert tok.source_status == status


class TestComputeCorrectionDamage:
    def test_all_corrected(self):
        source = [
            PronunciationToken(
                token_id=f"t{i}", utterance_id="u1", word="hello",
                feature="f", phone_label="HH", start_ms=0.0, end_ms=100.0,
                source_status=SourceStatus.DEVIANT,
            )
            for i in range(3)
        ]
        output = [
            PronunciationToken(
                token_id=f"t{i}", utterance_id="u1", word="hello",
                feature="f", phone_label="HH", start_ms=0.0, end_ms=100.0,
                source_status=SourceStatus.ALREADY_TARGET,
            )
            for i in range(3)
        ]
        corrected, damaged, ambiguous = compute_correction_damage(source, output)
        assert corrected == 3
        assert damaged == 0
        assert ambiguous == 0

    def test_all_damaged(self):
        source = [
            PronunciationToken(
                token_id=f"t{i}", utterance_id="u1", word="hello",
                feature="f", phone_label="HH", start_ms=0.0, end_ms=100.0,
                source_status=SourceStatus.ALREADY_TARGET,
            )
            for i in range(3)
        ]
        output = [
            PronunciationToken(
                token_id=f"t{i}", utterance_id="u1", word="hello",
                feature="f", phone_label="HH", start_ms=0.0, end_ms=100.0,
                source_status=SourceStatus.DEVIANT,
            )
            for i in range(3)
        ]
        corrected, damaged, ambiguous = compute_correction_damage(source, output)
        assert corrected == 0
        assert damaged == 3
        assert ambiguous == 0

    def test_all_ambiguous(self):
        source = [
            PronunciationToken(
                token_id="t0", utterance_id="u1", word="hello",
                feature="f", phone_label="HH", start_ms=0.0, end_ms=100.0,
                source_status=SourceStatus.AMBIGUOUS,
            )
        ]
        output = [
            PronunciationToken(
                token_id="t0", utterance_id="u1", word="hello",
                feature="f", phone_label="HH", start_ms=0.0, end_ms=100.0,
                source_status=SourceStatus.AMBIGUOUS,
            )
        ]
        corrected, damaged, ambiguous = compute_correction_damage(source, output)
        assert ambiguous == 1
