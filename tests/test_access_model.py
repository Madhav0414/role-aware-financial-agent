"""The permission model is configuration, so these tests read the real
config/roles.yaml rather than a fixture. A change to the shipped roles that
breaks the demonstrated behaviour should fail the suite.
"""

from pathlib import Path

from src.access.model import Tag, load_roles

ROLES_PATH = Path("config/roles.yaml")


def test_ceo_sees_everything():
    roles = load_roles(ROLES_PATH)
    assert roles["CEO"].allowed_tags == frozenset(Tag)
    assert roles["CEO"].recent_years_only is None


def test_cto_is_denied_only_the_hr_tags():
    cto = load_roles(ROLES_PATH)["CTO"]
    assert Tag.HR_COMPENSATION not in cto.allowed_tags
    assert Tag.HR_HEADCOUNT not in cto.allowed_tags
    # Everything else stays readable — a denial must be surgical, not a
    # blanket restriction that happens to cover the right tags.
    assert cto.allowed_tags == frozenset(Tag) - {Tag.HR_COMPENSATION,
                                                 Tag.HR_HEADCOUNT}
    assert cto.recent_years_only is None


def test_analyst_is_restricted_on_two_dimensions():
    analyst = load_roles(ROLES_PATH)["ANALYST"]
    assert analyst.allowed_tags == frozenset({Tag.FIN_STATEMENTS,
                                              Tag.FIN_SEGMENT})
    assert analyst.recent_years_only == 2


def test_roles_are_immutable():
    """A role handed to a gate must not be mutable by the caller — that would
    turn an access decision into shared mutable state."""
    import dataclasses
    import pytest

    ceo = load_roles(ROLES_PATH)["CEO"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ceo.name = "IMPOSTOR"
