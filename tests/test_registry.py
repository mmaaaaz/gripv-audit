"""Registry sanity tests — importing a contract must not need dataset access."""

from gripv import contracts


def test_registry_is_non_empty():
    assert contracts.CONTRACTS


def test_get_rejects_unknown_category():
    try:
        contracts.get("definitely_not_a_category")
    except SystemExit as exc:
        assert "No contract" in str(exc)
    else:
        raise AssertionError("expected SystemExit for an unknown category")


def test_registered_contracts_are_classes():
    for name, cls in contracts.CONTRACTS.items():
        assert isinstance(cls, type), name
