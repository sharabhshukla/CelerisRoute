import itertools

import numpy as np
import pytest

import celerisroute as cr

PORTS = ["Singapore", "Dubai", "Rotterdam", "Houston", "Mumbai"]
INVENTORY = [0.0, 0.25, 0.50, 0.75, 1.0]


@pytest.fixture
def vessel_states() -> cr.StateSpace:
    return cr.StateSpace(day=range(1, 36), port=PORTS, inventory=INVENTORY)


def test_size_shape_and_names(vessel_states):
    assert len(vessel_states) == 875
    assert vessel_states.size == 875
    assert vessel_states.shape == (35, 5, 5)
    assert vessel_states.dimension_names == ("day", "port", "inventory")


def test_ids_are_mixed_radix_in_declaration_order(vessel_states):
    # Row-major: the last dimension varies fastest.
    assert vessel_states.id((1, "Singapore", 0.0)) == 0
    assert vessel_states.id((1, "Singapore", 0.25)) == 1
    assert vessel_states.id((1, "Dubai", 0.0)) == 5
    assert vessel_states.id((2, "Singapore", 0.0)) == 25
    assert vessel_states.id((35, "Mumbai", 1.0)) == 874
    assert vessel_states.id((4, "Dubai", 0.25)) == 3 * 25 + 1 * 5 + 1


def test_keyword_tuple_and_mapping_forms_agree(vessel_states):
    expected = vessel_states.id((4, "Dubai", 0.25))
    assert vessel_states.id(day=4, port="Dubai", inventory=0.25) == expected
    assert vessel_states.id(inventory=0.25, day=4, port="Dubai") == expected
    assert vessel_states.id({"day": 4, "port": "Dubai", "inventory": 0.25}) == expected
    assert vessel_states.id([4, "Dubai", 0.25]) == expected


def test_decode_inverts_id_for_every_state(vessel_states):
    expected = itertools.product(range(1, 36), PORTS, INVENTORY)
    for state_id, state in enumerate(expected):
        assert vessel_states.decode(state_id) == state
        assert vessel_states.id(state) == state_id


def test_iteration_matches_decode():
    states = cr.StateSpace(a=[3, 1, 2], b=["x", "y"])
    assert list(states) == [states.decode(i) for i in range(len(states))]
    assert states.decode(0) == (3, "x")  # insertion order, not sorted order


def test_decode_many(vessel_states):
    ids = np.array([0, 86, 874])
    assert vessel_states.decode_many(ids) == [vessel_states.decode(int(i)) for i in ids]
    assert vessel_states.decode_many([]) == []


def test_vectorised_ids_match_scalar_ids(vessel_states):
    rng = np.random.default_rng(0)
    days = rng.integers(1, 36, size=200)
    ports = rng.choice(PORTS, size=200)
    inventory = rng.choice(INVENTORY, size=200)
    ids = vessel_states.ids(days, ports, inventory)
    assert ids.dtype == np.int64
    expected = [
        vessel_states.id((int(d), str(p), float(i))) for d, p, i in zip(days, ports, inventory)
    ]
    assert ids.tolist() == expected
    assert vessel_states.ids(day=days, port=ports, inventory=inventory).tolist() == expected


def test_vectorised_ids_broadcast_scalars(vessel_states):
    ids = vessel_states.ids(np.arange(1, 36), "Dubai", 0.5)
    assert ids.tolist() == [vessel_states.id((d, "Dubai", 0.5)) for d in range(1, 36)]


def test_supported_value_types():
    states = cr.StateSpace(
        number=[10, 20],
        real=[0.5, 1.5],
        text=["a", "b"],
        pair=[(0, "pre"), (0, "post"), (1, "pre")],
    )
    assert states.shape == (2, 2, 2, 3)
    state = (20, 1.5, "b", (0, "post"))
    assert states.decode(states.id(state)) == state
    assert state in states
    assert (20, 1.5, "b", (9, "post")) not in states


def test_large_space_is_not_materialised():
    states = cr.StateSpace(a=range(10**6), b=range(10**6), c=range(10**6))
    assert states.size == 10**18
    last = (10**6 - 1, 10**6 - 1, 10**6 - 1)
    assert states.id(last) == 10**18 - 1
    assert states.decode(10**18 - 1) == last


def test_range_with_step():
    states = cr.StateSpace(t=range(10, 0, -3))  # 10, 7, 4, 1
    assert [states.id((t,)) for t in (10, 7, 4, 1)] == [0, 1, 2, 3]
    with pytest.raises(cr.InvalidStateError):
        states.id((8,))


def test_equality():
    a = cr.StateSpace(x=[1, 2], y=["p", "q"])
    assert a == cr.StateSpace(x=[1, 2], y=["p", "q"])
    assert a != cr.StateSpace(x=[2, 1], y=["p", "q"])
    assert a != cr.StateSpace(y=["p", "q"], x=[1, 2])


def test_positional_mapping_constructor():
    states = cr.StateSpace({"class": [1, 2], "id": ["u", "v"]})
    assert states.dimension_names == ("class", "id")
    assert states.id({"class": 2, "id": "u"}) == 2


# ------------------------------------------------------------------ invalid


def test_duplicate_values_are_rejected():
    with pytest.raises(ValueError, match="port"):
        cr.StateSpace(day=range(3), port=["Dubai", "Doha", "Dubai"])
    # 1 and 1.0 are the same dictionary key, hence the same state value.
    with pytest.raises(ValueError):
        cr.StateSpace(level=[1, 1.0])


def test_empty_and_malformed_dimensions_are_rejected():
    with pytest.raises(ValueError):
        cr.StateSpace()
    with pytest.raises(ValueError):
        cr.StateSpace(day=[])
    with pytest.raises(TypeError):
        cr.StateSpace(port="Dubai")  # a string is not a list of values
    with pytest.raises(TypeError):
        cr.StateSpace(bad=[[1, 2], [3, 4]])  # unhashable values
    with pytest.raises(ValueError):
        cr.StateSpace(level=[0.0, float("nan")])


def test_unknown_values_are_rejected(vessel_states):
    with pytest.raises(cr.InvalidStateError, match="port"):
        vessel_states.id((4, "Atlantis", 0.25))
    with pytest.raises(cr.InvalidStateError):
        vessel_states.id((36, "Dubai", 0.25))
    with pytest.raises(cr.InvalidStateError):
        vessel_states.id((4, "Dubai", 0.3))
    with pytest.raises(cr.InvalidStateError):
        vessel_states.id(day=4, port="Dubai")  # inventory missing
    with pytest.raises(cr.InvalidStateError):
        vessel_states.id(day=4, port="Dubai", inventory=0.25, speed=3)
    with pytest.raises(cr.InvalidStateError):
        vessel_states.id((4, "Dubai"))
    with pytest.raises(TypeError):
        vessel_states.id(86)  # an id is not a state
    with pytest.raises(TypeError):
        vessel_states.id((4, "Dubai", 0.25), day=4)


def test_invalid_state_error_is_a_value_error(vessel_states):
    with pytest.raises(ValueError):
        vessel_states.id((4, "Atlantis", 0.25))


def test_out_of_range_ids_are_rejected(vessel_states):
    for bad in (-1, 875, 10**9):
        with pytest.raises(cr.InvalidStateError):
            vessel_states.decode(bad)
    with pytest.raises(TypeError):
        vessel_states.decode(1.5)
    with pytest.raises(cr.InvalidStateError):
        vessel_states.decode_many([0, 875])


def test_vectorised_ids_reject_unknown_values(vessel_states):
    with pytest.raises(cr.InvalidStateError):
        vessel_states.ids(np.array([1, 99]), "Dubai", 0.0)
    with pytest.raises(cr.InvalidStateError):
        vessel_states.ids(np.array([1, 2]), np.array(["Dubai", "Atlantis"]), 0.0)
    with pytest.raises(cr.InvalidStateError):
        vessel_states.ids(np.array([1, 2]), "Dubai", np.array([0.0, 0.3]))
    with pytest.raises(ValueError):
        vessel_states.ids(np.array([1, 2, 3]), np.array(["Dubai", "Mumbai"]), 0.0)
