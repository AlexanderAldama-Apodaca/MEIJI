from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_retreat_conflict():
    env = ColonialDiplomacyEnv()

    env.pending_retreats = [
        {"player_id": 1,"unit": {"type": "Army", "location": "Punjab"}, "from": "Punjab", "retreat_options": ["Delhi"]},

        {"player_id": 2, "unit": {"type": "Army", "location": "Rajputana"}, "from": "Rajputana", "retreat_options": ["Delhi"]}
    ]

    retreat_orders = [
        Order(1, "Punjab", "RETREAT", target="Delhi"),

        Order(2, "Rajputana", "RETREAT", target="Delhi")
    ]

    env.resolve_retreats(retreat_orders)

    # Retreating units should not exist
    pid1, unit1 = env.get_unit_at("Punjab")
    pid2, unit2 = env.get_unit_at("Rajputana")

    assert unit1 is None
    assert unit2 is None

    # Delhi should not contain retreating units
    delhi_units = []

    for pid, units in env.units.items():
        for unit in units:
            if unit["location"] == "Delhi":
                delhi_units.append(unit)

    # Only the original Delhi unit should exist
    assert len(delhi_units) == 1

    print("test_retreat_conflict passed")

if __name__ == "__main__":
    test_retreat_conflict()
