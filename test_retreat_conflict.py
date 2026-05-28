from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_retreat_conflict():
    env = ColonialDiplomacyEnv()

    env.pending_retreats = [
        {
            "player": 1,
            "unit": {"type": "Army", "location": "Punjab"},
            "retreats": ["Delhi"]
        },

        {
            "player": 2,
            "unit": {"type": "Army", "location": "Rajputana"},
            "retreats": ["Delhi"]
        }
    ]

    retreat_orders = [
        Order(1, "Punjab", "RETREAT", target="Delhi"),
        Order(2, "Rajputana", "RETREAT", target="Delhi")
    ]

    env.resolve_retreats(retreat_orders)

    # Neither retreat should succeed
    pid, unit = env.get_unit_at("Delhi")

    assert unit is None
    
    print("test_retreat_conflict passed")

if __name__ == "__main__":
    test_retreat_conflict()
