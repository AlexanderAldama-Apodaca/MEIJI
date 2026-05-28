from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_support_hold():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    # Defender
    env.units[1].append({"type": "Army", "location": "Delhi"})

    # Supporting unit
    env.units[1].append({"type": "Army", "location": "Lucknow"})

    # Attacker
    env.units[2].append({"type": "Army", "location": "Punjab"})

    orders = {
        1: [
            Order(1, "Delhi", "HOLD"),
            Order(1, "Lucknow", "SUPPORT", support_unit="Delhi", support_target=None)
        ],

        2: [Order(2, "Punjab", "MOVE", target="Delhi")]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    assert results["Punjab"] == ["bounce"]

    print("test_support_hold passed")

if __name__ == "__main__":
    test_support_hold()
