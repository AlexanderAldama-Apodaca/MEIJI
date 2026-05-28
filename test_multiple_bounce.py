from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_multiple_bounce():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Army", "location": "Punjab"})

    env.units[2].append({"type": "Army", "location": "Delhi"})

    env.units[3].append({"type": "Army", "location": "Lucknow"})

    orders = {
        1: [Order(1, "Punjab", "MOVE", target="Rajputana")],

        2: [Order(2, "Delhi", "MOVE", target="Rajputana")],

        3: [Order(3, "Lucknow", "MOVE", target="Rajputana")]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    assert results["Punjab"] == ["bounce"]
    assert results["Delhi"] == ["bounce"]
    assert results["Lucknow"] == ["bounce"]

    print("test_multiple_bounce passed")

if __name__ == "__main__":
    test_multiple_bounce()
