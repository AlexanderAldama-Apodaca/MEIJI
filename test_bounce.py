from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_simple_bounce():
    env = ColonialDiplomacyEnv()

    # Clear board
    for pid in env.units:
        env.units[pid] = []

    # Two armies attacking same province
    env.units[1].append({"type": "Army", "location": "Punjab"})

    env.units[2].append({"type": "Army", "location": "Delhi"})

    orders = {
        1: [Order(1, "Punjab", "MOVE", target="Rajputana")],
        2: [Order(2, "Delhi", "MOVE", target="Rajputana")]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    assert results["Punjab"] == ["bounce"]
    assert results["Delhi"] == ["bounce"]

    print("test_simple_bounce passed")
    
if __name__ == "__main__":
    test_simple_bounce()
