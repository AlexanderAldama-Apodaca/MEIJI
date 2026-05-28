from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_head_to_head():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Army", "location": "Punjab"})

    env.units[2].append({"type": "Army", "location": "Delhi"})

    orders = {
        1: [Order(1, "Punjab", "MOVE", target="Delhi")],
        
        2: [Order(2, "Delhi", "MOVE", target="Punjab")]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    assert results["Punjab"] == ["bounce"]
    assert results["Delhi"] == ["bounce"]

    print("test_head_to_head passed")

if __name__ == "__main__":
    test_head_to_head()
