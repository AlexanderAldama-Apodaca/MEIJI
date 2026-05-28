from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_supported_attack():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Army", "location": "Punjab"})

    env.units[1].append({"type": "Army", "location": "Delhi"})

    env.units[2].append({"type": "Army", "location": "Rajputana"})

    orders = {
        1: [
            Order(1, "Punjab", "MOVE", target="Rajputana"),
            Order(1, "Delhi", "SUPPORT", support_unit="Punjab", support_target="Rajputana")
        ],

        2: [
            Order(2, "Rajputana", "HOLD")
        ]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    pid, unit = env.get_unit_at("Rajputana")

    assert pid == 1

    print("test_supported_attack passed")

if __name__ == "__main__":
    test_supported_attack()
