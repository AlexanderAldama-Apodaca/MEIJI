from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_support_cut():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

        env.units[1].append({"type": "Army", "location": "Punjab"})

        env.units[1].append({"type": "Army", "location": "Delhi"})

        env.units[2].append({"type": "Army", "location": "Lucknow"})

        env.units[3].append({"type": "Army", "location": "Rajputana"})

        orders = {
            1: [
                Order(1, "Punjab", "MOVE", target="Rajputana"),
                Order(1, "Delhi", "SUPPORT", support_unit="Punjab", support_target="Rajputana")
            ],

            2: [Order(2, "Lucknow", "MOVE", target="Delhi")],

            3: [Order(3, "Rajputana", "Hold")]
        }

        env.submit_orders(orders)

        results = env.resolve_orders()

        assert results["Punjab"] == ["bounce"]

    print("test_support_cut passed")

if __name__ == "__main__":
    test_support_cut()
