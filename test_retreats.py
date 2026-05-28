from colonial_diplomacy_environment import(ColonialDiplomacyEnv, Order)

def test_retreat_generation():
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

        2: [Order(2, "Rajputana", "HOLD")]
    }

    env.submit_orders(orders)

    env.resolve_orders()

    assert len(env.pending_retreats) == 1

    retreat = env.pending_retreats[0]

    assert retreat["unit"]["location"] == "Rajputana"

    print("test_retreat_generation passed")

if __name__ == "__main__":
    test_retreat_generation()
