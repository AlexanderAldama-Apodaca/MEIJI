from colonial_diplomacy_environment import(ColonialDiplomacyEnv, Order)

def test_self_dislodgement_prevention():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Army", "location": "Punjab"})

    env.units[1].append({"type": "Army", "location": "Delhi"})

    orders = {
        1: [
            Order(1, "Punjab", "MOVE", target="Delhi"),
            Order(1, "Delhi", "HOLD")
        ]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    assert results["Punjab"] == ["bounce"]

    print("test_self_dislodgement_prevention passed")

if __name__ == "__main__":
    test_self_dislodgement_prevention()
