from colonial_diplomacy_environment import(ColonialDiplomacyEnv, Order)

def test_illegal_move():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Army", "location": "Punjab"})

    orders = {Order(1, "Punjab", "MOVE", target="Tokyo")}

    env.submit_orders(orders)

    assert "Punjab" in env.invalid_orders

    print("test_illegal_move passed")

if __name__ == "__main__":
    test_illegal_move()
