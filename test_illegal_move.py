from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_illegal_move():
    env = ColonialDiplomacyEnv()

    orders = [Order(1, "Punjab", "MOVE", target="Tokyo")]

    env.pending_orders = {1: orders}

    results = env.resolve_orders()

    assert "Punjab" in env.invalid_orders

    print("test_illegal_move passed")

if __name__ == "__main__":
    test_illegal_move()
