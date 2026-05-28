from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_tsr_movement():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[6].append({"type": "Army", "location": "Omsk"})

    orders = {
        1: [Order(6, "Omsk", "MOVE", target="Irkutsk", via_tsr=True)]
    }

    env.submit_orders(orders)

    env.resolve_orders()

    pid, unit = env.get_unit_at("Irkutsk")

    assert pid == 1

    print("test_tsr_movement passed")

if __name__ == "__main__":
    test_tsr_movement()
