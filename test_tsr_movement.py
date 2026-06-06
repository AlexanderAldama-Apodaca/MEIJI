from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_tsr_movement():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[6].append({"type": "Army", "location": "Omsk"})

    env.units[6].append({"type": "Army", "location": "Vladivostok"})

    orders = {
        6: [
            Order(6, "Omsk", "MOVE", target="Irkutsk", via_tsr=True),           # Legal TSR move
            Order(6, "Vladivostok", "MOVE", target="Port_Arthur", via_tsr=True) # Illegal TSR move
        ]
    }

    env.submit_orders(orders)

    env.resolve_orders()

    pid, unit = env.get_unit_at("Irkutsk")

    assert pid == 6

    print("test_tsr_movement passed")

    pid, unit = env.get_unit_at("Vladivostok")

    assert pid == 6
    assert unit is not None

    print("test_invalid_tsr_passed")

if __name__ == "__main__":
    test_tsr_movement()
