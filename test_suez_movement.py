from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_suez_movement():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Fleet", "location": "Red_Sea"})

    orders = {
        1: [Order(1, "Red_Sea", "MOVE", target="Mediterranean_Sea", via_suez=True)]
    }

    env.submit_orders()

    pid, unit = env.get_unit_at("Mediterranean_Sea")

    assert pid == 1

    print("test_suez_movement passed")

if __name__ == "__main__":
    test_suez_movement()
