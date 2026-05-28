from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_coastal_fleet_movement():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Fleet", "location": "Arabia_south_coast"})

    orders = {
        1: [Order(1, "Arabia_south_coast", "MOVE", target="Arabian_Sea")]
    }

    env.submit_orders(orders)

    env.resolve_orders()

    pid, unit = env.get_unit_at("Arabian_Sea")

    assert pid == 1

    print("test_coastal_fleet_movement passed")

if __name__ == "__main__":
    test_coastal_fleet_movement()
