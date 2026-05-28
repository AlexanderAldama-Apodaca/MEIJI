from colonial_diplomacy_environment import (ColonialDiplomacyEnv, Order)

def test_basic_convoy():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    env.units[1].append({"type": "Army", "location": "Ceylon"})

    env.units[1].append({"type": "Fleet", "location": "Gulf_of_Manaar"})

    env.units[1].append({"type": "Fleet", "location": "Bay_of_Bengal"})

    orders = {
        1: [
            Order(1, "Ceylon", "MOVE", target="Rangoon", via_convoy=True),
            Order(1, "Gulf_of_Manaar", "CONVOY", convoy_origin="Ceylon", convoy_destination="Rangoon"),
            Order(1, "Bay_of_Bengal", "CONVOY", convoy_origin="Ceylon", convoy_destination="Rangoon")
        ]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    pid, unit = env.get_unit_at("Rangoon")

    assert pid == 1

    print("test_basic_convoy passed")

if __name__ == "__main__":
    test_basic_convoy()
