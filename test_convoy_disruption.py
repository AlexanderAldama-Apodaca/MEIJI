from colonial_diplomacy_environment import(ColonialDiplomacyEnv, Order)

def test_convoy_disruption():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    # Convoyed army
    env.units[1].append({"type": "Army", "location": "Ceylon"})

    # Convoy fleet
    env.units[1].append({"type": "Fleet", "location": "Bay_of_Bengal"})

    # Enemy fleet attacking convoy
    env.units[2].append({"type": "Fleet", "location": "Andaman_Sea"})

    orders = {
        1: [
            Order(1, "Ceylon", "MOVE", target="Rangoon", via_convoy=True),
            Order(1, "Bay_of_Bengal", "CONVOY", convoy_origin="Ceylon", convoy_destination="Rangoon")
        ],

        2: [Order(2, "Andaman_Sea", "MOVE", target="Bay_of_Bengal")]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    # Convoy should fail if fleet dislodged
    assert results["Ceylon"] == ["bounce"]

    print("test_convoy_disruption passed")

if __name__ == "__main__":
    test_convoy_disruption()
