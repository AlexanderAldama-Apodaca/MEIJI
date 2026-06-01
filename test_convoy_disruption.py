from colonial_diplomacy_environment import(ColonialDiplomacyEnv, Order)

def test_convoy_disruption():
    env = ColonialDiplomacyEnv()

    for pid in env.units:
        env.units[pid] = []

    # Convoyed army
    env.units[1].append({"type": "Army", "location": "Ceylon"})

    # Convoy fleet
    env.units[1].append({"type": "Fleet", "location": "Gulf_of_Manaar"})

    # Enemy fleets attacking convoy
    env.units[2].append({"type": "Fleet", "location": "Andaman_Sea"})
    env.units[2].append({"type": "Fleet", "location": "Bay_of_Bengal"})

    orders = {
        1: [
            Order(1, "Ceylon", "MOVE", target="Madras", via_convoy=True),
            Order(1, "Gulf_of_Manaar", "CONVOY", convoy_origin="Ceylon", convoy_destination="Madras")
        ],

        2: [
            Order(2, "Andaman_Sea", "MOVE", target="Gulf_of_Manaar"),
            Order(2, "Bay_of_Bengal", "SUPPORT", support_unit="Andaman_Sea", support_target="Gulf_of_Manaar")
        ]
    }

    env.submit_orders(orders)

    results = env.resolve_orders()

    # Convoy should fail if fleet dislodged
    assert results["Ceylon"] == ["bounce"]

    print("test_convoy_disruption passed")

if __name__ == "__main__":
    test_convoy_disruption()
