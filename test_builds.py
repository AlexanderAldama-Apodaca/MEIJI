from colonial_diplomacy_environment import (
    ColonialDiplomacyEnv,
    Order
)

def test_builds():

    env = ColonialDiplomacyEnv()

    player_id = 1

    # Give Britain an extra controlled SC
    env.controlled_supply_centers[player_id].append(
        "Karachi"
    )

    # Add Karachi as a temporary home center
    env.home_supply_centers["Karachi"] = player_id

    # Ensure Karachi empty
    env.units[player_id] = [
        u for u in env.units[player_id]
        if u["location"] != "Karachi"
    ]

    builds_before = len(
        env.units[player_id]
    )

    build_orders = [
        Order(
            player_id,
            None,
            "BUILD",
            target="Karachi",
            build_type="Army"
        )
    ]

    env.resolve_builds(build_orders)

    builds_after = len(
        env.units[player_id]
    )

    assert builds_after == builds_before + 1

    pid, unit = env.get_unit_at("Karachi")

    assert pid == player_id
    assert unit["type"] == "Army"

    print("test_builds passed")

if __name__ == "__main__":
    test_builds()
