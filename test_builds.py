from colonial_diplomacy_environment import (ColonialDiplomacyEnv)

def test_builds():
    env = ColonialDiplomacyEnv()

    player = 1

    env.supply_centers["Delhi"] = player

    env.units[player] = []

    builds_before = len(env.units[player])

    env.build_unit(player, "Army", "Delhi")

    builds_after = len(env.units[player])

    assert builds_after == builds_before + 1

    pid, unit = env.get_unit_at("Delhi")

    assert pid == player

    print("test_builds passed")

if __name__ == "__main__":
    test_builds()
