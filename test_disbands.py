from colonial_diplomacy_environment import (ColonialDiplomacyEnv)

def test_disbands():
    env = ColonialDiplomacyEnv()

    player = 1

    env.units[player] = [
        {"type": "Army", "location": "Delhi"}
    ]

    assert len(env.units[player]) == 1

    env.disband_unit(player, "Delhi")

    assert len(env.units[player]) == 0

    print ("test_disbands passed")

if __name__ == "__main__":
    test_disbands()
