import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque

class ColonialDiplomacyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.num_players: int = self.config.get("num_players", 7) # Britain, China, France, Holland, Japan, Russia, Turkey
        self.max_years: int = self.config.get("max_years", 1908) # Colonial Diplomacy ends after 1908
        self.max_units: int = self.config.get("max_units", 58)
        
        # Game State Variables
        self.year: int = 1870 # Colonial Diplomacy begins in 1870
        self.phase: str = "Spring" # Spring / Fall / Retreat / Build
        self.done: bool = False

        self.tsr_used_this_turn: bool = False
        self.suez_permissions: Dict[int, set] = {}

        # Board representation
        self.provinces: List[str] = ["Abyssinia", "Aden", "Afghanistan", "Akita", "Akmolinsk", "Andaman_Sea", "Angora", "Annam", "Arabia", "Arabia_north_coast", "Arabia_south_coast", "Arabian_Sea", "Armenia", "Assam", "Baghdad", "Baku", "Bangkok", "Bangkok_east_coast", "Bangkok_west_coast", "Bay_of_Bengal", "Bengal", "Black_Sea", "Bokhara", "Bombay", "Borneo", "Cambodia", "Canton", "Cebu", "Celebes", "Celebes_Sea", "Ceylon", "Chungking", "Cochin", "Constantinople", "Davao", "Delhi", "East_China_Sea", "East_Indian_Ocean", "Egypt", "Eritrea", "Formosa", "Fusan", "Gulf_of_Aden", "Gulf_of_Manaar", "Gulf_of_Siam", "Hong_Kong", "Hyderabad", "Irkutsk", "Java", "Java_Sea", "Karachi", "Kashgar", "Kashmir", "Kirghiz", "Krasnoyarsk", "Kyoto", "Kyushu", "Langchow", "Lower_Pacific", "Lucknow", "Luzon_Strait", "Madras", "Malaya", "Manchuria", "Mandalay", "Manila", "Mecca", "Mediterranean_Sea", "Middle_Pacific", "Mongolia", "Moscow", "Mysore", "Nagpur", "Nanchang", "Nepal", "New_Guinea", "North_Siam", "Odessa", "Okhotsk_Sea", "Oman", "Omsk", "Orenburg", "Otaru", "Peking", "Perm", "Persia", "Persian_Gulf", "Port_Arthur", "Punjab", "Rajputana", "Rangoon", "Red_Sea", "Rumania", "Sakhalin", "Sarawak", "Sea_of_Japan", "Semipalatinsk", "Seoul", "Seoul_east_coast", "Seoul_west_coast", "Shanghai", "Shiraz", "Singapore", "Sinkiang", "Somaliland", "South_China_Sea", "Southeast_Indian_Ocean", "Sudan", "Sulu_Sea", "Sumatra", "Sunda_Sea", "Syria", "Tabriz", "Tashkent", "Tibet", "Timor_Sea", "Tokyo", "Tongking", "Upper_Burma", "Upper_Pacific", "Urumchi", "Vladivostok", "West_Indian_Ocean", "Yellow_Sea", "Yunnan"]
        self.adjacency: Dict[str, List[str]] = {
            "Abyssinia": ["Eritrea", "Somaliland", "Sudan"],
            "Aden": ["Arabia", "Arabia_south_coast", "Gulf_of_Aden", "Mecca", "Red_Sea"],
            "Afghanistan": ["Bokhara", "Karachi", "Kashgar", "Kashmir", "Persia", "Punjab", "Tashkent"],
            "Akita": ["Kyoto", "Okhotsk_Sea", "Otaru", "Sea_of_Japan", "Tokyo"],
            "Akmolinsk": ["Kirghiz", "Krasnoyarsk", "Omsk", "Orenburg", "Semipalatinsk", "Tashkent"],
            "Andaman_Sea": ["Bangkok", "Bangkok_west_coast", "Bay_of_Bengal", "East_Indian_Ocean", "Gulf_of_Manaar", "Java_Sea", "Malaya", "Rangoon", "Sumatra"],
            "Angora": ["Armenia", "Black_Sea", "Constantinople", "Mediterranean_Sea", "Syria"],
            "Annam": ["Cambodia", "Cochin", "Gulf_of_Siam", "South_China_Sea", "Tongking"],
            "Arabia": ["Aden", "Arabian_Sea", "Baghdad", "Gulf_of_Aden", "Mecca", "Oman", "Persian_Gulf", "Syria"],
            "Arabia_north_coast": ["Baghdad", "Oman", "Persian_Gulf"],
            "Arabia_south_coast": ["Aden", "Arabian_Sea", "Gulf_of_Aden", "Oman"],
            "Arabian_Sea": ["Arabia", "Arabia_south_coast", "Bombay", "Gulf_of_Aden", "Karachi", "Mysore", "Oman", "Persian_Gulf", "Rajputana", "West_Indian_Ocean"],
            "Armenia": ["Angora", "Baghdad", "Baku", "Black_Sea", "Syria", "Tabriz"],
            "Assam": ["Bengal", "Sinkiang", "Tibet", "Upper_Burma", "Yunnan"],
            "Baghdad": ["Arabia", "Arabia_north_coast", "Armenia", "Persian_Gulf", "Shiraz", "Syria", "Tabriz"],
            "Baku": ["Armenia", "Black_Sea", "Moscow", "Odessa", "Tabriz"],
            "Bangkok": ["Andaman_Sea", "Cambodia", "Gulf_of_Siam", "Malaya", "North_Siam", "Rangoon"],
            "Bangkok_east_coast": ["Cambodia", "Gulf_of_Siam", "Malaya"],
            "Bangkok_west_coast": ["Andaman_Sea", "Malaya", "Rangoon"],
            "Bay_of_Bengal": ["Andaman_Sea", "Bengal", "Gulf_of_Manaar", "Hyderabad", "Rangoon", "Upper_Burma"],
            "Bengal": ["Assam", "Bay_of_Bengal", "Hyderabad", "Lucknow", "Nepal", "Tibet", "Upper_Burma"],
            "Black_Sea": ["Angora", "Armenia", "Baku", "Constantinople", "Mediterranean_Sea", "Odessa", "Rumania"],
            "Bokhara": ["Afghanistan", "Moscow", "Orenburg", "Persia", "Tashkent"],
            "Bombay": ["Arabian_Sea", "Hyderabad", "Mysore", "Nagpur", "Rajputana"],
            "Borneo": ["Celebes_Sea", "Java_Sea", "Sarawak"],
            "Cambodia": ["Annam", "Bangkok", "Bangkok_east_coast", "Cochin", "Gulf_of_Siam", "North_Siam", "Tongking"],
            "Canton": ["Chungking", "Hong_Kong", "Mandalay", "Nanchang", "South_China_Sea", "Tongking", "Yunnan"],
            "Cebu": ["Davao", "Lower_Pacific", "Luzon_Strait", "Manila", "Middle_Pacific", "Sulu_Sea"],
            "Celebes": ["Celebes_Sea", "Java_Sea", "Timor_Sea"],
            "Celebes_Sea": ["Borneo", "Celebes", "Davao", "Java_Sea", "Lower_Pacific", "New_Guinea", "Sarawak", "Sulu_Sea", "Timor_Sea"],
            "Ceylon": ["East_Indian_Ocean", "Gulf_of_Manaar", "West_Indian_Ocean"],
            "Chungking": ["Canton", "Langchow", "Nanchang", "Sinkiang", "Yunnan"],
            "Cochin": ["Annam", "Cambodia", "Gulf_of_Siam"],
            "Constantinople": ["Angora", "Black_Sea", "Mediterranean_Sea", "Rumania"],
            "Davao": ["Cebu", "Celebes_Sea", "Lower_Pacific", "Sulu_Sea"],
            "Delhi": ["Lucknow", "Nagpur", "Nepal", "Punjab", "Rajputana"],
            "East_China_Sea": ["Formosa", "Kyushu", "Nanchang", "Shanghai", "South_China_Sea", "Upper_Pacific", "Yellow_Sea"],
            "East_Indian_Ocean": ["Andaman_Sea", "Ceylon", "Gulf_of_Manaar", "Southeast_Indian_Ocean", "Sumatra", "West_Indian_Ocean"],
            "Egypt": ["Mecca", "Mediterranean_Sea", "Red_Sea", "Sudan", "Syria"],
            "Eritrea": ["Abyssinia", "Gulf_of_Aden", "Red_Sea", "Somaliland", "Sudan"],
            "Formosa": ["East_China_Sea", "Luzon_Strait", "Middle_Pacific", "South_China_Sea", "Upper_Pacific"],
            "Fusan": ["Sea_of_Japan", "Seoul", "Seoul_east_coast", "Seoul_west_coast", "Yellow_Sea"],
            "Gulf_of_Aden": ["Aden", "Arabia", "Arabia_south_coast", "Arabian_Sea", "Eritrea", "Red_Sea", "Somaliland", "West_Indian_Ocean"],
            "Gulf_of_Manaar": ["Andaman_Sea", "Bay_of_Bengal", "Ceylon", "East_Indian_Ocean", "Hyderabad", "Madras", "West_Indian_Ocean"],
            "Gulf_of_Siam": ["Annam", "Bangkok", "Bangkok_east_coast", "Cambodia", "Cochin", "Malaya", "South_China_Sea", "Sunda_Sea"],
            "Hong_Kong": ["Canton", "South_China_Sea"],
            "Hyderabad": ["Bay_of_Bengal", "Bengal", "Bombay", "Gulf_of_Manaar", "Lucknow", "Madras", "Mysore", "Nagpur"],
            "Irkutsk": ["Krasnoyarsk", "Manchuria", "Mongolia", "Vladivostok"],
            "Java": ["Java_Sea", "Southeast_Indian_Ocean", "Timor_Sea"],
            "Java_Sea": ["Andaman_Sea", "Borneo", "Celebes", "Celebes_Sea", "Java", "Malaya", "Sarawak", "Singapore", "Southeast_Indian_Ocean", "Sumatra", "Sunda_Sea", "Timor_Sea"],
            "Karachi": ["Afghanistan", "Arabian_Sea", "Persia", "Persian_Gulf", "Punjab", "Rajputana"],
            "Kashgar": ["Afghanistan", "Kashmir", "Kirghiz", "Sinkiang", "Tashkent", "Tibet", "Urumchi"],
            "Kashmir": ["Afghanistan", "Kashgar", "Punjab", "Tibet"],
            "Kirghiz": ["Akmolinsk", "Kashgar", "Semipalatinsk", "Tashkent", "Urumchi"],
            "Krasnoyarsk": ["Akmolinsk", "Irkutsk", "Mongolia", "Omsk", "Semipalatinsk", "Urumchi"],
            "Kyoto": ["Akita", "Kyushu", "Sea_of_Japan", "Tokyo", "Upper_Pacific", "Yellow_Sea"],
            "Kyushu": ["East_China_Sea", "Kyoto", "Upper_Pacific", "Yellow_Sea"],
            "Langchow": ["Chungking", "Mongolia", "Nanchang", "Peking", "Shanghai", "Sinkiang"],
            "Lower_Pacific": ["Cebu", "Celebes_Sea", "Davao", "Middle_Pacific", "New_Guinea"],
            "Lucknow": ["Bengal", "Delhi", "Hyderabad", "Nagpur", "Nepal"],
            "Luzon_Strait": ["Cebu", "Formosa", "Manila", "Middle_Pacific", "South_China_Sea", "Sulu_Sea"],
            "Madras": ["Gulf_of_Manaar", "Hyderabad", "Mysore", "West_Indian_Ocean"],
            "Malaya": ["Andaman_Sea", "Bangkok", "Bangkok_east_coast", "Bangkok_west_coast", "Gulf_of_Siam", "Java_Sea", "Singapore", "Sunda_Sea"],
            "Manchuria": ["Irkutsk", "Mongolia", "Peking", "Port_Arthur", "Seoul", "Shanghai", "Vladivostok", "Yellow_Sea"],
            "Mandalay": ["Canton", "North_Siam", "Rangoon", "Tongking", "Upper_Burma", "Yunnan"],
            "Manila": ["Cebu", "Luzon_Strait", "Middle_Pacific"],
            "Mecca": ["Aden", "Arabia", "Egypt", "Red_Sea", "Syria"],
            "Mediterranean_Sea": ["Angora", "Black_Sea", "Constantinople", "Egypt", "Syria"],
            "Middle_Pacific": ["Cebu", "Formosa", "Lower_Pacific", "Luzon_Strait", "Manila", "Upper_Pacific"],
            "Mongolia": ["Irkutsk", "Krasnoyarsk", "Langchow", "Manchuria", "Peking", "Sinkiang", "Urumchi"],
            "Moscow": ["Baku", "Bokhara", "Odessa", "Orenburg", "Perm"],
            "Mysore": ["Arabian_Sea", "Bombay", "Hyderabad", "Madras", "West_Indian_Ocean"],
            "Nagpur": ["Bombay", "Delhi", "Hyderabad", "Lucknow", "Rajputana"],
            "Nanchang": ["Canton", "Chungking", "East_China_Sea", "Langchow", "Shanghai", "South_China_Sea"],
            "Nepal": ["Bengal", "Delhi", "Lucknow", "Punjab"],
            "New_Guinea": ["Celebes_Sea", "Lower_Pacific", "Timor_Sea"],
            "North_Siam": ["Bangkok", "Cambodia", "Mandalay", "Rangoon", "Tongking"],
            "Odessa": ["Baku", "Black_Sea", "Moscow", "Rumania"],
            "Okhotsk_Sea": ["Akita", "Otaru", "Sakhalin", "Sea_of_Japan", "Tokyo", "Upper_Pacific", "Vladivostok"],
            "Oman": ["Arabia", "Arabia_north_coast", "Arabia_south_coast", "Arabian_Sea", "Persian_Gulf"],
            "Omsk": ["Akmolinsk", "Krasnoyarsk", "Orenburg", "Perm"],
            "Orenburg": ["Akmolinsk", "Bokhara", "Moscow", "Omsk", "Perm", "Tashkent"],
            "Otaru": ["Akita", "Okhotsk_Sea", "Sakhalin", "Sea_of_Japan"],
            "Peking": ["Langchow", "Manchuria", "Mongolia", "Shanghai"],
            "Perm": ["Moscow", "Omsk", "Orenburg"],
            "Persia": ["Afghanistan", "Bokhara", "Karachi", "Persian_Gulf", "Shiraz", "Tabriz"],
            "Persian_Gulf": ["Arabia", "Arabia_north_coast", "Arabian_Sea", "Baghdad", "Karachi", "Oman", "Persia", "Shiraz"],
            "Port_Arthur": ["Manchuria", "Seoul", "Seoul_west_coast", "Yellow_Sea"],
            "Punjab": ["Afghanistan", "Delhi", "Karachi", "Kashmir", "Nepal", "Rajputana", "Tibet"],
            "Rajputana": ["Arabian_Sea", "Bombay", "Delhi", "Karachi", "Nagpur", "Punjab"],
            "Rangoon": ["Andaman_Sea", "Bangkok", "Bangkok_west_coast", "Bay_of_Bengal", "Mandalay", "North_Siam", "Upper_Burma"],
            "Red_Sea": ["Aden", "Egypt", "Gulf_of_Aden", "Mecca", "Sudan"],
            "Rumania": ["Black_Sea", "Constantinople", "Odessa"],
            "Sakhalin": ["Okhotsk_Sea", "Otaru"],
            "Sarawak": ["Borneo", "Celebes_Sea", "Java_Sea", "Sulu_Sea", "Sunda_Sea"],
            "Sea_of_Japan": ["Akita", "Fusan", "Kyoto", "Okhotsk_Sea", "Otaru", "Seoul", "Seoul_east_coast", "Vladivostok", "Yellow_Sea"],
            "Semipalatinsk": ["Akmolinsk", "Kirghiz", "Krasnoyarsk", "Urumchi"],
            "Seoul": ["Fusan", "Manchuria", "Port_Arthur", "Sea_of_Japan", "Vladivostok", "Yellow_Sea"],
            "Seoul_east_coast": ["Fusan", "Sea_of_Japan", "Vladivostok"],
            "Seoul_west_coast": ["Fusan", "Port_Arthur", "Yellow_Sea"],
            "Shanghai": ["East_China_Sea", "Langchow", "Manchuria", "Nanchang", "Peking", "Yellow_Sea"],
            "Shiraz": ["Baghdad", "Persia", "Persian_Gulf", "Tabriz"],
            "Singapore": ["Java_Sea", "Malaya"],
            "Sinkiang": ["Assam", "Chungking", "Kashgar", "Langchow", "Mongolia", "Tibet", "Urumchi", "Yunnan"],
            "Somaliland": ["Abyssinia", "Eritrea", "Gulf_of_Aden"],
            "South_China_Sea": ["Annam", "Canton", "East_China_Sea", "Formosa", "Gulf_of_Siam", "Hong_Kong", "Luzon_Strait", "Nanchang", "Sulu_Sea", "Sunda_Sea", "Tongking"],
            "Southeast_Indian_Ocean": ["East_Indian_Ocean", "Java", "Java_Sea", "Sumatra", "Timor_Sea", "West_Indian_Ocean"],
            "Sudan": ["Abyssinia", "Egypt", "Eritrea", "Red_Sea"],
            "Sulu_Sea": ["Cebu", "Celebes_Sea", "Davao", "Luzon_Strait", "Sarawak", "South_China_Sea", "Sunda_Sea"],
            "Sumatra": ["Andaman_Sea", "East_Indian_Ocean", "Java_Sea", "Southeast_Indian_Ocean"],
            "Sunda_Sea": ["Gulf_of_Siam", "Java_Sea", "Malaya", "Sarawak", "South_China_Sea", "Sulu_Sea"],
            "Syria": ["Angora", "Arabia", "Armenia", "Baghdad", "Egypt", "Mecca", "Mediterranean_Sea"],
            "Tabriz": ["Armenia", "Baghdad", "Baku", "Persia", "Shiraz"],
            "Tashkent": ["Afghanistan", "Akmolinsk", "Bokhara", "Kashgar", "Kirghiz", "Orenburg"],
            "Tibet": ["Assam", "Bengal", "Kashgar", "Kashmir", "Punjab", "Sinkiang"],
            "Timor_Sea": ["Celebes", "Celebes_Sea", "Java", "Java_Sea", "New_Guinea", "Southeast_Indian_Ocean"],
            "Tokyo": ["Akita", "Kyoto", "Okhotsk_Sea", "Upper_Pacific"],
            "Tongking": ["Annam", "Cambodia", "Canton", "Mandalay", "North_Siam"],
            "Upper_Burma": ["Assam", "Bay_of_Bengal", "Bengal", "Mandalay", "Rangoon", "Yunnan"],
            "Upper_Pacific": ["East_China_Sea", "Formosa", "Kyoto", "Kyushu", "Middle_Pacific", "Okhotsk_Sea", "Tokyo", "Yellow_Sea"],
            "Urumchi": ["Kashgar", "Kirghiz", "Krasnoyarsk", "Mongolia", "Semipalatinsk", "Sinkiang"],
            "Vladivostok": ["Irkutsk", "Manchuria", "Okhotsk_Sea", "Sea_of_Japan", "Seoul", "Seoul_east_coast"],
            "West_Indian_Ocean": ["Arabian_Sea", "Ceylon", "East_Indian_Ocean", "Gulf_of_Aden", "Gulf_of_Manaar", "Madras", "Mysore", "Southeast_Indian_Ocean"],
            "Yellow_Sea": ["East_China_Sea", "Fusan", "Kyoto", "Kyushu", "Manchuria", "Port_Arthur", "Sea_of_Japan", "Seoul", "Seoul_west_coast", "Shanghai", "Upper_Pacific"],
            "Yunnan": ["Assam", "Canton", "Chungking", "Mandalay", "Sinkiang", "Upper_Burma"]
        }

        # Units: {player_id: [(type, province, strength), ...]}
        self.units: Dict[int, List[Dict[str, object]]] = {
            # Britain
            1: [
                {"type": "Army", "location": "Delhi", "strength": 1},
                {"type": "Army", "location": "Madras", "strength": 1},
                {"type": "Fleet", "location": "Bombay", "strength": 1},
                {"type": "Fleet", "location": "Hong_Kong", "strength": 1},
                {"type": "Fleet", "location": "Aden", "strength": 1},
                {"type": "Fleet", "location": "Singapore", "strength": 1}
            ],
            # China
            2: [
                {"type": "Army", "location": "Peking", "strength": 1},
                {"type": "Army", "location": "Shanghai", "strength": 1},
                {"type": "Army", "location": "Canton", "strength": 1},
                {"type": "Army", "location": "Manchuria", "strength": 1},
                {"type": "Army", "location": "Sinkiang", "strength": 1}
            ],
            # France
            3: [
                {"type": "Army", "location": "Tongking", "strength": 1},
                {"type": "Fleet", "location": "Annam", "strength": 1},
                {"type": "Army", "location": "Cochin", "strength": 1}
            ],
            # Holland
            4: [
                {"type": "Army", "location": "Borneo", "strength": 1},
                {"type": "Fleet", "location": "Java", "strength": 1},
                {"type": "Fleet", "location": "Sumatra", "strength": 1}
            ],
            # Japan
            5: [
                {"type": "Fleet", "location": "Tokyo", "strength": 1},
                {"type": "Fleet", "location": "Otaru", "strength": 1},
                {"type": "Fleet", "location": "Kyushu", "strength": 1},
                {"type": "Army", "location": "Kyoto", "strength": 1}
            ],
            # Russia
            6: [
                {"type": "Army", "location": "Moscow", "strength": 1},
                {"type": "Army", "location": "Omsk", "strength": 1},
                {"type": "Army", "location": "Vladivostok", "strength": 1},
                {"type": "Fleet", "location": "Odessa", "strength": 1},
                {"type": "Fleet", "location": "Port_Arthur", "strength": 1}
            ],
            # Turkey
            7: [
                {"type": "Army", "location": "Angora", "strength": 1},
                {"type": "Fleet", "location": "Baghdad", "strength": 1},
                {"type": "Fleet", "location": "Constantinople", "strength": 1}
            ]
        }

        # Supply center ownership: {province: player_id}
        self.supply_centers: Dict[str, int] = {
            # Britain
            "Delhi": 1, "Madras": 1, "Bombay": 1, "Hong_Kong": 1, "Aden": 1, "Singapore": 1,
            # China
            "Peking": 2, "Shanghai": 2, "Canton": 2, "Manchuria": 2, "Sinkiang": 2,
            # France
            "Tongking": 3, "Annam": 3, "Cochin": 3,
            # Holland
            "Borneo": 4, "Java": 4, "Sumatra": 4,
            # Japan
            "Tokyo": 5, "Otaru": 5, "Kyushu": 5, "Kyoto": 5,
            # Russia
            "Moscow": 6, "Omsk": 6, "Vladivostok": 6, "Odessa": 6, "Port_Arthur": 6,
            # Turkey
            "Angora": 7, "Baghdad": 7, "Constantinople": 7,
            # Neutral
            "Assam": 0, "Bangkok": 0, "Bangkok_east_coast": 0, "Bangkok_west_coast": 0,
            "Bengal": 0, "Cebu": 0, "Ceylon": 0, "Chungking": 0,
            "Davao": 0, "Egypt": 0, "Formosa": 0, "Fusan": 0,
            "Karachi": 0, "Kashgar": 0, "Kashmir": 0, "Malaya": 0,
            "Mandalay": 0, "Manila": 0, "Mongolia": 0, "New_Guinea": 0,
            "Persia": 0, "Rangoon": 0, "Rumania": 0, "Sakhalin": 0,
            "Sarawak": 0, "Seoul": 0, "Seoul_east_coast": 0, "Seoul_west_coast": 0,
            "Shiraz": 0, "Sudan": 0, "Tabriz": 0, "Tashkent": 0, "Upper_Burma": 0
        }

        self.supply_center_groups = {
            "Bangkok": ["Bangkok", "Bangkok_east_coast", "Bangkok_west_coast"],
            "Seoul": ["Seoul", "Seoul_east_coast", "Seoul_west_coast"]
            }

        self.home_supply_centers: Dict[str, int] = {
            # Britain
            "Delhi": 1, "Madras": 1, "Bombay": 1, "Hong_Kong": 1, "Aden": 1, "Singapore": 1,
            # China
            "Peking": 2, "Shanghai": 2, "Canton": 2, "Manchuria": 2, "Sinkiang": 2,
            # France
            "Tongking": 3, "Annam": 3, "Cochin": 3,
            # Holland
            "Borneo": 4, "Java": 4, "Sumatra": 4,
            # Japan
            "Tokyo": 5, "Otaru": 5, "Kyushu": 5, "Kyoto": 5,
            # Russia
            "Moscow": 6, "Omsk": 6, "Vladivostok": 6, "Odessa": 6, "Port_Arthur": 6,
            # Turkey
            "Angora": 7, "Baghdad": 7, "Constantinople": 7
        }

        self.land_provinces: Dict[str, int] = {
            # Britain
            "Delhi": 1, "Kashmir": 1, "Lucknow": 1, "Nagpur": 1, "Punjab": 1,
            # China
            "Chungking": 2, "Kashgar": 2, "Langchow": 2, "Mongolia": 2, "Peking": 2, "Sinkiang": 2, "Tibet": 2, "Urumchi": 2, "Yunnan": 2,
            # Russia
            "Akmolinsk": 6, "Bokhara": 6, "Irkutsk": 6, "Kirghiz": 6, "Krasnoyarsk": 6, "Moscow": 6, "Omsk": 6, "Orenburg": 6, "Perm": 6, "Semipalatinsk": 6, "Tashkent": 6,
            # Neutral
            "Abyssinia": 0, "Afghanistan": 0, "Assam": 0, "Mandalay": 0, "Nepal": 0, "North_Siam": 0, "Tabriz": 0
        }

        self.water_provinces: Dict[str, int] = {
            "Andaman_Sea": 0, "Arabian_Sea": 0, "Bay_of_Bengal": 0, "Black_Sea": 0,
            "Celebes_Sea": 0, "East_China_Sea": 0, "East_Indian_Ocean": 0, "Gulf_of_Aden": 0,
            "Gulf_of_Manaar": 0, "Gulf_of_Siam": 0, "Java_Sea": 0, "Lower_Pacific": 0,
            "Luzon_Strait": 0, "Mediterranean_Sea": 0, "Middle_Pacific": 0, "Okhotsk_Sea": 0,
            "Persian_Gulf": 0, "Red_Sea": 0, "Sea_of_Japan": 0, "South_China_Sea": 0,
            "Southeast_Indian_Ocean": 0, "Sulu_Sea": 0, "Sunda_Sea": 0, "Timor_Sea": 0,
            "Upper_Pacific": 0, "West_Indian_Ocean": 0, "Yellow_Sea": 0
        }

        self.coast_provinces: Dict[str, int] = {
            # Britain
            "Aden": 1, "Bengal": 1, "Bombay": 1, "Ceylon": 1, "Hong_Kong": 1, "Hyderabad": 1, "Madras": 1, "Mysore": 1, "Rajputana": 1, "Singapore": 1,
            # China
            "Canton": 2, "Manchuria": 2, "Nanchang": 2, "Shanghai": 2,
            # France
            "Annam": 3, "Cambodia": 3, "Cochin": 3, "Tongking": 3,
            # Holland
            "Borneo": 4, "Celebes": 4, "Java": 4, "Sumatra": 4,
            # Japan
            "Akita": 5, "Kyoto": 5, "Kyushu": 5, "Otaru": 5, "Tokyo": 5,
            # Russia
            "Baku": 6, "Odessa": 6, "Port_Arthur": 6, "Sakhalin": 6, "Vladivostok": 6,
            # Turkey
            "Angora": 7, "Armenia": 7, "Baghdad": 7, "Constantinople": 7, "Mecca": 7, "Syria": 7,
            # Neutral
            "Arabia": 0, "Arabia_north_coast": 0, "Arabia_south_coast": 0,
            "Bangkok": 0, "Bangkok_east_coast": 0, "Bangkok_west_coast": 0,
            "Cebu": 0, "Davao": 0, "Egypt": 0, "Eritrea": 0, "Formosa": 0,
            "Fusan": 0, "Karachi": 0, "Malaya": 0, "Manila": 0, "New_Guinea": 0,
            "Oman": 0, "Persia": 0, "Rangoon": 0, "Rumania": 0, "Sarawak": 0,
            "Seoul": 0, "Seoul_east_coast": 0, "Seoul_west_coast": 0,
            "Shiraz": 0, "Somaliland": 0, "Sudan": 0, "Upper_Burma": 0
        }

        self.trans_siberian_railroad: Dict[str, int] = {
            "Irkutsk": 6, "Krasnoyarsk": 6, "Moscow": 6, "Omsk": 6, "Perm": 6, "Vladivostok": 6
        }

        self.tsr_path = ["Moscow", "Perm", "Omsk", "Krasnoyarsk", "Irkutsk", "Vladivostok"]

    def move_unit(self, player_id: int, unit_index: int, destination: str) -> bool:
        """
        Attempts to move a unit to a destination province.
        Returns True if the move succeeds, False otherwise.
        """

        # Basic validation
        if player_id not in self.units:
            return False
        
        if unit_index < 0 or unit_index >= len(self.units[player_id]):
            return False
        
        unit = self.units[player_id][unit_index]
        unit_type = unit["type"]
        origin = unit["location"]

        # Province must exist
        if destination not in self.provinces:
            return False
        
        # Adjacency check
        if origin not in self.adjacency:
            return False
        
        if destination not in self.adjacency.get(origin, []):
            return False
        
        # Terrain compatibility
        if unit_type == "Army":
            if destination in self.water_provinces:
                return False
            
        if unit_type == "Fleet":
            if destination in self.land_provinces:
                return False
            
        # Execute move
        unit["location"] = destination
        return True

    def hold_unit(self, player_id: int, unit_index: int) -> bool:
        """
        Orders a unit to hold its current province.
        Returns True if the order is valid.
        """

        # Validation
        if player_id not in self.units:
            return False
        
        if unit_index < 0 or unit_index >= len(self.units[player_id]):
            return False
        
        unit = self.units[player_id][unit_index]

        # Execute hold
        return True

    def support_unit(self, supporter_pid: int, supporter_idx: int, supported_pid: int, supported_idx: int, supported_destination: str | None) -> bool:
        """
        Allows one unit (supporter) to support another unit (supported).
        Adds +1 strength to the supported unit if legal.

        supported destination:
        - None -> support holding
        - str -> support movement to destination
        """

        # Validate player and unit IDs
        if supporter_pid not in self.units or supported_pid not in self.units:
            return False
        
        if supporter_idx < 0 or supporter_idx >= len(self.units[supporter_pid]):
            return False
        
        if supported_idx < 0 or supported_idx >= len(self.units[supported_pid]):
            return False
        
        supporter = self.units[supporter_pid][supporter_idx]
        supported = self.units[supported_pid][supported_idx]

        supporter_type = supporter["type"]
        supporter_loc = supporter["location"]

        supported_loc = supported["location"]

        # Determine target province
        if supported_destination is None:
            # Support holding
            target_province = supported_loc
        else:
            # Support movement
            target_province = supported_destination

        # Province must exist
        if target_province not in self.provinces:
            return False
        
        # Adjacency check
        if supporter_loc not in self.adjacency:
            return False
        
        if target_province not in self.adjacency.get(supporter_loc, []):
            return False
        
        # Movement capability check
        if supporter_type == "Army":
            if target_province in self.water_provinces:
                return False
            
        if supporter_type == "Fleet":
            if target_province in self.land_provinces:
                return False
            
        # Apply support
        supported["strength"] += 1
        return True
        
    def convoy_army(self, army_pid: int, army_idx: int, destination: str) -> bool:
        """
        Allows an Army to move via convoy from one coast province to another
        through a connected chain of Fleets in water provinces.
        Returns True if the convoy succeeds.
        """

        # Validate unit
        if army_pid not in self.units:
            return False
        
        if army_idx < 0 or army_idx >= len(self.units[army_pid]):
            return False
        
        army = self.units[army_pid][army_idx]

        if army["type"] != "Army":
            return False
        
        origin = army["location"]

        # Coast checks
        if origin not in self.coast_provinces:
            return False
        
        if destination not in self.coast_provinces:
            return False
        
        # Province must exist
        if destination not in self.provinces:
            return False
        
        # Build set of water provinces with fleets
        fleet_water_provinces = set()

        for units in self.units.values():
            for unit in units:
                if unit["type"] == "Fleet":
                    loc = unit["location"]
                    if loc in self.water_provinces:
                        fleet_water_provinces.add(loc)

        if not fleet_water_provinces:
            return False
        
        # Breadth-first search through fleets
        visited = set()
        queue = deque()

        # Start from water provinces to origin coast
        for adj in self.adjacency.get(origin, []):
            if adj in fleet_water_provinces:
                queue.append(adj)
                visited.add(adj)

        while queue:
            current = queue.popleft()

            # If this fleet borders the destination coast, convoy succeeds
            if destination in self.adjacency.get(current, []):
                army["location"] = destination
                return True

            # Continue searching fleet chain
            for adj in self.adjacency.get(current, []):
                if adj in fleet_water_provinces and adj not in visited:
                    visited.add(adj)
                    queue.append(adj)

        return False

    def get_unit_at(self, province: str):
        for pid, unit_list in self.units.items():
            for u in unit_list:
                if u["location"] == province:
                    return pid, u
        return None, None

    def can_unit_move_to(self, unit_type: str, target: str) -> bool:
        if unit_type == "Army":
            return target in self.land_provinces or target in self.coast_provinces
        elif unit_type == "Fleet":
            return target in self.water_provinces or target in self.coast_provinces
        return False
        
    def get_legal_retreat_locations(self, unit: Dict[str, object], attacker_origin: str, standoff_provinces: set) -> List[str]:
        """
        Returns all legal retreat provinces for a dislodged unit.
        """
        location = unit["location"]
        unit_type = unit["type"]

        # Adjacent provinces
        neighbors = self.adjacency.get(location, [])

        # Build set of currently occupied provinces
        occupied = set()
        for _, units in self.units.items():
            for u in units:
                occupied.add(u["location"])
        
        legal_retreats = []

        for province in neighbors:
            # Must be a place unit type could normally move
            if not self.can_unit_move_to(unit_type, province):
                continue

            # Cannot retreat into occupied province
            if province in occupied:
                continue

            # Cannot retreat into attacker origin
            if province == attacker_origin:
                continue

            # Cannot retreat into standoff province
            if province in standoff_provinces:
                continue

            legal_retreats.append(province)

        return legal_retreats
        
    def retreat_unit(self, player_id: int, unit_location: str, attacker_origin: str, standoff_provinces: set, chosen_retreat: str | None) -> bool:
        """
        Handles retreating of a dislodged unit.
        If chosen_retreat is None or illegal, and no legal options exist,
        the unit is automatically disbanded.
        """

        # Find the unit
        player_units = self.units.get(player_id, [])
        unit = next((u for u in player_units if u["location"] == unit_location), None)

        # Find index instead of passing location
        unit_idx = next((i for i, u in enumerate(player_units) if u["location"] == unit_location), None)

        if unit_idx is None:
            return False

        legal = self.get_legal_retreat_locations(unit, attacker_origin, standoff_provinces)

        # No retreats available
        if not legal:
            self.disband_unit(player_id, unit_idx) # changed from unit_location
            return False

        # Declined retreat
        if chosen_retreat is None:
            self.disband_unit(player_id, unit_idx) # changed from unit_location
            return False

        # Illegal retreat
        if chosen_retreat not in legal:
            self.disband_unit(player_id, unit_idx) # changed from unit_location
            return False
        
        # Perform retreat
        unit["location"] = chosen_retreat
        return True
    
    def disband_unit(self, player_id: int, unit_idx: int) -> bool:
        """
        Removes a unit from the game.
        Returns True if the unit was successfully disbanded.
        """

        # Validate player
        if player_id not in self.units:
            return False
        
        # Validate unit index
        if unit_idx < 0 or unit_idx >= len(self.units[player_id]):
            return False
        
        # Remove the unit
        self.units[player_id].pop(unit_idx)

        # Remove eliminated player
        if len(self.units[player_id]) == 0:
            del self.units[player_id]

        return True

    def is_province_occupied(self, province: str) -> bool:
        for units in self.units.values():
            for unit in units:
                if unit["location"] == province:
                    return True
        return False

    def build_unit(self, player_id: int, province: str, unit_type: str) -> bool:
        """
        Attempts to build a unit for player_id at province.
        Returns True if build succeeds.
        """

        # Phase check
        if self.phase != "Build":
            return False
        
        # Home supply center ownership check
        if province not in self.home_supply_centers:
            return False
        
        if self.home_supply_centers[province] != player_id:
            return False
        
        # Province occupancy check
        if self.is_province_occupied(province):
            return False
        
        # Unit entitlement check
        supply_count = sum(1 for p, owner in self.supply_centers.items() if owner == player_id)

        unit_count = len(self.units.get(player_id, []))

        if unit_count >= supply_count:
            return False # No builds allowed
        
        # Unit type validation
        if unit_type == "Army":
            if province not in self.land_provinces and province not in self.coast_provinces:
                return False
        
        elif unit_type == "Fleet":
            if province not in self.coast_provinces:
                return False
            
        else:
            return False # Invalid unit type
        
        # Build the unit
        new_unit = {
            "type": unit_type,
            "location": province,
            "strength": 1
        }

        self.units[player_id].append(new_unit)

        return True

    def use_trans_siberian_railroad(self, player_id: int, start_province: str, chosen_destination: str, standoff_provinces: set | None = None):
        """
        Executes a Trans-Siberian Railroad move.

        Rules enforced:
        - Only Russia (player 6)
        - Only once per turn
        - Unit must be an Army
        - Must be on TSR line
        - Must travel only along TSR line
        - Can pass through Russian units
        - Must stop before foreign occupied TSR provinces
        - Cannot end in occupied province
        - Must stop before standoff province
        """

        standoff_provinces = standoff_provinces or set()

        if player_id != 6:
            raise ValueError("Only Russia may use the Trans-Siberian Railroad.")
        
        if self.tsr_used_this_turn:
            raise ValueError("TSR may only be used once per turn.")
        
        # Find the unit
        unit_list = self.units.get(player_id, [])
        unit = next((u for u in unit_list if u["location"] == start_province), None)

        if unit is None:
            raise ValueError("No Russian unit found  in starting TSR province.")
        
        if unit["type"] != "Army":
            raise ValueError("Only Army units may use the TSR.")
        
        # Ensure start is on TSR
        if start_province not in self.tsr_path:
            raise ValueError("Unit is not on the Trans-Siberian Railroad line.")
        
        # Ensure destination is on TSR
        if chosen_destination not in self.tsr_path:
            raise ValueError("Destination is not on the TSR line.")
        
        # Determine direction of travel along TSR
        start_index = self.tsr_path.index(start_province)
        dest_index = self.tsr_path.index(chosen_destination)

        if start_index == dest_index:
            raise ValueError("Unit must move somewhere along the TSR.")
        
        # Determine traversal direction
        step = 1 if dest_index > start_index else -1
        path_segment = self.tsr_path[start_index + step : dest_index + step: step]

        # Evaluate legality along path
        final_stop = None

        for province in path_segment:
            pid, _ = self.get_unit_at(province)

            # Standoff rule -> must stop before this province
            if province in standoff_provinces:
                break

            # Foreign army blocks TSR beyond it
            if pid not in (None, 6):
                break

            # If empty, it may be a valid stopping place
            if pid is None:
                final_stop = province
            
            # If Russian unit may pass through but cannot end there, continue
            continue

        # If no legal empty stopping province
        if final_stop is None:
            raise ValueError("No valid TSR destination available.")
        
        # Must match chosen destination
        if final_stop != chosen_destination:
            raise ValueError(f"Illegal TSR order. Best legal destination is '{final_stop}', "f"but '{chosen_destination}' was ordered.")
        
        # Move the unit
        unit["location"] = final_stop

        # TSR consumed for this turn
        self.tsr_used_this_turn = True
        
        return final_stop
    
    def get_controller_of_egypt(self):
        """
        Returns player_id controlling Egypt by occupation.
        Must be physically occupied by that power.
        Returns None if Egypt not occupied.
        """

        for pid, units in self.units.items():
            for unit in units:
                if unit["location"] == "Egypt":
                    return pid
        return None
        
    def grant_suez_permission(self, controller_id: int, foreign_player: int):
        if controller_id not in self.suez_permissions:
            self.suez_permissions[controller_id] = set()
        self.suez_permissions[controller_id].add(foreign_player)

    def can_use_suez_canal(self, player_id: int, start: str, destination: str, egypt_dislodged: bool = False):
        """
        Determines if a given Fleet order across the Suez Canal is legal.

        start must be Mediterranean Sea or Red Sea
        destination must be other side
        """

        # Must be a Fleet
        pid, unit = self.get_unit_at(start)
        if pid != player_id or unit is None or unit["type"] != "Fleet":
            return False

        # Must be correct canal movement
        valid_pairs = {
            ("Mediterranean_Sea", "Red_Sea"), ("Red_Sea", "Mediterranean_Sea")
        }
        if (start, destination) not in valid_pairs:
            return False
        
        # Egypt must be controlled by occupation entire turn
        controller = self.get_controller_of_egypt()
        if controller is None:
            return False
        
        # If same power controls Egypt, always allowed
        if controller == player_id:
            return True
        
        # Otherwise, need explicit Suez Canal permission
        allowed = self.suez_permissions.get(controller, set())
        if player_id in allowed:
            return True
        
        return False
        
    def execute_suez_move(self, player_id: int, start: str, destination: str, egypt_dislodged: bool = False):
        if not self.can_use_suez_canal(player_id, start, destination, egypt_dislodged):
            raise ValueError("Suez Canal rule violation")
        
        # Normal conflict rules still determine success
        # This only sets allowed movement path
        unit_pid, unit = self.get_unit_at(start)
        unit["location"] = destination
        return True

    def get_canonical_supply_center(self, province: str) -> str:
        temp_province = province

        if "Bangkok" in temp_province:
            temp_province = "Bangkok"
        elif "Seoul" in temp_province:
            temp_province = "Seoul"
        return temp_province
    
    def update_supply_center_control(self):
        """
        Updates supply center ownership after a Fall turn.
        """
        if self.phase != "Fall":
            return
        
        # Track which supply centers are occupied and by whom
        occupied_centers: Dict[str, int] = {}

        for player_id, units in self.units.items():
            for unit in units:
                province = unit["location"]

                canonical = self.get_canonical_supply_center(province)

                if canonical in self.supply_centers:
                    occupied_centers[canonical] = player_id
                    
        # Apply ownership changes
        for sc in list(self.supply_centers.keys()):
            canonical = self.get_canonical_supply_center(sc)

            if canonical in occupied_centers:
                new_owner = occupied_centers[canonical]

                # Hong Kong special rule: China can never control Hong Kong
                if canonical == "Hong_Kong" and new_owner == 2:
                    continue

                self.supply_centers[canonical] = new_owner

    def count_supply_centers(self, player_id: int) -> int:
        return sum(1 for owner in self.supply_centers.values() if owner == player_id)
    
    def start_new_turn(self):
        self.tsr_used_this_turn = False
        self.suez_permissions = {}

    def check_victory(self) -> int | None:
        """
        Returns the winning player_id if someone has won, otherwise None.
        """
        for player_id in range(1, self.num_players + 1):
            if self.count_supply_centers(player_id) >= 30:
                return player_id
        return None
    
    def end_of_fall_phase(self):
        self.update_supply_center_control()

        winner = self.check_victory()
        if winner is not None:
            self.done = True
            self.winner = winner
            return
            
        self.phase = "Build"

    @staticmethod
    def assert_equal(actual, expected, message):
        if actual != expected:
            raise AssertionError(f"{message} | expected={expected}, got={actual}")

    @staticmethod
    def assert_true(condition, message):
        if not condition:
            raise AssertionError(message)
        
    @staticmethod
    def assert_raises(expected_exception, func, message):
        try:
            func()
        except expected_exception:
            return
        except Exception as e:
            raise AssertionError(f"{message} | wrong exception: {type(e)}")
        raise AssertionError(f"{message} | exception not raised")

    @staticmethod
    def test_move_unit(env):
        print("Testing move_unit()...")

        # Setup temporary player
        env.units[10] = [{"type": "Army", "location": "Persia", "strength": 1}, {"type": "Fleet", "location": "Shiraz", "strength": 1}]

        # Valid Army move to adjacent land
        result = env.move_unit(10, 0, "Tabriz")
        ColonialDiplomacyEnv.assert_true(result, "Army should move to adjacent land")
        ColonialDiplomacyEnv.assert_equal(env.units[10][0]["location"], "Tabriz", "Army location should update")

        # Valid Army move to adjacent coast
        result = env.move_unit(10, 0, "Baghdad")
        ColonialDiplomacyEnv.assert_true(result, "Army should move to adjacent land")
        ColonialDiplomacyEnv.assert_equal(env.units[10][0]["location"], "Baghdad", "Army location should update")

        # Invalid Army move to water
        result = env.move_unit(10, 0, "Persian_Gulf")
        ColonialDiplomacyEnv.assert_true(not result, "Army cannot move to water")
        ColonialDiplomacyEnv.assert_equal(env.units[10][0]["location"], "Baghdad", "Army location should not change")

        # Valid Fleet move to adjacent water
        result = env.move_unit(10, 1, "Persian_Gulf")
        ColonialDiplomacyEnv.assert_true(result, "Fleet should move to adjacent water")
        ColonialDiplomacyEnv.assert_equal(env.units[10][1]["location"], "Persian_Gulf", "Fleet location should update")

        # Valid Fleet move to adjacent coast
        result = env.move_unit(10, 1, "Karachi")
        ColonialDiplomacyEnv.assert_true(result, "Fleet should move to adjacent coast")
        ColonialDiplomacyEnv.assert_equal(env.units[10][1]["location"], "Karachi", "Fleet location should update")

        # Invalid Fleet move to land
        result = env.move_unit(10, 1, "Afghanistan")
        ColonialDiplomacyEnv.assert_true(not result, "Fleet cannot move to land")
        ColonialDiplomacyEnv.assert_equal(env.units[10][1]["location"], "Karachi", "Fleet location should not change")

        # Invalid move: non-adjacent province
        result = env.move_unit(10, 0, "Persia")
        ColonialDiplomacyEnv.assert_true(not result, "Unit cannot move to non-adjacent province")

        # Invalid move: nonexistent player
        result = env.move_unit(99, 0, "Tabriz")
        ColonialDiplomacyEnv.assert_true(not result, "Move fails for nonexistent player")

        # Invalid move: invalid unit index
        result = env.move_unit(10, 99, "Tabriz")
        ColonialDiplomacyEnv.assert_true(not result, "Move fails for invalid unit index")

        print("move_unit() tests passed\n")

    @staticmethod
    def test_hold_unit(env):
        print("Testing hold_unit()...")

        # Valid holds
        result = env.hold_unit(1, 0)
        ColonialDiplomacyEnv.assert_true(result, "Army hold should succeed")
        ColonialDiplomacyEnv.assert_equal(env.units[1][0]["location"], "Delhi", "Army location should remain the same")

        result = env.hold_unit(1, 2)
        ColonialDiplomacyEnv.assert_true(result, "Fleet hold should succeed")
        ColonialDiplomacyEnv.assert_equal(env.units[1][2]["location"], "Bombay", "Fleet location should remain the same")

        # Invalid hold: nonexistent player
        result = env.hold_unit(99, 0)
        ColonialDiplomacyEnv.assert_true(not result, "Hold fails for nonexistent player")

        # Invalid hold: invalid unit index
        result = env.hold_unit(1, 99)
        ColonialDiplomacyEnv.assert_true(not result, "Hold fails for invalid unit index")

        print("hold_unit() tests passed\n")

    @staticmethod
    def test_support_unit_hold(env):
        print("Testing support_unit() hold...")

        supported = env.units[3][2]
        original_strength = supported["strength"]

        success = env.support_unit(supporter_pid=3, supporter_idx=1, supported_pid=3, supported_idx=2, supported_destination=None)
        
        ColonialDiplomacyEnv.assert_true(success, "Support hold should succeed")
        ColonialDiplomacyEnv.assert_equal(supported["strength"], original_strength + 1, "Support hold did not increase strength")

        print("support_unit() hold tests passed\n")

    @staticmethod
    def test_support_unit_move(env):
        print("Testing support_unit() move...")

        supported = env.units[3][2]
        original_strength = supported["strength"]

        success = env.support_unit(supporter_pid=3, supporter_idx=0, supported_pid=3, supported_idx=2, supported_destination="Cambodia")

        ColonialDiplomacyEnv.assert_true(success, "Support move should succeed")
        ColonialDiplomacyEnv.assert_equal(supported["strength"], original_strength + 1, "Support move did not increase strength")

        print("support_unit() move tests passed\n")

    @staticmethod
    def test_support_unit_illegal_adjacency(env):
        print("Testing support_unit() with illegal adjacency...")

        supported = env.units[1][1]
        original_strength = supported["strength"]

        success = env.support_unit(supporter_pid=1, supporter_idx=0, supported_pid=1, supported_idx=1, supported_destination="Hyderabad")

        ColonialDiplomacyEnv.assert_true(not success, "Support should fail due to illegal adjacency")
        ColonialDiplomacyEnv.assert_equal(supported["strength"], original_strength, "Illegal support should not change strength")

        print("support_unit() with illegal adjacency tests passed\n")

    @staticmethod
    def test_support_unit_army_into_water(env):
        print("Testing support_unit() with army into water...")

        supported = env.units[5][0]
        original_strength = supported["strength"]

        success = env.support_unit(supporter_pid=5, supporter_idx=3, supported_pid=5, supported_idx=0, supported_destination="Upper_Pacific")

        ColonialDiplomacyEnv.assert_true(not success, "Army should not support into water")
        ColonialDiplomacyEnv.assert_equal(supported["strength"], original_strength, "Illegal support should not change strength")

        print("support_unit() with army into water tests passed\n")

    @staticmethod
    def test_support_unit_fleet_into_land(env):
        print("Testing support_unit() with fleet into land...")

        supported = env.units[6][0]
        original_strength = supported["strength"]

        success = env.support_unit(supporter_pid=6, supporter_idx=3, supported_pid=6, supported_idx=0, supported_destination="Perm")

        ColonialDiplomacyEnv.assert_true(not success, "Fleet should not support into land")
        ColonialDiplomacyEnv.assert_equal(supported["strength"], original_strength, "Illegal support should not change strength")

        print("support_unit() with fleet into land tests passed\n")

    @staticmethod
    def test_convoy_army_single_fleet(env):
        print("Testing convoy_army() with single fleet...")

        env.units[4].append({"type": "Fleet", "location": "Timor_Sea", "strength": 1})
        env.units[4].append({"type": "Army", "location": "Celebes", "strength": 1})

        army_idx = len(env.units[4]) - 1

        success = env.convoy_army(army_pid=4, army_idx=army_idx, destination="New_Guinea")

        ColonialDiplomacyEnv.assert_true(success, "Convoy should succeed")
        ColonialDiplomacyEnv.assert_equal(env.units[4][army_idx]["location"], "New_Guinea", "Army did not arrive at destination")

        print("convoy_army() with single fleet tests passed\n")

    @staticmethod
    def test_convoy_army_multi_fleet(env):
        print("Testing convoy_army() with multiple fleets...")
        
        env.units[2].append({"type": "Fleet", "location": "East_China_Sea", "strength": 1})
        env.units[2].append({"type": "Fleet", "location": "Yellow_Sea", "strength": 1})
        env.units[2].append({"type": "Army", "location": "Nanchang", "strength": 1})

        army_idx = len(env.units[2]) - 1

        success = env.convoy_army(army_pid=2, army_idx=army_idx, destination="Fusan")

        ColonialDiplomacyEnv.assert_true(success, "Multi-fleet convoy should succeed")
        ColonialDiplomacyEnv.assert_equal(env.units[2][army_idx]["location"], "Fusan", "Army did not arrive at convoy destination")

        print("convoy_army() with multiple fleets tests passed\n")

    @staticmethod
    def test_convoy_army_no_fleets(env):
        print("Testing convoy_army() with no fleets...")

        blocked_waters = {"Celebes_Sea"}

        for pid, units in env.units.items():
            env.units[pid] = [u for u in units if not (u["type"] == "Fleet" and u["location"] in blocked_waters)]

        success = env.convoy_army(army_pid=4, army_idx=0, destination="Davao")

        ColonialDiplomacyEnv.assert_true(not success, "Convoy should fail with no fleets")

        print("convoy_army() with no fleets tests passed\n")

    @staticmethod
    def test_convoy_army_non_coastal_destination(env):
        print("Testing convoy_army() with non-coastal destination...")

        success = env.convoy_army(army_pid=3, army_idx=0, destination="Mandalay")

        ColonialDiplomacyEnv.assert_true(not success, "Convoy should fail to non-coastal destination")

        print("convoy_army() with non-coastal destination tests passed\n")

    @staticmethod
    def test_get_unit_at(env):
        print("Testing get_unit_at()...")

        pid, unit = env.get_unit_at("Delhi")
        ColonialDiplomacyEnv.assert_equal(pid, 1, "Delhi should contain a British unit")
        ColonialDiplomacyEnv.assert_equal(unit["type"], "Army", "Delhi unit should be an Army")

        pid, unit = env.get_unit_at("Tokyo")
        ColonialDiplomacyEnv.assert_equal(pid, 5, "Tokyo should contain a Japanese unit")
        ColonialDiplomacyEnv.assert_equal(unit["type"], "Fleet", "Tokyo unit should be a Fleet")

        pid, unit = env.get_unit_at("Assam")
        ColonialDiplomacyEnv.assert_true(pid is None and unit is None, "Assam should be empty")

        pid, unit = env.get_unit_at("Rome")
        ColonialDiplomacyEnv.assert_true(pid is None and unit is None, "Unknown province should return None")

        print("get_unit_at() tests passed\n")

    @staticmethod
    def test_can_unit_move_to(env):
        print("Testing can_unit_move_to()...")
        
        ColonialDiplomacyEnv.assert_true(
            env.can_unit_move_to("Army", "Delhi"),
            "Army should be able to move to land province"
        )

        ColonialDiplomacyEnv.assert_true(
            not env.can_unit_move_to("Army", "Arabian_Sea"),
            "Army should NOT be able to move to water province"
        )

        ColonialDiplomacyEnv.assert_true(
            env.can_unit_move_to("Fleet", "Arabian_Sea"),
            "Fleet should be able to move to water province"
        )

        ColonialDiplomacyEnv.assert_true(
            not env.can_unit_move_to("Fleet", "Delhi"),
            "Fleet should NOT be able to move to land province"
        )

        ColonialDiplomacyEnv.assert_true(
            not env.can_unit_move_to("Airship", "Delhi"),
            "Invalid unit type should return false"
        )

        print("can_unit_move_to() tests passed\n")

    @staticmethod
    def test_get_legal_retreat_locations(env):
        print("Testing get_legal_retreat_locations()...")

        unit = {"type": "Army", "location": "Delhi", "strength": 1}

        attacker_origin = "Punjab"
        standoff_provinces = {"Nagpur"}

        env.units[1].append({"type": "Army", "location": "Lucknow", "strength": 1})

        retreats = env.get_legal_retreat_locations(unit=unit, attacker_origin=attacker_origin, standoff_provinces=standoff_provinces)

        expected = ["Nepal", "Rajputana"]

        ColonialDiplomacyEnv.assert_equal(sorted(retreats), sorted(expected), "Incorrect legal retreat locations")

        print("get_legal_retreat_locations() tests passed\n")

    @staticmethod
    def test_retreat_unit_success(env):
        print("testing retreat_unit() with success...")

        env.units[1].append({"type": "Army", "location": "Delhi", "strength": 1})

        success = env.retreat_unit(player_id=1, unit_location="Delhi", attacker_origin="Punjab", standoff_provinces=set(), chosen_retreat="Rajputana")

        ColonialDiplomacyEnv.assert_true(success, "Retreat should succeed")

        unit_locations = [u["location"] for u in env.units[1]]
        ColonialDiplomacyEnv.assert_true("Rajputana" in unit_locations, "Unit did not retreat to correct location")

        print("retreat_unit() with success tests passed\n")

    @staticmethod
    def test_retreat_unit_illegal_choice_disbands(env):
        print("testing retreat_unit() with illegal choice...")

        env.units[1].append({"type": "Army", "location": "Bengal", "strength": 1})

        success = env.retreat_unit(player_id=1, unit_location="Bengal", attacker_origin="Assam", standoff_provinces=set(), chosen_retreat="Assam")
        
        ColonialDiplomacyEnv.assert_true(not success, "Illegal retreat should fail")

        unit_locations = [u["location"] for u in env.units[1]]
        ColonialDiplomacyEnv.assert_true("Bengal" not in unit_locations, "Unit should have been disbanded")

        print("retreat_unit() with illegal choice tests passed\n")

    def test_retreat_unit_declined_disbands(env):
        print("testing retreat_unit() with declined retreat...")

        env.units[1].append({"type": "Army", "location": "Bengal", "strength": 1})

        success = env.retreat_unit(player_id=1, unit_location="Bengal", attacker_origin="Assam", standoff_provinces=set(), chosen_retreat=None)

        ColonialDiplomacyEnv.assert_true(not success, "Declined retreat should fail")

        unit_locations = [u["location"] for u in env.units[1]]
        ColonialDiplomacyEnv.assert_true("Bengal" not in unit_locations, "Unit should have been disbanded")

        print("retreat_unit() with declined retreat tests passed\n")

    @staticmethod
    def test_retreat_unit_no_legal_retreats(env):
        print("test_retreat_unit() with no legal retreats")

        env.units[1].append({"type": "Army", "location": "Abyssinia", "strength": 1})

        env.units[2].append({"type": "Army", "location": "Sudan", "strength": 1})
        env.units[3].append({"type": "Army", "location": "Somaliland", "strength": 1})
        env.units[4].append({"type": "Army", "location": "Sudan", "strength": 1})

        success = env.retreat_unit(player_id=1, unit_location="Abyssinia", attacker_origin="Sudan", standoff_provinces=set(), chosen_retreat="Somaliland")

        ColonialDiplomacyEnv.assert_true(not success, "Unit should be disbanded when no legal retreats exist")

        unit_locations = [u["location"] for u in env.units[1]]
        ColonialDiplomacyEnv.assert_true("Abyssinia" not in unit_locations, "Unit should have been disbanded")

        print("retreat_unit() with no legal retreats tests passed\n")

    @staticmethod
    def test_disband_unit(env):
        print("Testing disband_unit()...")

        # Create temporary player with two units
        env.units[42] = [{"type": "Army", "location": "Persia", "strength": 1}, {"type": "Army", "location": "Shiraz", "strength": 1}]

        # Successful disband
        initial_count = len(env.units[42])
        result = env.disband_unit(player_id=42, unit_idx=0)
        ColonialDiplomacyEnv.assert_true(result, "Valid disband should return True")
        ColonialDiplomacyEnv.assert_equal(len(env.units[42]), initial_count - 1, "Unit count should decrease by one after disband")

        # Invalid index: negative
        result = env.disband_unit(player_id=42, unit_idx=-1)
        ColonialDiplomacyEnv.assert_true(not result, "Negative unit index should fail")

        # Invalid index: out of range
        result = env.disband_unit(player_id=42, unit_idx=999)
        ColonialDiplomacyEnv.assert_true(not result, "Out-of-range unit index should fail")

        # No side effects on failure
        count_before = len(env.units[42])
        env.disband_unit(player_id=42, unit_idx=999)
        ColonialDiplomacyEnv.assert_equal(len(env.units[42]), count_before, "Failed disband should not modify unit list")

        # Remove last unit and eliminate player
        result = env.disband_unit(player_id=42, unit_idx=0)
        ColonialDiplomacyEnv.assert_true(result, "Disbanding the last unit should succeed")
        ColonialDiplomacyEnv.assert_true(42 not in env.units, "Player should be removed after last unit is disbanded")

        print("disband_unit() tests passed\n")

    @staticmethod
    def test_is_province_occupied(env):
        print("Testing is_province_occupied()...")

        ColonialDiplomacyEnv.assert_true(env.is_province_occupied("Delhi"), "Delhi should be occupied")
        ColonialDiplomacyEnv.assert_true(env.is_province_occupied("Bombay"), "Bombay should be occupied")
        ColonialDiplomacyEnv.assert_true(not env.is_province_occupied("Kashmir"), "Kashmir should not be occupied")
        ColonialDiplomacyEnv.assert_true(not env.is_province_occupied("Assam"), "Assam should not be occupied")
        ColonialDiplomacyEnv.assert_true(not env.is_province_occupied("Arabian_Sea"), "Arabian_Sea should not be occupied")

        print("is_province_occupied() tests passed\n")

    @staticmethod
    def test_build_unit(env):
        print("Testing build_unit()...")

        # Enter build phase
        env.phase = "Build"

        # Temporary country
        env.units[10] = []
        env.home_supply_centers.update({"Persia": 10, "Shiraz": 10, "Tabriz": 10})
        env.supply_centers["Persia"] = 10
        env.supply_centers["Shiraz"] = 10
        env.supply_centers["Tabriz"] = 10

        initial_count = len(env.units[10])

        # Successful Army build on land home SC
        result = env.build_unit(player_id=10, province="Tabriz", unit_type="Army")
        ColonialDiplomacyEnv.assert_true(result, "Valid Army build should succeed")
        ColonialDiplomacyEnv.assert_equal(len(env.units[10]), initial_count + 1, "Army build should increase unit count")

        pid, unit = env.get_unit_at("Tabriz")
        ColonialDiplomacyEnv.assert_equal(unit["type"], "Army", "Built unit should be an Army")

        initial_count = len(env.units[10])

        # Successful Fleet build on coastal home SC
        result = env.build_unit(player_id=10, province="Shiraz", unit_type="Fleet")
        ColonialDiplomacyEnv.assert_true(result, "Valid Fleet build should succeed")
        ColonialDiplomacyEnv.assert_equal(len(env.units[10]), initial_count + 1, "Fleet build should increase unit count")

        pid, unit = env.get_unit_at("Shiraz")
        ColonialDiplomacyEnv.assert_equal(unit["type"], "Fleet", "Built unit should be a Fleet")

        # Wrong phase
        env.phase = "Spring"
        result = env.build_unit(player_id=10, province="Persia", unit_type="Army")
        ColonialDiplomacyEnv.assert_true(not result, "Build outside Build phase should fail")

        env.phase = "Build"

        # Not a home supply center
        result = env.build_unit(player_id=10, province="Karachi", unit_type="Army")
        ColonialDiplomacyEnv.assert_true(not result, "Build in non-home supply center should fail")

        # Home SC owned by another player
        result = env.build_unit(player_id=10, province="Baghdad", unit_type="Army")
        ColonialDiplomacyEnv.assert_true(not result, "Build in another player's home SC should fail")

        # Province occupied
        result = env.build_unit(player_id=10, province="Tabriz", unit_type="Army")
        ColonialDiplomacyEnv.assert_true(not result, "Build in occupied province should fail")

        # No available build slots
        env.supply_centers["Persia"] = 0

        result = env.build_unit(player_id=10, province="Tabriz", unit_type="Army")
        ColonialDiplomacyEnv.assert_true(not result, "Build should fail when unit count >= supply count")

        # Restore supply center
        env.supply_centers["Persia"] = 10

        # Fleet in non-coastal province
        result = env.build_unit(player_id=10, province="Tabriz", unit_type="Fleet")
        ColonialDiplomacyEnv.assert_true(not result, "Fleet should not be buildable in non-coastal province")

        # Invalid unit type
        result = env.build_unit(player_id=10, province="Persia", unit_type="Airship")
        ColonialDiplomacyEnv.assert_true(not result, "Invalid unit type should fail")
        
        print("build_unit() tests passed\n")

    @staticmethod
    def test_use_tsr_success(env):
        print("Testing use_trans_siberian_railroad success...")

        env.tsr_used_this_turn = False
        
        result = env.use_trans_siberian_railroad(player_id=6, start_province="Omsk", chosen_destination="Irkutsk")

        ColonialDiplomacyEnv.assert_equal(result, "Irkutsk", "TSR should move army to Irkutsk")

        unit_locs = [u["location"] for u in env.units[6]]
        ColonialDiplomacyEnv.assert_true("Irkutsk" in unit_locs, "Army not found at TSR destination")

        env.tsr_used_this_turn = False

        env.use_trans_siberian_railroad(player_id=6, start_province="Irkutsk", chosen_destination="Omsk")

        print("use_trans_siberian_railroad() success tests passed\n")

    @staticmethod
    def test_use_tsr_pass_through_russian_unit(env):
        print("Testing use_trans_siberian_railroad pass through Russian unit...")

        env.tsr_used_this_turn = False

        result = env.use_trans_siberian_railroad(player_id=6, start_province="Moscow", chosen_destination="Krasnoyarsk")

        ColonialDiplomacyEnv.assert_equal(result, "Krasnoyarsk", "TSR should pass through Russian unit and stop later")
        
        print("use_trans_siberian_railroad() through Russian unit tests passed\n")

    @staticmethod
    def test_use_tsr_foreign_unit_blocks(env):
        print("Testing use_trans_siberian_railroad() foreign block...")

        env.tsr_used_this_turn = False

        env.units[7] = [{"type": "Army", "location": "Perm", "strength": 1}]

        ColonialDiplomacyEnv.assert_raises(ValueError, lambda: env.use_trans_siberian_railroad(player_id=6, start_province="Moscow", chosen_destination="Omsk"), "TSR should fail when foreign unit blocks destination")

        print("use_trans_siberian_railroad() foreign unit block tests passed\n")

    @staticmethod
    def test_use_tsr_standoff_blocks(env):
        print("Testing use_trans_siberian_railroad() with standoff...")

        env.tsr_used_this_turn = False

        ColonialDiplomacyEnv.assert_raises(ValueError, lambda: env.use_trans_siberian_railroad(player_id=6, start_province="Omsk", chosen_destination="Irkutsk", standoff_provinces={"Krasnoyarsk"}), "TSR should stop before standoff province")

        print("use_trans_siberian_railroad() with standoff tests passed\n")

    @staticmethod
    def test_use_tsr_cannot_end_on_russian_unit(env):
        print("Testing use_trans_siberian_railroad() with occupied province...")

        env.tsr_used_this_turn = False

        ColonialDiplomacyEnv.assert_raises(ValueError, lambda: env.use_trans_siberian_railroad(player_id=6, start_province="Moscow", chosen_destination="Omsk"), "TSR should not allow ending on occupied province")

        print("use_trans_siberian_railroad() with occupied province tests passed\n")

    @staticmethod
    def test_use_tsr_once_per_turn(env):
        print("Testing use_trans_siberian_railroad() once per turn...")

        for pid, units in env.units.items():
            env.units[pid] = [u for u in units if u["location"] not in env.tsr_path]

        env.tsr_used_this_turn = False

        env.units[6] = [{"type": "Army", "location": "Moscow", "strength": 1}]

        result = env.use_trans_siberian_railroad(player_id=6, start_province="Moscow", chosen_destination="Perm")

        ColonialDiplomacyEnv.assert_equal(result, "Perm", "First TSR move should succeed")

        ColonialDiplomacyEnv.assert_raises(ValueError, lambda: env.use_trans_siberian_railroad(player_id=6, start_province="Perm", chosen_destination="Omsk"), "TSR should only be usable once per turn")

        print("use_trans_siberian_railroad() once per turn tests passed\n")

    @staticmethod
    def test_use_tsr_only_russia(env):
        print("Testing use_trans_siberian_railroad() only Russia...")

        ColonialDiplomacyEnv.assert_raises(ValueError, lambda: env.use_trans_siberian_railroad(player_id=2, start_province="Irkutsk", chosen_destination="Krasnoyarsk"), "Only Russia should be allowed to use TSR")

        print("use_trans_siberian_railroad() only Russia tests passed\n")

    @staticmethod
    def test_use_tsr_requires_army_and_tsr_start(env):
        print("Testing use_trans_siberian_railroad() with army and TSR start...")

        env.tsr_used_this_turn = False

        ColonialDiplomacyEnv.assert_raises(ValueError, lambda: env.use_trans_siberian_railroad(player_id=6, start_province="Odessa", chosen_destination="Perm"), "Fleet should not be allowed on TSR")

        print("use_trans_siberian_railroad() requires army and TSR tests passed\n")

    @staticmethod
    def test_get_controller_of_egypt_none(env):
        print("Testing get_controller_of_egypt() none...")

        for pid, units in env.units.items():
            env.units[pid] = [u for u in units if u["location"] != "Egypt"]
        
        controller = env.get_controller_of_egypt()

        ColonialDiplomacyEnv.assert_equal(controller, None, "Egypt should have no controller when unoccupied")

        print("get_canonical_supply_center() none tests passed\n")

    @staticmethod
    def test_get_controller_of_egypt_occupied(env):
        print("Testing get_controller_of_egypt() occupied...")

        for pid, units in env.units.items():
            env.units[pid] = [u for u in units if u["location"] != "Egypt"]

        env.units[1].append({"type": "Army", "location": "Egypt", "strength": 1})

        controller = env.get_controller_of_egypt()

        ColonialDiplomacyEnv.assert_equal(controller, 1, "Egypt should be controlled by the occupying power")

        print("get_controller_of_egypt() occupied tests passed\n")

    @staticmethod
    def test_grant_suez_permission_initial(env):
        print("Testing grant_suez_permission initial...")

        env.suez_permissions.clear()

        env.grant_suez_permission(controller_id=1, foreign_player=3)

        ColonialDiplomacyEnv.assert_true(1 in env.suez_permissions, "Controller should have an entry in suez_permissions")

        ColonialDiplomacyEnv.assert_true(3 in env.suez_permissions[1], "Foreign player should be granted Suez permission")

        print("grant_suez_permission() initial tests passed\n")

    @staticmethod
    def test_grant_suez_permission_multiple(env):
        print("Testing grant_suez_permission() multiple...")

        env.suez_permissions.clear()

        env.grant_suez_permission(controller_id=1, foreign_player=3)
        env.grant_suez_permission(controller_id=1, foreign_player=4)

        ColonialDiplomacyEnv.assert_equal(env.suez_permissions[1], {3, 4}, "Controller should be able to grant multiple permissions")

        print("grant_suez_permission() multiple tests passed\n")

    @staticmethod
    def test_suez_control_and_permission_flow(env):
        print("Testing grant_suez_permission() idempotent...")

        env.suez_permissions.clear()

        env.grant_suez_permission(controller_id=1, foreign_player=3)
        env.grant_suez_permission(controller_id=1, foreign_player=3)

        ColonialDiplomacyEnv.assert_equal(env.suez_permissions[1], {3}, "Granting permission twice should not duplicate entries")

        print("grant_suez_permission() idempotent tests passed\n")

    @staticmethod
    def test_suez_unoccupied_egypt(env):
        print("Testing Suez Canal with unoccupied Egypt...")

        for pid, units in env.units.items():
            env.units[pid] = [
                u for u in units
                if u["location"] not in {"Egypt", "Mediterranean_Sea", "Red_Sea"}
            ]

        env.units[1].append({"type": "Fleet", "location": "Mediterranean_Sea", "strength": 1})

        allowed = env.can_use_suez_canal(1, "Mediterranean_Sea", "Red_Sea")
        ColonialDiplomacyEnv.assert_equal(allowed, False, "Should not allow Suez Canal move when Egypt is unoccupied")

        print("Suez Canal with unoccupied Egypt test passed\n")

    @staticmethod
    def test_suez_same_controller(env):
        print("Testing Suez Canal with same controller...")

        for pid, units in env.units.items():
            env.units[pid] = [
                u for u in units
                if u["location"] not in {"Egypt", "Mediterranean_Sea", "Red_Sea"}
            ]

        env.units[1].append({"type": "Army", "location": "Egypt", "strength": 1})
        env.units[1].append({"type": "Fleet", "location": "Mediterranean_Sea", "strength": 1})

        allowed = env.can_use_suez_canal(1, "Mediterranean_Sea", "Red_Sea")
        ColonialDiplomacyEnv.assert_true(allowed, "Controller of Egypt should always be allowed")

        print("Suez Canal same controller test passed\n")

    @staticmethod
    def test_suez_foreign_no_permission(env):
        print("Testing Suez Canal foreign control without permission...")

        env.suez_permissions.clear()

        for pid, units in env.units.items():
            env.units[pid] = [
                u for u in units
                if u["location"] not in {"Egypt", "Mediterranean_Sea", "Red_Sea"}
            ]

        env.units[7].append({"type": "Army", "location": "Egypt", "strength": 1})
        env.units[1].append({"type": "Fleet", "location": "Mediterranean_Sea", "strength": 1})

        allowed = env.can_use_suez_canal(1, "Mediterranean_Sea", "Red_Sea")
        ColonialDiplomacyEnv.assert_equal(allowed, False, "Should deny Suez without permission")

        print("Suez Canal foreign no permission test passed\n")

    @staticmethod
    def test_suez_foreign_with_permission(env):
        print("Testing Suez Canal foreign control with permission...")

        env.suez_permissions.clear()
        for pid, units in env.units.items():
            env.units[pid] = [
                u for u in units
                if u["location"] not in {"Egypt", "Mediterranean_Sea", "Red_Sea"}
            ]

        env.units[7].append({"type": "Army", "location": "Egypt", "strength": 1})
        env.units[1].append({"type": "Fleet", "location": "Mediterranean_Sea", "strength": 1})

        env.grant_suez_permission(controller_id=7, foreign_player=1)

        allowed = env.can_use_suez_canal(1, "Mediterranean_Sea", "Red_Sea")
        ColonialDiplomacyEnv.assert_true(allowed, "Permission should allow Suez passage")

        print("Suez Canal foreign with permission test passed\n")

    @staticmethod
    def test_execute_suez_move(env):
        print("Testing execute_suez_move() success...")

        env.suez_permissions.clear()

        for pid, units in env.units.items():
            env.units[pid] = [
                u for u in units
                if u["location"] not in {"Egypt", "Mediterranean_Sea", "Red_Sea"}
            ]

        env.units[1].append({"type": "Army", "location": "Egypt", "strength": 1})
        env.units[1].append({"type": "Fleet", "location": "Mediterranean_Sea", "strength": 1})

        # HARD DIAGNOSTICS
        pid, unit = env.get_unit_at("Mediterranean_Sea")
        print("DEBUG fleet check:", pid, unit)

        controller = env.get_controller_of_egypt()
        print("DEBUG Egypt controller:", controller)

        print("DEBUG can_use_suez_canal():", env.can_use_suez_canal(1, "Mediterranean_Sea", "Red_Sea"))

        # This assert MUST pass before execution
        ColonialDiplomacyEnv.assert_true(env.can_use_suez_canal(1, "Mediterranean_Sea", "Red_Sea"), "Precondition failed: can_use_suez_canal() returned False")

        result = env.execute_suez_move(1, "Mediterranean_Sea", "Red_Sea")
        ColonialDiplomacyEnv.assert_true(result, "execute_suez_move should succeed")

        pid, unit = env.get_unit_at("Red_Sea")
        ColonialDiplomacyEnv.assert_equal(pid, 1, "Fleet should be in Red_Sea")

        print("execute_suez_move() test passed\n")

    @staticmethod
    def test_get_canonical_supply_center(env):
        print("Testing get_canonical_supply_center()...")

        # Canonical names should return themselves
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Bangkok"), "Bangkok", "Canonical  supply center should return itself")
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Seoul"), "Seoul", "Canonical supply center should return itself")

        # Variant names should map to canonical
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Bangkok_east_coast"), "Bangkok", "Bangkok_east_coast should canonicalize to Bangkok")
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Bangkok_west_coast"), "Bangkok", "Bangkok_west_coast should canonicalize to Bangkok")
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Seoul_east_coast"), "Seoul", "Seoul_east_coast should canonicalize to Seoul")
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Seoul_west_coast"), "Seoul", "Seoul_west_coast should canonicalize to Seoul")

        # Non-grouped supply centers should pass through unchanged
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Delhi"), "Delhi", "Non-grouped supply center should remain unchanged")

        # Non-supply-center provinces should pass through unchanged
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Arabian_Sea"), "Arabian_Sea", "Non-supply-center province should remain unchanged")

        # Unknown provinces should pass through unchanged
        ColonialDiplomacyEnv.assert_equal(env.get_canonical_supply_center("Rome"), "Rome", "Unknown province should remain unchanged")

        print("get_canonical_supply_center() tests passed\n")

    @staticmethod
    def test_update_supply_center_control_not_fall(env):
        print("Testing update_supply_center_control() when phase != Fall...")

        env.phase = "Spring"

        before = dict(env.supply_centers)
        env.update_supply_center_control()

        ColonialDiplomacyEnv.assert_equal(env.supply_centers, before, "Supply centers should not change outside Fall phase")

        print("update_supply_center_control() non-Fall test passed\n")

    @staticmethod
    def test_update_supply_center_control_simple_capture(env):
        print("Testing update_supply_center_control() simple capture...")

        env.phase = "Fall"

        # Ensure Bangkok starts neutral
        ColonialDiplomacyEnv.assert_equal(env.supply_centers["Bangkok"], 0, "Bangkok should start neutral")

        # Add a French army to Bangkok
        env.units[3].append({"type": "Army", "location": "Bangkok", "strength": 1})

        env.update_supply_center_control()

        ColonialDiplomacyEnv.assert_equal(env.supply_centers["Bangkok"], 3, "France should capture Karachi")

        print("update_supply_center_control() simple capture test passed\n")

    @staticmethod
    def test_update_supply_center_control_coastal_canonical(env):
        print("Testing update_supply_center_control() coastal canonical mapping...")

        env.phase = "Fall"

        # Reset ownership
        env.supply_centers["Bangkok"] = 0

        # Dutch fleet on east coast
        env.units[4].append({"type": "Fleet", "location": "Bangkok_east_coast", "strength": 1})

        env.update_supply_center_control()

        ColonialDiplomacyEnv.assert_equal(env.supply_centers["Bangkok"], 4, "Coastal occupation should grant control of canonical supply center")

        print("update_supply_center_control() coastal canonical test passed\n")

    @staticmethod
    def test_update_supply_center_control_hong_kong_rule(env):
        print("Testing update_supply_center_control() Hong Kong special rule...")

        env.phase = "Fall"

        # Hong Kong starts British
        ColonialDiplomacyEnv.assert_equal(env.supply_centers["Hong_Kong"], 1, "Hong Kong should start British")

        # Chinese army occupies Hong Kong
        env.units[2].append({"type": "Army", "location": "Hong_Kong", "strength": 1})

        env.update_supply_center_control()

        ColonialDiplomacyEnv.assert_equal(env.supply_centers["Hong_Kong"], 1, "China should not be able to capture Hong Kong")

        print("update_supply_center_control() Hong Kong rule test passed\n")

    @staticmethod
    def test_count_supply_centers(env):
        print("Testing count_supply_centers()...")

        ColonialDiplomacyEnv.assert_true(env.count_supply_centers(1) == 6, "Britain should have 6 supply centers")
        ColonialDiplomacyEnv.assert_true(env.count_supply_centers(2) == 5, "China should have 5 supply centers")
        ColonialDiplomacyEnv.assert_true(env.count_supply_centers(3) == 3, "France should have 3 supply centers")
        ColonialDiplomacyEnv.assert_true(env.count_supply_centers(4) == 3, "Holland should have 3 supply centers")
        ColonialDiplomacyEnv.assert_true(env.count_supply_centers(5) == 4, "Japan should have 4 supply centers")
        ColonialDiplomacyEnv.assert_true(env.count_supply_centers(6) == 5, "Russia should have 5 supply centers")
        ColonialDiplomacyEnv.assert_true(env.count_supply_centers(7) == 3, "Turkey should have 3 supply centers")

        print("count_supply_centers() tests passed\n")

    @staticmethod
    def test_start_new_turn(env):
        print("Testing start_new_turn()...")

        env.tsr_used_this_turn = True
        env.suez_permissions = {1: {2, 3}}

        env.start_new_turn()

        ColonialDiplomacyEnv.assert_true(env.tsr_used_this_turn is False, "TSR flag should be reset to False")
        ColonialDiplomacyEnv.assert_equal(env.suez_permissions, {}, "Suez permissions should be cleared")

        print("start_new_turn() tests passed\n")

    @staticmethod
    def test_end_of_fall_phase_no_winner(env):
        print("Testing end_of_fall_phase() with no winner...")

        env.phase = "Fall"
        env.done = False

        env.end_of_fall_phase()

        ColonialDiplomacyEnv.assert_equal(env.phase, "Build", "Phase should advance to Build when no winner is found")
        ColonialDiplomacyEnv.assert_true(env.done is False, "Game should not be marked done when no winner exists")

        print("end_of_fall_phase() (no winner) tests passed\n")

    @staticmethod
    def test_end_of_fall_phase_with_winner(env):
        print("Testing end_of_fall_phase() with winner...")

        # Clear units so they do not overwrite supply centers
        env.units = {pid: [] for pid in env.units}

        # Force a victory for China
        provinces = list(env.supply_centers.keys())
        for i in range(30):
            env.supply_centers[provinces[i]] = 2
        
        env.phase = "Fall"
        env.done = False
        env.winner = None

        env.end_of_fall_phase()

        ColonialDiplomacyEnv.assert_true(env.done is True, "Game should be marked done when a winner is found")
        ColonialDiplomacyEnv.assert_equal(env.winner, 2, "Winner should be China")

        # Phase should NOT advance if game is over
        ColonialDiplomacyEnv.assert_equal(env.phase, "Fall", "Phase should not advance after game ends")

        print("end_of_fall_phase() (with winner) tests passed\n")

    @staticmethod
    def test_check_victory(env):
        print("Testing check_victory()...")

        # No one should win at game start
        ColonialDiplomacyEnv.assert_equal(env.check_victory(), None, "No player should have won at game start")

        # Give British player 30 supply centers
        provinces = list(env.supply_centers.keys())
        for i in range(30):
            env.supply_centers[provinces[i]] = 1

        ColonialDiplomacyEnv.assert_equal(env.check_victory(), 1, "British player should be detected as winner")

        print("check_victory() tests passed\n")

    @staticmethod
    def main():
        print("Running ColonialDiplomacyEnv tests...\n")
        env = ColonialDiplomacyEnv()

        try:
            ColonialDiplomacyEnv.test_move_unit(env)
            ColonialDiplomacyEnv.test_hold_unit(env)

            ColonialDiplomacyEnv.test_support_unit_hold(env)
            ColonialDiplomacyEnv.test_support_unit_move(env)
            ColonialDiplomacyEnv.test_support_unit_illegal_adjacency(env)
            ColonialDiplomacyEnv.test_support_unit_army_into_water(env)
            ColonialDiplomacyEnv.test_support_unit_fleet_into_land(env)

            ColonialDiplomacyEnv.test_convoy_army_single_fleet(env)
            ColonialDiplomacyEnv.test_convoy_army_multi_fleet(env)
            ColonialDiplomacyEnv.test_convoy_army_no_fleets(env)
            ColonialDiplomacyEnv.test_convoy_army_non_coastal_destination(env)

            ColonialDiplomacyEnv.test_get_unit_at(env)
            ColonialDiplomacyEnv.test_can_unit_move_to(env)

            ColonialDiplomacyEnv.test_get_legal_retreat_locations(env)
            ColonialDiplomacyEnv.test_retreat_unit_success(env)
            ColonialDiplomacyEnv.test_retreat_unit_illegal_choice_disbands(env)
            ColonialDiplomacyEnv.test_retreat_unit_declined_disbands(env)

            ColonialDiplomacyEnv.test_disband_unit(env)
            ColonialDiplomacyEnv.test_is_province_occupied(env)
            ColonialDiplomacyEnv.test_build_unit(env)

            ColonialDiplomacyEnv.test_use_tsr_success(env)
            ColonialDiplomacyEnv.test_use_tsr_pass_through_russian_unit(env)
            ColonialDiplomacyEnv.test_use_tsr_foreign_unit_blocks(env)
            ColonialDiplomacyEnv.test_use_tsr_standoff_blocks(env)
            ColonialDiplomacyEnv.test_use_tsr_cannot_end_on_russian_unit(env)
            ColonialDiplomacyEnv.test_use_tsr_once_per_turn(env)
            ColonialDiplomacyEnv.test_use_tsr_only_russia(env)
            ColonialDiplomacyEnv.test_use_tsr_requires_army_and_tsr_start(env)

            ColonialDiplomacyEnv.test_get_controller_of_egypt_none(env)
            ColonialDiplomacyEnv.test_get_controller_of_egypt_occupied(env)

            ColonialDiplomacyEnv.test_grant_suez_permission_initial(env)
            ColonialDiplomacyEnv.test_grant_suez_permission_multiple(env)
            ColonialDiplomacyEnv.test_suez_control_and_permission_flow(env)

            ColonialDiplomacyEnv.test_suez_unoccupied_egypt(env)
            ColonialDiplomacyEnv.test_suez_same_controller(env)
            ColonialDiplomacyEnv.test_suez_foreign_no_permission(env)
            ColonialDiplomacyEnv.test_suez_foreign_with_permission(env)
            ColonialDiplomacyEnv.test_execute_suez_move(env)

            ColonialDiplomacyEnv.test_get_canonical_supply_center(env)
            ColonialDiplomacyEnv.test_count_supply_centers(env)
            ColonialDiplomacyEnv.test_update_supply_center_control_not_fall(env)
            ColonialDiplomacyEnv.test_update_supply_center_control_simple_capture(env)
            ColonialDiplomacyEnv.test_update_supply_center_control_coastal_canonical(env)
            ColonialDiplomacyEnv.test_update_supply_center_control_hong_kong_rule(env)

            # Endgame tests
            ColonialDiplomacyEnv.test_start_new_turn(env)
            ColonialDiplomacyEnv.test_end_of_fall_phase_no_winner(env)
            ColonialDiplomacyEnv.test_check_victory(env)
            ColonialDiplomacyEnv.test_end_of_fall_phase_with_winner(env)
        except AssertionError as e:
            print("TEST FAILED")
            print(e)
            return 1

        print("ALL TESTS PASSED")
        return 0

if __name__ == "__main__":
    ColonialDiplomacyEnv.main()