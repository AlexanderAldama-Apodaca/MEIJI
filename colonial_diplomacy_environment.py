import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque

from dataclasses import dataclass
from collections import defaultdict

import json

class ColonialDiplomacyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.num_players: int = self.config.get("num_players", 7) # Britain, China, France, Holland, Japan, Russia, Turkey
        self.max_years: int = self.config.get("max_years", 1908) # Colonial Diplomacy ends after 1908
        self.max_units: int = self.config.get("max_units", 58)

        # Game History Logging
        self.history = []

        # Game State Variables
        self.year: int = 1870 # Colonial Diplomacy begins in 1870
        self.phase_index= 0

        self.phase_cycle = [
            "Spring_Movement",
            "Spring_Retreat",
            "Fall_Movement",
            "Fall_Retreat",
            "Winter_Adjustment"
        ]

        self.phase = self.phase_cycle[self.phase_index]

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

        self.multi_coast_provinces = {
            "Arabia": ["NC", "SC"],
            "Bangkok": ["EC", "WC"],
            "Seoul": ["EC", "WC"]
        }

        self.trans_siberian_railroad: Dict[str, int] = {
            "Irkutsk": 6, "Krasnoyarsk": 6, "Moscow": 6, "Omsk": 6, "Perm": 6, "Vladivostok": 6
        }

        self.tsr_path = ["Moscow", "Perm", "Omsk", "Krasnoyarsk", "Irkutsk", "Vladivostok"]

    def validate_move_order(self, unit, destination):
        current = unit["location"]

        # Adjacency check
        if destination not in self.adjacency.get(current, []):
            return False
        
        # Army movement rules
        if unit["type"] == "Army":
            if destination in self.water_provinces:
                return False
            
        # Fleet movement rules
        elif unit["type"] == "Fleet":
            if destination in self.land_provinces:
                return False
            
            # Coast restrictions
            if (
                current in self.multi_coast_provinces
                and destination in self.multi_coast_provinces
            ):
                if current == destination:
                    return False

        return True
        
    def validate_support_order(self, support_order, all_orders):
        supporter_location = support_order.unit_location
        support_unit = support_order.support_unit
        support_target = support_order.support_target

        # Supported unit must exist
        pid, unit = self.get_unit_at(support_unit)

        if unit is None:
            return False
        
        # Supporter must be adjacent
        if support_target not in self.adjacency.get(supporter_location, []):
            return False
        
        # Supported move must exist
        found = False

        for _, orders in all_orders.items():
            for order in orders:
                if (order.unit_location == support_unit
                    and order.order_type == "MOVE"
                    and order.target == support_target
                    ):
                    found = True

        return found
    
    def is_support_cut(self, support_order, move_orders):
        supporter = support_order.unit_location

        protected_source = support_order.support_target

        for move in move_orders:
            # Attack on supporter
            if move.target != supporter:
                continue

            # Exception: supported unit attacking supporter
            if move.unit_location == protected_source:
                continue

            return True
        
        return False
    
    def detect_head_to_head(self, move_a, move_b):
        return (
            move_a.unit_location == move_b.target
            and move_a.target == move_b.unit_location
        )
        
    def validate_convoy_order(self, convoy_order):
        start = convoy_order.convoy_origin
        destination = convoy_order.convoy_destination

        visited = set()

        queue = [start]

        while queue:
            current = queue.pop(0)

            if current == destination:
                return True
            
            visited.add(current)

            for adjacent in self.adjacency.get(current, []):
                if adjacent in visited:
                    continue

                pid, unit = self.get_unit_at(adjacent)

                if (
                    unit is not None
                    and unit["type"] == "Fleet"
                ):
                    queue.append(adjacent)

        return False
    
    def is_convoy_disrupted(self, convoy_fleets, dislodged_units):
        dislodged_locations = {
            unit["location"]
            for _, unit in dislodged_units
        }

        for fleet in convoy_fleets:
            if fleet in dislodged_locations:
                return True
            
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
        
    def get_legal_retreat_locations(self, unit, attacked_provinces=None):
        if attacked_provinces is None:
            attacked_provinces = set()

        retreats = []

        current = unit["location"]

        for adjacent in self.adjacency.get(current, []):
            if adjacent in attacked_provinces:
                continue
            if self.is_province_occupied(adjacent):
                continue
            if unit["type"] == "Army":
                if adjacent in self.water_provinces:
                    continue
            elif unit["type"] == "Fleet":
                if adjacent in self.land_provinces:
                    continue

            retreats.append(adjacent)

        return retreats
        
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
    
    def validate_build(self, player_id, province):
        if province not in self.home_supply_centers[player_id]:
            return False
        
        if self.supply_centers.get(province) != player_id:
            return False
        
        if self.is_province_occupied(province):
            return False

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
    
    def validate_all_orders(self, joint_orders):
        valid_orders = {}

        for pid, orders in joint_orders.items():
            valid_orders[pid] = []

            for order in orders:
                _, unit = self.get_unit_at(order.unit_location)

                if unit is None:
                    continue

                if order.order_type == "MOVE":
                    if self.validate_move_order(unit, order.target):
                        valid_orders[pid].append(order)

                    elif order.order_type == "HOLD":
                        valid_orders[pid].append(order)

                    elif order.order_type == "SUPPORT":
                        valid_orders[pid].append(order)

                    elif order.order_type == "CONVOY":
                        valid_orders[pid].append(order)

            return valid_orders

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

    def export_state(self):
        units_export = {}

        for pid, units in self.units.items():
            units_export[str(pid)] = []

            for unit in units:
                prefix = "A" if unit["type"] == "Army" else "F"
                units_export[str(pid)].append(f"{prefix} {unit['location']}")
        
        centers_export = {}

        for pid in range(1, self.num_players + 1):
            owned = []

            for province, owner in self.supply_centers.items():
                if owner == pid:
                    owned.append(province)

            centers_export[str(pid)] = sorted(owned)

        return {
            "year": self.year,
            "phase": self.phase,
            "units": units_export,
            "centers": centers_export
        }
    
    def submit_orders(self, joint_orders):
        """
        joint_orders:
        {
            player_id: [Order, Order, ...]
        }
        """

        self.pending_orders = self.validate_all_orders(joint_orders)

    def resolve_orders(self):
        """
        Simultaneously adjudicate all movement orders.
        """

        move_orders = []
        hold_orders = []
        support_orders = []

        order_results = {}

        # Categorize orders
        for pid, orders in self.pending_orders.items():
            for order in orders:
                if order.order_type == "MOVE":
                    move_orders.append(order)
                
                elif order.order_type == "HOLD":
                    hold_orders.append(order)
                
                elif order.order_type == "SUPPORT":
                    support_orders.append(order)

        # Compute support strengths
        attack_strength = defaultdict(lambda: 1)
        defense_strength = defaultdict(lambda: 1)

        for order in support_orders:
            if self.is_support_cut(order, move_orders):
                continue

            if not self.validate_support_order(order, self.pending_orders):
                continue

            attack_strength[(order.support_unit, order.support_target)] += 1

        # Collect attacks by destination
        attacks_by_destination = defaultdict(list)

        for order in move_orders:
            attacks_by_destination[order.target].append(order)
        
        successful_moves = []
        bounced_moves = []

        for i in range(len(move_orders)):
            for j in range(i + 1, len(move_orders)):
                a = move_orders[i]
                b = move_orders[j]

                if self.detect_head_to_head(a, b):
                    strength_a = attack_strength[(a.unit_location, a.target)]
                    strength_b = attack_strength[(b.unit_location, b.target)]

                    if strength_a > strength_b:
                        order_results[b.unit_location] = ["bounce"]

                    elif strength_b > strength_a:
                        order_results[a.unit_location] = ["bounce"]

                    else:
                        order_results[a.unit_location] = ["bounce"]
                        order_results[b.unit_location] = ["bounce"]

        # Resolve movement conflicts
        for destination, attacks in attacks_by_destination.items():
            if len(attacks) == 1:
                successful_moves.append(attacks[0])
                continue

            strengths = []

            for attack in attacks:
                strength = attack_strength[(attack.unit_location, attack.target)]
                strengths.append((strength, attack))

            strengths.sort(key=lambda x: x[0], reverse=True)

            best_strength = strengths[0][0]

            top = [x for x in strengths if x[0] == best_strength]

            if len(top) > 1:
                for _, attack in top:
                    bounced_moves.append(attack)
                    order_results[attack.unit_location] = ["bounce"]
            else:
                successful_moves.append(top[0][1])

        # Apply successful moves simultaneously
        new_positions = {}

        for pid, units in self.units.items():
            for unit in units:
                new_positions[unit["location"]] = unit

        for move in successful_moves:
            pid, unit = self.get_unit_at(move.unit_location)

            if unit is None:
                continue

            old_location = unit["location"]
            new_location = move.target

            unit["location"] = new_location

            order_results[old_location] = []

        # Holds
        for hold in hold_orders:
            order_results[hold.unit_location] = []

        return 
    
    def advance_phase(self):
        self.phase_index += 1

        if self.phase_index >= len(self.phase_cycle):
            self.phase_index = 0
            self.year += 1

        self.phase = self.phase_cycle[self.phase_index]

    def record_phase(self, orders, results):
        phase_record = {
            "name": f"{self.phase}_{self.year}",
            "state": self.export_state(),
            "orders": {},
            "results": results,
            "messages": []
        }

        for pid, order_list in orders.items():
            phase_record["orders"][str(pid)] = []

            for order in order_list:
                if order.order_type == "MOVE":
                    text = f"{order.unit_location} - {order.target}"

                elif order.order_type == "HOLD":
                    text = f"{order.unit_location} H"

                elif order.order_type == "SUPPORT":
                    text = (
                        f"{order.unit_location} S "
                        f"{order.support_unit} - {order.support_target}"
                    )

                else:
                    text = "UNKNOWN"

                phase_record["orders"][str(pid)].append(text)

        self.history.append(phase_record)

    def step(self, joint_orders):
        self.submit_orders(joint_orders)

        results = self.resolve_orders()
        
        if self.phase == "Fall_Movement":
            self.update_supply_center_control()

        self.record_phase(joint_orders, results)

        self.advance_phase()

        observations = self.build_all_observations()

        rewards = self.compute_rewards()

        done = self.check_victory() is not None
        
        info = {
            "history_size": len(self.history)
        }

        return observations, rewards, done, info

    def compute_rewards(self):
        rewards = {}

        for pid in range(1, self.num_players + 1):
            rewards[pid] = self.count_supply_centers(pid)

        return rewards
    
    def build_observation(self, player_id):
        obs = {
            "phase": self.phase,
            "year": self.year,
            "owned_centers": [],
            "friendly_units": [],
            "enemy_units": []
        }

        for province, owner in self.supply_centers.items():
            if owner == player_id:
                obs["owned_centers"].append(province)

        for pid, units in self.units.items():
            for unit in units:
                data = {
                    "type": unit["type"],
                    "location": unit["location"]
                }

                if pid == player_id:
                    obs["friendly_units"].append(data)
                else:
                    obs["enemy_units"].append(data)

    def build_all_observations(self):
        return {
            pid: self.build_observation(pid)
            for pid in range(1, self.num_players + 1)
        }
    
    def export_game_json(self):
        return {
            "map": "colonial",
            "phases": self.history
        }
    
    def save_game(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.export_game_json(), f, indent=2)

    def main():
        print("Creating Colonial Diplomacy Environment\n")
        env = ColonialDiplomacyEnv()
        print("Colonial Diplomacy Environment created!\n")
        return 0

if __name__ == "__main__":
    ColonialDiplomacyEnv.main()

@dataclass
class Order:
    player_id: int
    unit_location: str
    order_type: str

    target: str | None = None

    support_unit: str | None = None
    support_target: str | None = None

    convoy_origin: str | None = None
    convoy_destination: str | None = None

    via_tsr: bool = False
    via_suez: bool = False
