# colonial_diplomacy_environment_initial.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Any, Set

class ColonialDiplomacyEnv(gym.Env):
    """
    Colonial Diplomacy - New Experiment

    - Full province list and adjacency map are included (exactly as provided previously).
    - The experiment itself focuses on a subset of provinces listed in the spec:
        Arabian Sea, Bay of Bengal, Bengal, Bombay, Ceylon, Delhi, East Indian Ocean,
        Gulf of Manaar, Hyderabad, Kashmir, Lucknow, Madras, Mysore, Nagpur, Punjab,
        Rajputana, West Indian Ocean

    - Britain controls 3 starting units: A_Delhi, A_Madras, F_Bombay.
    - To control a supply center the agent must have a unit in that province at the end
      of an even-numbered turn (first evaluation occurs at the end of turn 2).
    - Goal: control six supply centers: Bengal, Bombay, Ceylon, Delhi, Kashmir, Madras.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    PROVINCE_NAMES = [
        "Abyssinia","Aden","Afghanistan","Akita","Akmolinsk","Andaman Sea","Angora","Annam","Arabia",
        "Arabia north coast","Arabia south coast","Arabian Sea","Armenia","Assam","Baghdad","Baku",
        "Bangkok","Bangkok east coast","Bangkok west coast", "Bay of Bengal","Bengal", "Black Sea","Bokhara",
        "Bombay","Borneo","Cambodia","Canton","Cebu","Celebes","Celebes Sea","Ceylon","Chungking",
        "Cochin","Constantinople","Davao","Delhi","East China Sea","East Indian Ocean","Egypt",
        "Eritrea","Formosa","Fusan","Gulf of Aden","Gulf of Manaar","Gulf of Siam","Hong Kong",
        "Hyderabad","Irkutsk","Java","Java Sea","Karachi","Kashgar","Kashmir","Kirghiz","Krasnoyarsk",
        "Kyoto","Kyushu","Langchow","Lower Pacific","Lucknow","Luzon Strait","Madras","Malaya",
        "Manchuria","Mandalay","Manila","Mecca","Mediterranean Sea","Middle Pacific","Mongolia",
        "Moscow","Mysore","Nagpur","Nanchang","Nepal","New Guinea","North Siam","Odessa","Okhotsk Sea",
        "Oman","Omsk","Orenburg","Otaru","Peking","Perm","Persia","Persian Gulf","Port Arthur","Punjab",
        "Rajputana","Rangoon","Red Sea","Rumania","Sakhalin","Sarawak","Sea of Japan","Semipalatinsk",
        "Seoul","Seoul east coast","Seoul west coast","Shanghai","Shiraz","Singapore","Sinkiang",
        "Somaliland","South China Sea","Southeast Indian Ocean","Sudan","Sulu Sea","Sumatra","Sunda Sea",
        "Syria","Tabriz","Tashkent","Tibet","Timor Sea","Tokyo","Tongking","Upper Burma","Upper Pacific",
        "Urumchi","Vladivostok","West Indian Ocean","Yellow Sea","Yunnan"
    ]

    # Index constants
    ABYSSINIA, ADEN, AFGHANISTAN, AKITA, AKMOLINSK, ANDAMAN_SEA, ANGORA, ANNAM, ARABIA, ARABIA_NORTH_COAST, ARABIA_SOUTH_COAST, ARABIAN_SEA, ARMENIA, ASSAM, BAGHDAD, BAKU, BANGKOK, BANGKOK_EAST_COAST, BANGKOK_WEST_COAST, BAY_OF_BENGAL, BENGAL, BLACK_SEA, BOKHARA, BOMBAY, BORNEO, CAMBODIA, CANTON, CEBU, CELEBES, CELEBES_SEA, CEYLON, CHUNGKING, COCHIN, CONSTANTINOPLE, DAVAO, DELHI, EAST_CHINA_SEA, EAST_INDIAN_OCEAN, EGYPT, ERITREA, FORMOSA, FUSAN, GULF_OF_ADEN, GULF_OF_MANAAR, GULF_OF_SIAM, HONG_KONG, HYDERABAD, IRKUTSK, JAVA, JAVA_SEA, KARACHI, KASHGAR, KASHMIR, KIRGHIZ, KRASNOYARSK, KYOTO, KYUSHU, LANGCHOW, LOWER_PACIFIC, LUCKNOW, LUZON_STRAIT, MADRAS, MALAYA, MANCHURIA, MANDALAY, MANILA, MECCA, MEDITERRANEAN_SEA, MIDDLE_PACIFIC, MONGOLIA, MOSCOW, MYSORE, NAGPUR, NANCHANG, NEPAL, NEW_GUINEA, NORTH_SIAM, ODESSA, OKHOTSK_SEA, OMAN, OMSK, ORENBURG, OTARU, PEKING, PERM, PERSIA, PERSIAN_GULF, PORT_ARTHUR, PUNJAB, RAJPUTANA, RANGOON, RED_SEA, RUMANIA, SAKHALIN, SARAWAK, SEA_OF_JAPAN, SEMIPALATINSK, SEOUL, SEOUL_EAST_COAST, SEOUL_WEST_COAST, SHANGHAI, SHIRAZ, SINGAPORE, SINKIANG, SOMALILAND, SOUTH_CHINA_SEA, SOUTHEAST_INDIAN_OCEAN, SUDAN, SULU_SEA, SUMATRA, SUNDA_SEA, SYRIA, TABRIZ, TASHKENT, TIBET, TIMOR_SEA, TOKYO, TONGKING, UPPER_BURMA, UPPER_PACIFIC, URUMCHI, VLADIVOSTOK, WEST_INDIAN_OCEAN, YELLOW_SEA, YUNNAN = range(125)

    print(YUNNAN)

    def __init__(self, max_steps: int = 50, initial_units: List[Dict[str, Any]] = None):
        super().__init__()

        self.n_provinces = len(self.PROVINCE_NAMES)

        # adjacency map (exactly as provided)
        self.adjacency = {
            # Britain
            self.ADEN: {self.ARABIA, self.ARABIA_SOUTH_COAST, self.GULF_OF_ADEN, self.MECCA, self.RED_SEA},
            self.BENGAL: {self.ASSAM, self.BAY_OF_BENGAL, self.HYDERABAD, self.LUCKNOW, self.NEPAL, self.TIBET, self.UPPER_BURMA},
            self.BOMBAY: {self.ARABIAN_SEA, self.HYDERABAD, self.MYSORE, self.NAGPUR, self.RAJPUTANA},
            self.CEYLON: {self.EAST_INDIAN_OCEAN, self.GULF_OF_MANAAR, self.WEST_INDIAN_OCEAN},
            self.DELHI: {self.LUCKNOW, self.NAGPUR, self.NEPAL, self.PUNJAB, self.RAJPUTANA},
            self.HONG_KONG: {self.CANTON, self.SOUTH_CHINA_SEA},
            self.HYDERABAD: {self.BAY_OF_BENGAL, self.BENGAL, self.BOMBAY, self.GULF_OF_MANAAR, self.LUCKNOW, self.MADRAS, self.MYSORE, self.NAGPUR},
            self.KASHMIR: {self.AFGHANISTAN, self.KASHGAR, self.PUNJAB, self.TIBET},
            self.LUCKNOW: {self.BENGAL, self.DELHI, self.HYDERABAD, self.NAGPUR, self.NEPAL},
            self.MADRAS: {self.GULF_OF_MANAAR, self.HYDERABAD, self.MYSORE, self.WEST_INDIAN_OCEAN},
            self.MYSORE: {self.ARABIAN_SEA, self.BOMBAY, self.HYDERABAD, self.MADRAS, self.WEST_INDIAN_OCEAN},
            self.NAGPUR: {self.BOMBAY, self.DELHI, self.HYDERABAD, self.LUCKNOW, self.RAJPUTANA},
            self.PUNJAB: {self.AFGHANISTAN, self.DELHI, self.KARACHI, self.KASHMIR, self.NEPAL, self.RAJPUTANA, self.TIBET},
            self.RAJPUTANA: {self.ARABIAN_SEA, self.BOMBAY, self.DELHI, self.KARACHI, self.NAGPUR, self.PUNJAB},
            self.SINGAPORE: {self.JAVA_SEA, self.MALAYA},
            
            # China
            self.CANTON: {self.CHUNGKING, self.HONG_KONG, self.MANDALAY, self.NANCHANG, self.SOUTH_CHINA_SEA, self.TONGKING, self.YUNNAN},
            self.CHUNGKING: {self.CANTON, self.LANGCHOW, self.NANCHANG, self.SINKIANG, self.YUNNAN},
            self.KASHGAR: {self.AFGHANISTAN, self.KASHMIR, self.KIRGHIZ, self.SINKIANG, self.TASHKENT, self.TIBET, self.URUMCHI},
            self.LANGCHOW: {self.CHUNGKING, self.MONGOLIA, self.NANCHANG, self.PEKING, self.SHANGHAI, self.SINKIANG},
            self.MANCHURIA: {self.IRKUTSK, self.MONGOLIA, self.PEKING, self.PORT_ARTHUR, self.SEOUL, self.SHANGHAI, self.VLADIVOSTOK, self.YELLOW_SEA},
            self.MONGOLIA: {self.IRKUTSK, self.KRASNOYARSK, self.LANGCHOW, self.MANCHURIA, self.PEKING, self.SINKIANG, self.URUMCHI},
            self.NANCHANG: {self.CANTON, self.CHUNGKING, self.EAST_CHINA_SEA, self.LANGCHOW, self.SHANGHAI, self.SOUTH_CHINA_SEA},
            self.PEKING: {self.LANGCHOW, self.MANCHURIA, self.MONGOLIA, self.SHANGHAI},
            self.SHANGHAI: {self.EAST_CHINA_SEA, self.LANGCHOW, self.MANCHURIA, self.NANCHANG, self.PEKING, self.YELLOW_SEA},
            self.SINKIANG: {self.ASSAM, self.CHUNGKING, self.KASHGAR, self.LANGCHOW, self.MONGOLIA, self.TIBET, self.URUMCHI, self.YUNNAN},
            self.TIBET: {self.ASSAM, self.BENGAL, self.KASHGAR, self.KASHMIR, self.PUNJAB, self.SINKIANG},
            self.URUMCHI: {self.KASHGAR, self.KIRGHIZ, self.KRASNOYARSK, self.MONGOLIA, self.SEMIPALATINSK, self.SINKIANG},
            self.YUNNAN: {self.ASSAM, self.CANTON, self.CHUNGKING, self.MANDALAY, self.SINKIANG, self.UPPER_BURMA},

            # France
            self.ANNAM: {self.CAMBODIA, self.COCHIN, self.GULF_OF_SIAM, self.SOUTH_CHINA_SEA, self.TONGKING},
            self.CAMBODIA: {self.ANNAM, self.BANGKOK, self.BANGKOK_EAST_COAST, self.COCHIN, self.GULF_OF_SIAM, self.NORTH_SIAM, self.TONGKING},
            self.COCHIN: {self.ANNAM, self.CAMBODIA, self.GULF_OF_SIAM},
            self.TONGKING: {self.ANNAM, self.CAMBODIA, self.CANTON, self.MANDALAY, self.NORTH_SIAM},

            # Holland
            self.BORNEO: {self.CELEBES_SEA, self.JAVA_SEA, self.SARAWAK},
            self.CELEBES: {self.CELEBES_SEA, self.JAVA_SEA, self.TIMOR_SEA},
            self.JAVA: {self.JAVA_SEA, self.SOUTHEAST_INDIAN_OCEAN, self.TIMOR_SEA},
            self.SUMATRA: {self.ANDAMAN_SEA, self.EAST_INDIAN_OCEAN, self.JAVA_SEA, self.SOUTHEAST_INDIAN_OCEAN},

            # Japan
            self.AKITA: {self.KYOTO, self.OKHOTSK_SEA, self.OTARU, self.SEA_OF_JAPAN, self.TOKYO},
            self.KYOTO: {self.AKITA, self.KYUSHU, self.SEA_OF_JAPAN, self.TOKYO, self.UPPER_PACIFIC, self.YELLOW_SEA},
            self.KYUSHU: {self.EAST_CHINA_SEA, self.KYOTO, self.UPPER_PACIFIC, self.YELLOW_SEA},
            self.OTARU: {self.AKITA, self.OKHOTSK_SEA, self.SAKHALIN, self.SEA_OF_JAPAN},
            self.TOKYO: {self.AKITA, self.KYOTO, self.OKHOTSK_SEA, self.UPPER_PACIFIC},

            # Russia
            self.AKMOLINSK: {self.KIRGHIZ, self.KRASNOYARSK, self.OMSK, self.ORENBURG, self.SEMIPALATINSK, self.TASHKENT},
            self.BAKU: {self.ARMENIA, self.BLACK_SEA, self.MOSCOW, self.ODESSA, self.TABRIZ},
            self.BOKHARA: {self.AFGHANISTAN, self.MOSCOW, self.ORENBURG, self.PERSIA, self.TASHKENT},
            self.IRKUTSK: {self.KRASNOYARSK, self.MANCHURIA, self.MONGOLIA, self.VLADIVOSTOK},
            self.KIRGHIZ: {self.AKMOLINSK, self.KASHGAR, self.SEMIPALATINSK, self.TASHKENT, self.URUMCHI},
            self.KRASNOYARSK: {self.AKMOLINSK, self.IRKUTSK, self.MONGOLIA, self.OMSK, self.SEMIPALATINSK, self.URUMCHI},
            self.MOSCOW: {self.BAKU, self.BOKHARA, self.ODESSA, self.ORENBURG, self.PERM},
            self.ODESSA: {self.BAKU, self.BLACK_SEA, self.MOSCOW, self.RUMANIA},
            self.OMSK: {self.AKMOLINSK, self.KRASNOYARSK, self.ORENBURG, self.PERM},
            self.ORENBURG: {self.AKMOLINSK, self.BOKHARA, self.MOSCOW, self.OMSK, self.PERM, self.TASHKENT},
            self.PERM: {self.MOSCOW, self.OMSK, self.ORENBURG},
            self.PORT_ARTHUR: {self.MANCHURIA, self.SEOUL, self.SEOUL_WEST_COAST, self.YELLOW_SEA},
            self.SAKHALIN: {self.OKHOTSK_SEA, self.OTARU},
            self.SEMIPALATINSK: {self.AKMOLINSK, self.KIRGHIZ, self.KRASNOYARSK, self.URUMCHI},
            self.TASHKENT: {self.AFGHANISTAN, self.AKMOLINSK, self.BOKHARA, self.KASHGAR, self.KIRGHIZ, self.ORENBURG},
            self.VLADIVOSTOK: {self.IRKUTSK, self.MANCHURIA, self.OKHOTSK_SEA, self.SEA_OF_JAPAN, self.SEOUL, self.SEOUL_EAST_COAST},

            # Turkey
            self.ANGORA: {self.ARMENIA, self.BLACK_SEA, self.CONSTANTINOPLE, self.MEDITERRANEAN_SEA, self.SYRIA},
            self.ARMENIA: {self.ANGORA, self.BAGHDAD, self.BAKU, self.BLACK_SEA, self.SYRIA, self.TABRIZ},
            self.BAGHDAD: {self.ARABIA, self.ARABIA_NORTH_COAST, self.ARMENIA, self.PERSIAN_GULF, self.SHIRAZ, self.SYRIA, self.TABRIZ},
            self.CONSTANTINOPLE: {self.ANGORA, self.BLACK_SEA, self.MEDITERRANEAN_SEA, self.RUMANIA},
            self.MECCA: {self.ADEN, self.ARABIA, self.EGYPT, self.RED_SEA, self.SYRIA},
            self.SYRIA: {self.ANGORA, self.ARABIA, self.ARMENIA, self.BAGHDAD, self.EGYPT, self.MECCA, self.MEDITERRANEAN_SEA},

            # Neutrals
            self.ABYSSINIA: {self.ERITREA, self.SOMALILAND, self.SUDAN},
            self.AFGHANISTAN: {self.BOKHARA, self.KARACHI, self.KASHGAR, self.KASHMIR, self.PERSIA, self.PUNJAB, self.TASHKENT},
            self.ARABIA: {self.ADEN, self.ARABIAN_SEA, self.BAGHDAD, self.GULF_OF_ADEN, self.MECCA, self.OMAN, self.PERSIAN_GULF, self.SYRIA},
            self.ARABIA_NORTH_COAST: {self.BAGHDAD, self.OMAN, self.PERSIAN_GULF},
            self.ARABIA_SOUTH_COAST: {self.ADEN, self.ARABIAN_SEA, self.GULF_OF_ADEN, self.OMAN},
            self.ASSAM: {self.BENGAL, self.SINKIANG, self.TIBET, self.UPPER_BURMA, self.YUNNAN},
            self.BANGKOK: {self.ANDAMAN_SEA, self.CAMBODIA, self.GULF_OF_SIAM, self.MALAYA, self.NORTH_SIAM, self.RANGOON},
            self.BANGKOK_EAST_COAST: {self.CAMBODIA, self.GULF_OF_SIAM, self.MALAYA},
            self.BANGKOK_WEST_COAST: {self.ANDAMAN_SEA, self.MALAYA, self.RANGOON},
            self.CEBU: {self.DAVAO, self.LOWER_PACIFIC, self.LUZON_STRAIT, self.MANILA, self.MIDDLE_PACIFIC, self.SULU_SEA},
            self.DAVAO: {self.CEBU, self.CELEBES_SEA, self.LOWER_PACIFIC, self.SULU_SEA},
            self.EGYPT: {self.MECCA, self.MEDITERRANEAN_SEA, self.RED_SEA, self.SUDAN, self.SYRIA},
            self.ERITREA: {self.ABYSSINIA, self.GULF_OF_ADEN, self.RED_SEA, self.SOMALILAND, self.SUDAN},
            self.FORMOSA: {self.EAST_CHINA_SEA, self.LUZON_STRAIT, self.MIDDLE_PACIFIC, self.SOUTH_CHINA_SEA, self.UPPER_PACIFIC},
            self.FUSAN: {self.SEA_OF_JAPAN, self.SEOUL, self.SEOUL_EAST_COAST, self.SEOUL_WEST_COAST, self.YELLOW_SEA},
            self.KARACHI: {self.AFGHANISTAN, self.ARABIAN_SEA, self.PERSIA, self.PERSIAN_GULF, self.PUNJAB, self.RAJPUTANA},
            self.MALAYA: {self.ANDAMAN_SEA, self.BANGKOK, self.BANGKOK_EAST_COAST, self.BANGKOK_WEST_COAST, self.GULF_OF_SIAM, self.JAVA_SEA, self.SINGAPORE, self.SUNDA_SEA},
            self.MANDALAY: {self.CANTON, self.NORTH_SIAM, self.RANGOON, self.TONGKING, self.UPPER_BURMA, self.YUNNAN},
            self.MANILA: {self.CEBU, self.LUZON_STRAIT, self.MIDDLE_PACIFIC},
            self.NEPAL: {self.BENGAL, self.DELHI, self.LUCKNOW, self.PUNJAB},
            self.NEW_GUINEA: {self.CELEBES_SEA, self.LOWER_PACIFIC, self.TIMOR_SEA},
            self.NORTH_SIAM: {self.BANGKOK, self.CAMBODIA, self.MANDALAY, self.RANGOON, self.TONGKING},
            self.OMAN: {self.ARABIA, self.ARABIA_NORTH_COAST, self.ARABIA_SOUTH_COAST, self.ARABIAN_SEA, self.PERSIAN_GULF},
            self.PERSIA: {self.AFGHANISTAN, self.BOKHARA, self.KARACHI, self.PERSIAN_GULF, self.SHIRAZ, self.TABRIZ},
            self.RANGOON: {self.ANDAMAN_SEA, self.BANGKOK, self.BANGKOK_WEST_COAST, self.BAY_OF_BENGAL, self.MANDALAY, self.NORTH_SIAM, self.UPPER_BURMA},
            self.RUMANIA: {self.BLACK_SEA, self.CONSTANTINOPLE, self.ODESSA},
            self.SARAWAK: {self.BORNEO, self.CELEBES_SEA, self.JAVA_SEA, self.SULU_SEA, self.SUNDA_SEA},
            self.SEOUL: {self.FUSAN, self.MANCHURIA, self.PORT_ARTHUR, self.SEA_OF_JAPAN, self.VLADIVOSTOK, self.YELLOW_SEA},
            self.SEOUL_EAST_COAST: {self.FUSAN, self.SEA_OF_JAPAN, self.VLADIVOSTOK},
            self.SEOUL_WEST_COAST: {self.FUSAN, self.PORT_ARTHUR, self.YELLOW_SEA},
            self.SHIRAZ: {self.BAGHDAD, self.PERSIA, self.PERSIAN_GULF, self.TABRIZ},
            self.SOMALILAND: {self.ABYSSINIA, self.ERITREA, self.GULF_OF_ADEN},
            self.SUDAN: {self.ABYSSINIA, self.EGYPT, self.ERITREA, self.RED_SEA},
            self.TABRIZ: {self.ARMENIA, self.BAGHDAD, self.BAKU, self.PERSIA, self.SHIRAZ},
            self.UPPER_BURMA: {self.ASSAM, self.BAY_OF_BENGAL, self.BENGAL, self.MANDALAY, self.RANGOON, self.YUNNAN},

            # Water
            self.ANDAMAN_SEA: {self.BANGKOK, self.BANGKOK_WEST_COAST, self.BAY_OF_BENGAL, self.EAST_INDIAN_OCEAN, self.GULF_OF_MANAAR, self.JAVA_SEA, self.MALAYA, self.RANGOON, self.SUMATRA},
            self.ARABIAN_SEA: {self.ARABIA, self.ARABIA_SOUTH_COAST, self.BOMBAY, self.GULF_OF_ADEN, self.KARACHI, self.MYSORE, self.OMAN, self.PERSIAN_GULF, self.RAJPUTANA, self.WEST_INDIAN_OCEAN},
            self.BAY_OF_BENGAL: {self.ANDAMAN_SEA, self.BENGAL, self.GULF_OF_MANAAR, self.HYDERABAD, self.RANGOON, self.UPPER_BURMA},
            self.BLACK_SEA: {self.ANGORA, self.ARMENIA, self.BAKU, self.CONSTANTINOPLE, self.MEDITERRANEAN_SEA, self.ODESSA, self.RUMANIA},
            self.CELEBES_SEA: {self.BORNEO, self.CELEBES, self.DAVAO, self.JAVA_SEA, self.LOWER_PACIFIC, self.NEW_GUINEA, self.SARAWAK, self.SULU_SEA, self.TIMOR_SEA},
            self.EAST_CHINA_SEA: {self.FORMOSA, self.KYUSHU, self.NANCHANG, self.SHANGHAI, self.SOUTH_CHINA_SEA, self.UPPER_PACIFIC, self.YELLOW_SEA},
            self.EAST_INDIAN_OCEAN: {self.ANDAMAN_SEA, self.CEYLON, self.GULF_OF_MANAAR, self.SOUTHEAST_INDIAN_OCEAN, self.SUMATRA, self.WEST_INDIAN_OCEAN},
            self.GULF_OF_ADEN: {self.ADEN, self.ARABIA, self.ARABIA_SOUTH_COAST, self.ARABIAN_SEA, self.ERITREA, self.RED_SEA, self.SOMALILAND, self.WEST_INDIAN_OCEAN},
            self.GULF_OF_MANAAR: {self.ANDAMAN_SEA, self.BAY_OF_BENGAL, self.CEYLON, self.EAST_INDIAN_OCEAN, self.HYDERABAD, self.MADRAS, self.WEST_INDIAN_OCEAN},
            self.GULF_OF_SIAM: {self.ANNAM, self.BANGKOK, self.BANGKOK_EAST_COAST, self.CAMBODIA, self.COCHIN, self.MALAYA, self.SOUTH_CHINA_SEA, self.SUNDA_SEA},
            self.JAVA_SEA: {self.ANDAMAN_SEA, self.BORNEO, self.CELEBES, self.CELEBES_SEA, self.JAVA, self.MALAYA, self.SARAWAK, self.SINGAPORE, self.SOUTHEAST_INDIAN_OCEAN, self.SUMATRA, self.SUNDA_SEA, self.TIMOR_SEA},
            self.LOWER_PACIFIC: {self.CEBU, self.CELEBES_SEA, self.DAVAO, self.MIDDLE_PACIFIC, self.NEW_GUINEA},
            self.LUZON_STRAIT: {self.CEBU, self.FORMOSA, self.MANILA, self.MIDDLE_PACIFIC, self.SOUTH_CHINA_SEA, self.SULU_SEA},
            self.MEDITERRANEAN_SEA: {self.ANGORA, self.BLACK_SEA, self.CONSTANTINOPLE, self.EGYPT, self.SYRIA},
            self.MIDDLE_PACIFIC: {self.CEBU, self.FORMOSA, self.LOWER_PACIFIC, self.LUZON_STRAIT, self.MANILA, self.UPPER_PACIFIC},
            self.OKHOTSK_SEA: {self.AKITA, self.OTARU, self.SAKHALIN, self.SEA_OF_JAPAN, self.TOKYO, self.UPPER_PACIFIC, self.VLADIVOSTOK},
            self.PERSIAN_GULF: {self.ARABIA, self.ARABIA_NORTH_COAST, self.ARABIAN_SEA, self.BAGHDAD, self.KARACHI, self.OMAN, self.PERSIA, self.SHIRAZ},
            self.RED_SEA: {self.ADEN, self.EGYPT, self.GULF_OF_ADEN, self.MECCA, self.SUDAN},
            self.SEA_OF_JAPAN: {self.AKITA, self.FUSAN, self.KYOTO, self.OKHOTSK_SEA, self.OTARU, self.SEOUL, self.SEOUL_EAST_COAST, self.VLADIVOSTOK, self.YELLOW_SEA},
            self.SOUTH_CHINA_SEA: {self.ANNAM, self.CANTON, self.EAST_CHINA_SEA, self.FORMOSA, self.GULF_OF_SIAM, self.HONG_KONG, self.LUZON_STRAIT, self.NANCHANG, self.SULU_SEA, self.SUNDA_SEA, self.TONGKING},
            self.SOUTHEAST_INDIAN_OCEAN: {self.EAST_INDIAN_OCEAN, self.JAVA, self.JAVA_SEA, self.SUMATRA, self.TIMOR_SEA, self.WEST_INDIAN_OCEAN},
            self.SULU_SEA: {self.CEBU, self.CELEBES_SEA, self.DAVAO, self.LUZON_STRAIT, self.SARAWAK, self.SOUTH_CHINA_SEA, self.SUNDA_SEA},
            self.SUNDA_SEA: {self.GULF_OF_SIAM, self.JAVA_SEA, self.MALAYA, self.SARAWAK, self.SOUTH_CHINA_SEA, self.SULU_SEA},
            self.TIMOR_SEA: {self.CELEBES, self.CELEBES_SEA, self.JAVA, self.JAVA_SEA, self.NEW_GUINEA, self.SOUTHEAST_INDIAN_OCEAN},
            self.UPPER_PACIFIC: {self.EAST_CHINA_SEA, self.FORMOSA, self.KYOTO, self.KYUSHU, self.MIDDLE_PACIFIC, self.OKHOTSK_SEA, self.TOKYO, self.YELLOW_SEA},
            self.WEST_INDIAN_OCEAN: {self.ARABIAN_SEA, self.CEYLON, self.EAST_INDIAN_OCEAN, self.GULF_OF_ADEN, self.GULF_OF_MANAAR, self.MADRAS, self.MYSORE, self.SOUTHEAST_INDIAN_OCEAN},
            self.YELLOW_SEA: {self.EAST_CHINA_SEA, self.FUSAN, self.KYOTO, self.KYUSHU, self.MANCHURIA, self.PORT_ARTHUR, self.SEA_OF_JAPAN, self.SEOUL, self.SEOUL_WEST_COAST, self.SHANGHAI, self.UPPER_PACIFIC},
        }

        # Define land provinces (subset used for validation)
        self.LAND_PROVINCES = {
            self.ABYSSINIA, self.AFGHANISTAN, self.AKMOLINSK, self.ASSAM,
            self.BOKHARA, self.CHUNGKING, self.DELHI, self.IRKUTSK,
            self.KASHGAR, self.KASHMIR, self.KIRGHIZ, self.KRASNOYARSK,
            self.LANGCHOW, self.LUCKNOW, self.MANDALAY, self.MONGOLIA,
            self.MOSCOW, self.NAGPUR, self.NEPAL, self.NORTH_SIAM,
            self.OMSK, self.ORENBURG, self.PEKING, self.PERM,
            self.PUNJAB, self.SEMIPALATINSK, self.SINKIANG, self.TABRIZ,
            self.TASHKENT, self.TIBET, self.URUMCHI, self.YUNNAN
        }

        # Define water provinces
        self.WATER_PROVINCES = {
            self.ANDAMAN_SEA, self.ARABIAN_SEA, self.BAY_OF_BENGAL, self.BLACK_SEA,
            self.CELEBES_SEA, self.EAST_CHINA_SEA, self.EAST_INDIAN_OCEAN,
            self.GULF_OF_ADEN, self.GULF_OF_MANAAR, self.GULF_OF_SIAM, self.JAVA_SEA,
            self.LOWER_PACIFIC, self.LUZON_STRAIT, self.MEDITERRANEAN_SEA, self.MIDDLE_PACIFIC,
            self.OKHOTSK_SEA, self.PERSIAN_GULF, self.RED_SEA, self.SEA_OF_JAPAN,
            self.SOUTH_CHINA_SEA, self.SOUTHEAST_INDIAN_OCEAN, self.SULU_SEA, self.SUNDA_SEA,
            self.TIMOR_SEA, self.UPPER_PACIFIC, self.WEST_INDIAN_OCEAN, self.YELLOW_SEA
        }

        # Define coast provinces (a broad set; includes many coastal provinces)
        self.COAST_PROVINCES = {
            self.ADEN, self.AKITA, self.ANGORA, self.ANNAM,
            self.ARABIA_NORTH_COAST, self.ARABIA_SOUTH_COAST, self.ARMENIA, self.BAGHDAD,
            self.BAKU, self.BANGKOK_EAST_COAST, self.BANGKOK_WEST_COAST, self.BENGAL,
            self.BOMBAY, self.BORNEO, self.CAMBODIA, self.CANTON,
            self.CEBU, self.CELEBES, self.CEYLON, self.COCHIN,
            self.CONSTANTINOPLE, self.DAVAO, self.EGYPT, self.ERITREA,
            self.FORMOSA, self.FUSAN, self.HONG_KONG, self.HYDERABAD,
            self.JAVA, self.KARACHI, self.KYOTO, self.KYUSHU,
            self.MADRAS, self.MALAYA, self.MANCHURIA, self.MANILA,
            self.MECCA, self.MYSORE, self.NANCHANG, self.NEW_GUINEA,
            self.ODESSA, self.OMAN, self.OTARU, self.PERSIA,
            self.PORT_ARTHUR, self.RAJPUTANA, self.RANGOON, self.RUMANIA,
            self.SAKHALIN, self.SARAWAK, self.SEOUL_EAST_COAST, self.SEOUL_WEST_COAST,
            self.SHANGHAI, self.SHIRAZ, self.SINGAPORE, self.SOMALILAND,
            self.SUDAN, self.SUMATRA, self.SYRIA, self.TOKYO,
            self.TONGKING, self.UPPER_BURMA, self.VLADIVOSTOK
        }

        # Define supply centers (full set; we will focus on a subset for the experiment)
        self.SUPPLY_CENTERS = {
            self.ADEN, self.ANGORA, self.ANNAM, self.ASSAM,
            self.BAGHDAD, self.BANGKOK, self.BANGKOK_EAST_COAST, self.BANGKOK_WEST_COAST,
            self.BENGAL, self.BOMBAY, self.BORNEO, self.CANTON,
            self.CEBU, self.CEYLON, self.CHUNGKING, self.COCHIN,
            self.CONSTANTINOPLE, self.DAVAO, self.DELHI, self.EGYPT,
            self.FORMOSA, self.FUSAN, self.HONG_KONG, self.JAVA,
            self.KARACHI, self.KASHGAR, self.KASHMIR, self.KYOTO,
            self.KYUSHU, self.MADRAS, self.MALAYA, self.MANCHURIA,
            self.MANDALAY, self.MANILA, self.MONGOLIA, self.MOSCOW,
            self.NEW_GUINEA, self.ODESSA, self.OMSK, self.OTARU,
            self.PEKING, self.PERSIA, self.PORT_ARTHUR, self.RANGOON,
            self.RUMANIA, self.SAKHALIN, self.SARAWAK, self.SEOUL,
            self.SEOUL_EAST_COAST, self.SEOUL_WEST_COAST, self.SHANGHAI, self.SHIRAZ,
            self.SINGAPORE, self.SINKIANG, self.SUDAN, self.SUMATRA,
            self.TABRIZ, self.TASHKENT, self.TOKYO, self.TONGKING,
            self.UPPER_BURMA, self.VLADIVOSTOK
        }

        # ----------------------------
        # Experiment-specific subset & goals
        # ----------------------------
        self.EXPERIMENT_PROVINCES: List[str] = [
            "Arabian Sea", "Bay of Bengal", "Bengal", "Bombay", "Ceylon", "Delhi",
            "East Indian Ocean", "Gulf of Manaar", "Hyderabad", "Kashmir", "Lucknow",
            "Madras", "Mysore", "Nagpur", "Punjab", "Rajputana", "West Indian Ocean"
        ]

        # robust name->index helper
        def _find_province_index(name: str) -> int:
            """
            Try several matching strategies to find the province index for 'name':
            1) exact match
            2) case-insensitive match
            3) match after lowercasing and removing spaces/underscores/hyphens
            If not found, raise ValueError with a helpful message listing available provinces.
            """
            # exact
            try:
                return self.PROVINCE_NAMES.index(name)
            except ValueError:
                pass
            # case-insensitive
            lname = name.lower()
            for i, pn in enumerate(self.PROVINCE_NAMES):
                if pn.lower() == lname:
                    return i
            # remove spaces/underscores/hyphens for approximate match
            def normalize(s: str) -> str:
                return s.replace(" ", "").replace("_", "").replace("-", "").lower()
            nname = normalize(name)
            for i, pn in enumerate(self.PROVINCE_NAMES):
                if normalize(pn) == nname:
                    return i
            # not found -> helpful error
            available = ", ".join(self.EXPERIMENT_PROVINCES) if hasattr(self, "EXPERIMENT_PROVINCES") else ", ".join(self.PROVINCE_NAMES[:40]) + "..."
            raise ValueError(
                f"Province name '{name}' not found in PROVINCE_NAMES. "
                f"Available example names: (first 40) {', '.join(self.PROVINCE_NAMES[:40])} ... "
                f"Please check spelling/spacing. (Searched for: '{name}')"
            )

        # Convert to indexes for quick lookup using robust helper
        self.EXP_PROVINCE_IDX: List[int] = [_find_province_index(n) for n in self.EXPERIMENT_PROVINCES]

        # Supply centers required for British victory in this experiment
        self.EXPERIMENT_TARGET_SUPPLY_CENTERS = {
            _find_province_index("Bengal"),
            _find_province_index("Bombay"),
            _find_province_index("Ceylon"),
            _find_province_index("Delhi"),
            _find_province_index("Kashmir"),
            _find_province_index("Madras"),
        }

        # Default starting units for Britain (A Delhi, A Madras, F Bombay)
        default_units = [
            {"id": "A_Delhi", "type": "army", "location": int(self.DELHI)},
            {"id": "A_Madras", "type": "army", "location": int(self.MADRAS)},
            {"id": "F_Bombay", "type": "fleet", "location": int(self.BOMBAY)},
        ]
        if initial_units is None:
            initial_units = default_units

        self.units_template = initial_units

        # runtime state
        self.units: List[Dict[str, Any]] = []
        self.unit_counters: Dict[str, Dict[str, int]] = {}
        self.turn_counter = 0  # increments at start of step(); first even-check occurs at turn 2
        self.max_steps = max_steps
        self.truncated = False

        # track supply control: mapping province_idx -> owner (None or "Britain")
        self.supply_control: Dict[int, str] = {idx: None for idx in self.EXPERIMENT_TARGET_SUPPLY_CENTERS}

        # build spaces placeholder (set in reset)
        self.action_space = None
        self.observation_space = None

        # initialize env state
        self.reset()

    # ---------------------------
    # Helper methods
    # ---------------------------
    def _is_water_only(self, province_idx: int) -> bool:
        return province_idx in self.WATER_PROVINCES

    def _is_land_only(self, province_idx: int) -> bool:
        return province_idx in self.LAND_PROVINCES

    def _is_coast(self, province_idx: int) -> bool:
        return province_idx in self.COAST_PROVINCES

    def _is_supply_center(self, province_idx: int) -> bool:
        return province_idx in self.SUPPLY_CENTERS

    def _valid_move_for_unit(self, unit: Dict[str, Any], target: int) -> bool:
        # allow hold
        if target == unit["location"]:
            return True
        # adjacency
        if target not in self.adjacency.get(unit["location"], set()):
            return False
        # army rules: cannot enter water-only provinces
        if unit["type"] == "army":
            if self._is_water_only(target):
                return False
        # fleet rules: cannot enter land-only provinces (must be water or coastal)
        elif unit["type"] == "fleet":
            if self._is_land_only(target):
                return False
            # fleets allowed in water provinces and coastal provinces
        else:
            return False
        return True

    def _build_spaces(self):
        action_space_dict = {}
        obs_space_dict = {}
        for unit in self.units:
            uid = unit["id"]
            action_space_dict[uid] = spaces.Discrete(self.n_provinces + 1)  # 0 hold, 1..n -> target idx = action-1
            obs_space_dict[uid] = spaces.Discrete(self.n_provinces)
        self.action_space = spaces.Dict(action_space_dict)
        self.observation_space = spaces.Dict(obs_space_dict)

    # ---------------------------
    # Gym API
    # ---------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # instantiate units fresh from template
        self.units = []
        self.unit_counters = {}
        for u in self.units_template:
            unit_copy = {"id": u["id"], "type": u["type"], "location": int(u["location"])}
            self.units.append(unit_copy)
            # init counters (we might track special counters later)
            self.unit_counters[unit_copy["id"]] = {}

        # reset turn counter and supply control
        self.turn_counter = 0
        self.truncated = False
        self.supply_control = {idx: None for idx in self.EXPERIMENT_TARGET_SUPPLY_CENTERS}

        # build spaces now that units exist
        self._build_spaces()

        obs = {u["id"]: int(u["location"]) for u in self.units}
        info = {"turn": self.turn_counter, "controlled_centers": 0}
        return obs, info

    def step(self, action: Dict[str, int]):
        """
        action: dict mapping unit id -> integer action
         0 -> hold
         1..n_provinces -> move to province (action-1 is target)
        """
        assert isinstance(action, dict), "Action must be a dict mapping unit ids to integers."
        assert self.action_space is not None, "Call reset() before step()."
        for uid in self.action_space.spaces.keys():
            if uid not in action:
                raise KeyError(f"Missing action for unit '{uid}'")

        # increment turn counter at start of step; first even-check will be at turn 2
        self.turn_counter += 1

        # apply moves (independent, no conflicts resolution for now)
        info: Dict[str, Any] = {"invalid_actions": {}}
        for unit in self.units:
            uid = unit["id"]
            a = int(action[uid])
            if a == 0:
                target = unit["location"]
            else:
                target = int(a - 1)

            if self._valid_move_for_unit(unit, target):
                unit["location"] = target
            else:
                info["invalid_actions"][uid] = {"attempted_action": a, "attempted_target": target}
                # invalid: unit stays in place

        # After all moves, if this is an even-numbered turn, evaluate supply control
        # (first evaluation occurs at end of turn 2)
        reward = 0.0
        terminated = False
        if self.turn_counter % 2 == 0:
            controlled_count = 0
            # For each target supply center, check whether any British unit occupies it now
            for sc_idx in self.EXPERIMENT_TARGET_SUPPLY_CENTERS:
                owner = None
                for unit in self.units:
                    if unit["location"] == sc_idx:
                        owner = "Britain"
                        break
                self.supply_control[sc_idx] = owner
                if owner == "Britain":
                    controlled_count += 1

            # If Britain controls >=6 targets, victory
            if controlled_count >= 6:
                terminated = True
                reward = 1.0

            info["controlled_centers"] = controlled_count
        else:
            # odd turns: do not change supply_control (ownership persists until next even evaluation)
            info["controlled_centers"] = sum(1 for v in self.supply_control.values() if v == "Britain")

        # truncated by max steps
        truncated = False
        if self.turn_counter >= self.max_steps and not terminated:
            truncated = True
            self.truncated = True

        obs = {u["id"]: int(u["location"]) for u in self.units}
        info.update({"turn": self.turn_counter, "supply_control": {self.PROVINCE_NAMES[idx]: owner for idx, owner in self.supply_control.items()}})
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---------------------------
    # Renderer (ASCII) - shows only experiment provinces
    # ---------------------------
    def render(self, mode="human"):
        p = self.PROVINCE_NAMES
        exp_idx_set: Set[int] = set(self.EXP_PROVINCE_IDX)
        # header
        print("\n" + "=" * 80)
        print(f"Turn {self.turn_counter}  (Even-turn control checks on turns 2,4,6...)")
        print("Britain units:")
        for unit in self.units:
            print(f"  {unit['id']:8s} ({unit['type']}) -> {p[unit['location']]} (idx {unit['location']})")
        print("-" * 80)

        # Print table of experiment provinces: name | type | unit present?
        def province_type(idx: int) -> str:
            if idx in self.WATER_PROVINCES:
                return "water"
            if idx in self.COAST_PROVINCES:
                return "coast"
            if idx in self.LAND_PROVINCES:
                return "land"
            # default fallback
            return "land/coast"

        # Build rows
        rows = []
        for name in self.EXPERIMENT_PROVINCES:
            idx = self.PROVINCE_NAMES.index(name) if name in self.PROVINCE_NAMES else next((i for i in range(len(self.PROVINCE_NAMES)) if self.PROVINCE_NAMES[i].lower() == name.lower()), None)
            if idx is None:
                # fallback: skip if not found
                continue
            ptype = province_type(idx)
            present_units = [f"{u['id']}" for u in self.units if u["location"] == idx]
            unit_str = ", ".join(present_units) if present_units else ""
            owner = self.supply_control.get(idx, None)
            owner_str = owner if owner is not None else ""
            rows.append((name, ptype, unit_str, owner_str))

        # Print rows neatly
        print(f"{'Province':20s} | {'Type':6s} | {'Units present':20s} | {'Controlled by':12s}")
        print("-" * 80)
        for name, ptype, unit_str, owner_str in rows:
            print(f"{name:20s} | {ptype:6s} | {unit_str:20s} | {owner_str:12s}")
        print("-" * 80)

        # Summary of control
        controlled = sum(1 for v in self.supply_control.values() if v == "Britain")
        print(f"Controlled experiment supply centers: {controlled} / {len(self.EXPERIMENT_TARGET_SUPPLY_CENTERS)}")
        print("=" * 80 + "\n")

    def close(self):
        pass


if __name__ == "__main__":
    # Demo: run a short sequence showing the experiment behavior
    env = ColonialDiplomacyEnv(max_steps=20)
    obs, info = env.reset()
    env.render()  # Step 0 shown

    # Example actions — agent moves toward some supply centers.
    # Actions are dicts mapping unit id -> integer action (0 = hold, 1..N => target = action-1)
    # For readability here we compute actions by using province name indexes.
    def normalize_action_dict(env, act_partial):
        full = {}
        for unit in env.units:
            full[unit["id"]] = int(act_partial.get(unit["id"], 0))
        return full

    # Build a simple sequence (illustrative)
    seq_actions = [
        # Turn 1 (odd)
        {"A_Delhi": env.DELHI + 1, "A_Madras": env.MADRAS + 1, "F_Bombay": env.BOMBAY + 1},
        # Turn 2 (even) - hold to capture
        {"A_Delhi": env.DELHI + 1, "A_Madras": env.MADRAS + 1, "F_Bombay": env.BOMBAY + 1},
        # Turn 3 (odd)
        {"A_Delhi": env.PUNJAB, "A_Madras": env.HYDERABAD, "F_Bombay": env.ARABIAN_SEA},
        # Turn 4 (even) - hold to capture
        {"A_Delhi": env.KASHMIR, "A_Madras": env.BENGAL, "F_Bombay": env.WEST_INDIAN_OCEAN},
        # Turn 5 (odd)
        {"A_Delhi": env.KASHMIR + 1, "A_Madras": env.BENGAL + 1, "F_Bombay": env.CEYLON},
        # Turn 6 (even) - hold to capture
        {"A_Delhi": env.KASHMIR + 1, "A_Madras": env.BENGAL + 1, "F_Bombay": env.CEYLON + 1}
    ]

    for a_partial in seq_actions:
        action = normalize_action_dict(env, a_partial)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        if terminated or truncated:
            break

    env.close()