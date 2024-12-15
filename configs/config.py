dataset_yaml = "datasets.yaml"
entity_gender_metadata = "data/metadata/gender_info_final.csv"
pronoun_dialogue_metadata = "data/metadata/pron_dialogue_info.csv"

PRONOUNS_GROUPS = {
    "i": 0,
    "me": 0,
    "my": 0,  ## my is a appropostrophe form
    "mine": 0,
    "myself": 0,
    "i.": 0,
    "you": 1,
    "your": 1,
    "yours": 1,
    "yourself": 1,
    "yourselves": 1,
    "thou": 1,
    "thee": 1,
    "thy": 1,
    "thine": 1,
    "thyself": 1,
    "ye": 1,
    "he": 2,
    "him": 2,
    "his": 2,
    "himself": 2,
    "she": 3,
    "her": 3,
    "hers": 3,
    "herself": 3,
    "it": 4,
    "its": 4,
    "itself": 4,
    "we": 5,
    "us": 5,
    "our": 5,
    "ours": 5,
    "ourselves": 5,
    "they": 6,
    "them": 6,
    "their": 6,
    "themselves": 6,
    "that": 7,
    "this": 7,
}

MAX_ENTITIES = 5
MIN_MENTIONS = 5

PERSONAL_PRONOUNS = [
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "i.",
]

POSSESIVE_PRONOUNS_APPOS = [
    "my",
    "your",
    "thy",
    "thine",
    "his",
    "her",
    "its",
    "our",
    "their",
    "thy",
]

PLURAL_PRONOUNS = [
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
]  ## Removing plural pronouns except "you"

selected_keys = [
    "mentions_vs_mentionstr",
    "mentions_vs_clusters",
    "mentions_vs_tbound",
]
