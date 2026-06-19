import abc
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Dict, Optional


class SyntheticTask(abc.ABC):
    """Base class for all synthetic tasks."""

    @abc.abstractmethod
    def generate_samples(self, n_samples: int) -> List[Dict[str, str]]:
        """Return a list of dicts with 'input' and 'output' keys."""
        pass

    def save_samples(self, samples: List[Dict[str, str]], path: str) -> None:
        """Save samples to a JSONL file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")


def tokenize(
    jsonl_path: str,
    output_path: str,
    tokenizer_name_or_path: str,
    max_length: int = 512,
    chunk_size: int = 2000,
) -> None:
    """Tokenize a JSONL file of {input, output} samples and save as a .pt file.

    Input tokens are masked in labels (-100); output tokens carry their token id.
    The saved file contains: input_ids, attention_mask, labels.

    Output tensors are pre-allocated and filled in chunks rather than built from
    per-sample Python lists, so peak memory stays close to the size of the final
    tensors (matters for large datasets — the list-of-lists approach used ~3-4x
    more and OOMs on memory-capped nodes).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos = tokenizer.eos_token or ""

    # Read just the text fields. The answer carries a trailing eos so the model
    # learns to stop; it is tokenized without special tokens.
    inputs: List[str] = []
    answers: List[str] = []
    with open(jsonl_path) as f:
        for line in f:
            s = json.loads(line)
            inputs.append(s["input"])
            answers.append(s["output"] + eos)

    n = len(inputs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((n, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_length), dtype=torch.long)
    labels = torch.full((n, max_length), -100, dtype=torch.long)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        in_enc = tokenizer(inputs[start:end], add_special_tokens=True)["input_ids"]
        out_enc = tokenizer(answers[start:end], add_special_tokens=False)["input_ids"]
        for j, (ie, oe) in enumerate(zip(in_enc, out_enc)):
            row = start + j
            ids = (ie + oe)[:max_length]
            L = len(ids)
            input_ids[row, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :L] = 1
            # labels: -100 over the (input + pad) tokens, token ids over the
            # answer tokens that survived truncation.
            ans_start = len(ie)
            if ans_start < L:
                labels[row, ans_start:L] = torch.tensor(ids[ans_start:L], dtype=torch.long)

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved {n} tokenized samples to {output_path}")


class TokenizedDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]
        self.teacher_logits = data.get("teacher_logits", None)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
        if self.teacher_logits is not None:
            item["teacher_logits"] = self.teacher_logits[idx]
        return item


def get_loader(
    file_paths: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader from one or more tokenized .pt files.

    If multiple paths are given the datasets are concatenated. A .pt file may
    optionally contain a 'teacher_logits' tensor for distillation training.
    """
    datasets = []
    for path in file_paths:
        data = torch.load(path, weights_only=True)
        datasets.append(TokenizedDataset(data))

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def merge_tokenized_files(
    file_paths: List[str],
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Merge tokenized .pt files (each a dict of tensors) into one dict.

    Each input file is loaded with `torch.load` and must be a dict whose values
    are tensors that share a leading sample dimension (e.g. input_ids,
    attention_mask, labels, optionally teacher_logits). All files must contain
    the same set of keys and matching trailing dimensions per key; tensors are
    concatenated along dim 0.

    If *output_path* is given the merged dict is saved there.
    """
    if not file_paths:
        raise ValueError("file_paths must contain at least one path")

    loaded = [torch.load(p, weights_only=True) for p in file_paths]

    keys = set(loaded[0].keys())
    for path, d in zip(file_paths[1:], loaded[1:]):
        if set(d.keys()) != keys:
            raise ValueError(
                f"Key mismatch: {file_paths[0]} has {sorted(keys)}, "
                f"{path} has {sorted(d.keys())}"
            )

    merged: Dict[str, torch.Tensor] = {
        k: torch.cat([d[k] for d in loaded], dim=0) for k in keys
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, output_path)
        n = next(iter(merged.values())).shape[0]
        print(f"Saved {n} merged samples to {output_path}")

    return merged


# ── shared vocabulary pools ───────────────────────────────────────────────────

ADJECTIVES: List[str] = [
    "ancient", "angry", "anxious", "arctic", "artificial", "atomic",
    "bamboo", "baroque", "battered", "broken", "bronze", "burning",
    "ceramic", "chemical", "chrome", "circular", "classical", "clinical",
    "cloudy", "coastal", "colonial", "colossal", "copper", "cosmic",
    "crystal", "cubic", "curved", "damaged", "dark", "digital",
    "distant", "domestic", "dramatic", "dusty", "dynamic", "early",
    "eastern", "elastic", "electric", "elegant", "emerald", "empty",
    "enormous", "eternal", "exotic", "experimental", "extreme", "faded",
    "famous", "fancy", "fermented", "fictional", "fierce", "flaming",
    "flexible", "floating", "floral", "forgotten", "formal", "frozen",
    "fuzzy", "geometric", "giant", "gilded", "glacial", "glowing",
    "golden", "gothic", "grand", "granite", "green", "hollow",
    "hybrid", "icy", "industrial", "infrared", "invisible", "iron",
    "isolated", "jagged", "jade", "jumbo", "juvenile", "kinetic",
    "large", "lateral", "liquid", "local", "lunar", "magnetic",
    "mechanical", "medical", "mega", "melted", "metallic", "miniature",
    "modern", "molecular", "molten", "mossy", "mysterious", "narrow",
    "neon", "neural", "nocturnal", "northern", "nuclear", "obsidian",
    "oceanic", "optical", "orbital", "organic", "ornate", "oval",
    "parallel", "plastic", "polished", "portable", "primary", "prismatic",
    "quantum", "radiant", "radioactive", "rapid", "rectangular", "remote",
    "rigid", "rocky", "rotating", "rough", "rusty", "sacred",
    "sandy", "serene", "shattered", "silent", "silver", "skeletal",
    "smooth", "solar", "sonic", "spectral", "spiral", "static",
    "stellar", "stone", "stormy", "submerged", "synthetic", "thermal",
    "titanium", "toxic", "tropical", "turbulent", "twisted", "ultrasonic",
    "underground", "urban", "vibrant", "vintage", "volcanic", "western",
    "wooden", "yellow", "zenith", "zigzag",
]

# Union of nouns previously defined in s5.py and memorization.py
NOUNS: List[str] = [
    # Animals – mammals
    "cat", "dog", "horse", "cow", "sheep", "pig", "rabbit", "fox",
    "wolf", "bear", "lion", "tiger", "leopard", "elephant", "giraffe",
    "zebra", "hippo", "rhino", "camel", "llama", "deer", "moose",
    "bison", "otter", "beaver", "squirrel", "raccoon", "skunk", "bat",
    "panda", "koala", "kangaroo", "monkey", "gorilla", "dolphin", "whale",
    "seal", "walrus", "mole", "hedgehog", "chipmunk", "porcupine",
    "donkey", "mule", "pony", "buffalo", "antelope", "gazelle", "jaguar",
    "cheetah", "cougar", "baboon", "orangutan", "chimpanzee",
    # Animals – birds
    "eagle", "hawk", "falcon", "owl", "raven", "crow", "pigeon",
    "seagull", "pelican", "flamingo", "penguin", "peacock", "ostrich",
    "parrot", "toucan", "robin", "sparrow", "cardinal", "finch",
    "hummingbird", "woodpecker", "bluebird", "canary", "turkey",
    "goose", "duck", "swan", "stork", "crane",
    # Animals – aquatic / reptile / insect
    "shark", "salmon", "tuna", "goldfish", "clownfish", "octopus",
    "squid", "jellyfish", "starfish", "crab", "lobster", "shrimp",
    "turtle", "tortoise", "snake", "lizard", "frog", "toad",
    "crocodile", "alligator", "salamander", "chameleon",
    "ant", "bee", "butterfly", "dragonfly", "firefly", "grasshopper",
    "ladybug", "spider", "scorpion", "centipede", "caterpillar",
    # Household / furniture / tools
    "table", "chair", "sofa", "lamp", "desk", "shelf", "cabinet",
    "mirror", "clock", "vase", "pot", "pan", "bowl", "plate", "cup",
    "mug", "bottle", "jar", "box", "basket", "bag", "pillow",
    "blanket", "broom", "hammer", "nail", "wrench", "scissors",
    "brush", "ruler", "pen", "pencil", "book", "notebook", "stapler",
    "ladder", "bucket", "mop", "towel", "curtain", "rug",
    # Technology / transport
    "phone", "computer", "tablet", "camera", "television", "radio",
    "microphone", "speaker", "keyboard", "printer", "battery",
    "charger", "cable", "engine", "motor", "gear", "spring", "wheel",
    "axle", "magnet", "circuit", "chip", "satellite", "rocket",
    "car", "truck", "bus", "train", "plane", "boat", "ship",
    "bicycle", "motorcycle", "helicopter", "kayak", "canoe",
    "skateboard", "scooter", "submarine",
    # Nature objects
    "rock", "stone", "pebble", "sand", "mud", "leaf", "flower",
    "tree", "branch", "root", "seed", "mushroom", "moss", "vine",
    "coral", "shell", "pearl", "crystal", "diamond", "ruby",
    "emerald", "sapphire", "coal", "volcano", "mountain", "river",
    "lake", "ocean", "island", "forest", "desert", "meadow",
    "glacier", "canyon", "cave", "marsh",
    # Food items
    "apple", "banana", "orange", "grape", "strawberry", "watermelon",
    "mango", "cherry", "peach", "pear", "lemon", "coconut", "avocado",
    "tomato", "potato", "carrot", "broccoli", "lettuce", "onion",
    "garlic", "pepper", "cucumber", "pumpkin", "corn", "bean",
    "rice", "bread", "cake", "cookie", "chocolate", "candy",
    "cheese", "butter", "egg", "honey", "sugar", "salt",
    # Miscellaneous
    "ball", "balloon", "kite", "doll", "puzzle", "coin", "key",
    "lock", "chain", "rope", "ribbon", "needle", "button", "ring",
    "tube", "pipe", "wire", "net", "flag", "stamp", "medal",
    "trophy", "crown", "sword", "shield", "anchor", "sail", "oar",
    "candle", "torch", "lantern", "lens", "prism", "globe", "map",
    "compass", "thermometer", "barometer", "telescope", "microscope",
    # Memorization-task extras (objects, structures, instruments)
    "anvil", "arrow", "axe", "badge", "barrel", "beacon", "bell",
    "blade", "bolt", "bone", "boulder", "bridge", "cage", "cannon",
    "canvas", "capsule", "cart", "cask", "castle", "chamber", "chest",
    "chimney", "chisel", "cloak", "coil", "column", "cone",
    "container", "cord", "cube", "cylinder", "dagger", "dial", "dome",
    "door", "drum", "envelope", "flask", "frame", "furnace", "gate",
    "gem", "goblet", "hatch", "helmet", "hook", "horn", "jewel",
    "latch", "lattice", "lever", "machine", "mantle", "mask",
    "medallion", "module", "monument", "mural", "node", "orb",
    "panel", "pedestal", "pendant", "pillar", "pin", "platform",
    "portal", "probe", "pulley", "relic", "rudder", "shard", "signal",
    "slab", "slate", "sphere", "spike", "statue", "switch", "tank",
    "terminal", "throne", "tile", "token", "tower", "trap", "trigger",
    "turret", "valve", "vault", "vessel", "vial", "wave", "wedge",
    "wick", "scroll", "plaque", "cradle",
]

NAMES: List[str] = [
    # Female names
    "Alice", "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Charlotte",
    "Mia", "Amelia", "Harper", "Evelyn", "Abigail", "Emily", "Elizabeth",
    "Mila", "Ella", "Avery", "Sofia", "Camila", "Aria", "Scarlett",
    "Victoria", "Madison", "Luna", "Grace", "Chloe", "Penelope", "Layla",
    "Riley", "Zoey", "Nora", "Lily", "Eleanor", "Hannah", "Lillian",
    "Addison", "Aubrey", "Ellie", "Stella", "Natalie", "Zoe", "Leah",
    "Hazel", "Violet", "Aurora", "Savannah", "Audrey", "Brooklyn", "Bella",
    "Claire", "Skylar", "Lucy", "Paisley", "Everly", "Anna", "Caroline",
    "Nova", "Genesis", "Emilia", "Kennedy", "Samantha", "Maya", "Willow",
    "Kinsley", "Naomi", "Aaliyah", "Elena", "Sarah", "Ariana", "Allison",
    "Gabriella", "Madeline", "Cora", "Ruby", "Eva", "Serenity",
    "Autumn", "Adeline", "Hailey", "Gianna", "Valentina", "Isla", "Eliana",
    "Quinn", "Nevaeh", "Ivy", "Sadie", "Piper", "Lydia", "Alexa",
    "Josephine", "Emery", "Julia", "Delilah", "Arianna", "Vivian", "Kaylee",
    "Sophie", "Brielle", "Madelyn", "Hadley", "Jade", "Katherine",
    "Isabel", "Natalia", "Raelynn", "Jasmine", "Juliette", "Lila", "Faith",
    "Kayla", "Adriana", "Juliana", "Molly", "Alyssa", "Melody", "Diana",
    "Daisy", "Margaret", "Leilani", "Mikayla", "Kylie", "Chelsea",
    "Lacey", "Ximena", "Aliyah", "Francesca", "Clara", "Bianca", "Amber",
    "Freya", "Nina", "Cecilia", "Katharine", "Sienna", "Miriam", "Selena",
    "Brooke", "Crystal", "Cassidy", "Rebekah", "Felicity", "Ingrid",
    "Rosemary", "Melissa", "Theresa", "Wren", "Arabella", "Esme",
    "Genevieve", "Harriet", "Imogen", "Juniper", "Lena", "Madeleine",
    "Ophelia", "Rosalie", "Sylvia", "Tatiana", "Uma",
    # Male names
    "Liam", "Noah", "Oliver", "Elijah", "William", "James", "Benjamin",
    "Lucas", "Henry", "Alexander", "Mason", "Ethan", "Daniel", "Jacob",
    "Logan", "Jackson", "Sebastian", "Jack", "Aiden", "Owen", "Samuel",
    "Joseph", "John", "David", "Wyatt", "Matthew", "Luke", "Asher",
    "Carter", "Julian", "Grayson", "Leo", "Jayden", "Gabriel", "Isaac",
    "Lincoln", "Anthony", "Hudson", "Dylan", "Ezra", "Thomas", "Charles",
    "Christopher", "Jaxon", "Maverick", "Josiah", "Isaiah", "Andrew",
    "Elias", "Joshua", "Nathan", "Caleb", "Ryan", "Adrian", "Miles",
    "Eli", "Nolan", "Christian", "Aaron", "Cameron", "Ezekiel", "Colton",
    "Levi", "Landon", "Roman", "Damian", "Connor", "Brayden", "Theodore",
    "Declan", "Jose", "Micah", "Greyson", "Maxwell", "Adam", "Ian",
    "Nicholas", "Dominic", "Austin", "Everett", "Bryson",
    "Xavier", "Carson", "Jace", "Cooper", "Bentley", "Sawyer", "Weston",
    "Tristan", "Silas", "George", "Ryder", "Emmett", "Harrison", "Braxton",
    "Jason", "Kingston", "Robert", "Jameson", "Brandon", "Easton", "Beckham",
    "Rowan", "Peyton", "Kevin", "Rylan", "Tyler", "Wesley", "Zachary",
    "Vincent", "Patrick", "Giovanni", "Marcus", "Bradley", "Eric",
    "Jonah", "Felix", "August", "Xander", "Rhett", "Kyler", "Arthur",
    "Adriel", "Arlo", "Atticus", "Barrett", "Bowen", "Cade", "Caiden",
    "Calvin", "Dawson", "Dean", "Drake", "Drew", "Elliot", "Emmitt",
    "Finn", "Fletcher", "Ford", "Grant", "Gunner", "Hayden", "Hugh",
    "Hunter", "Ivan", "Jaden", "Joel", "Justin", "Kaiden", "Keegan",
    "Kendrick", "Knox", "Kyle", "Luca", "Malcolm", "Marshall", "Mitchell",
    "Morgan", "Murphy", "Nathaniel", "Neil", "Nelson", "Oscar", "Otto",
    "Pierce", "Preston", "Rafael", "Reece", "Reese", "Reid", "Rex",
    "Rhys", "Ridge", "Roland", "Russell", "Scott", "Seth", "Shawn",
    "Spencer", "Tanner", "Timothy", "Todd", "Travis", "Trevor", "Troy",
    "Tucker", "Turner", "Ty", "Tyson", "Wade", "Warren", "Wayne",
    "Wilson", "Yusuf", "Zane", "Zayden", "Zion",
    # Gender-neutral names
    "Alex", "Jordan", "Taylor", "Casey",
    "Parker", "Blake", "Skyler", "River",
    "Phoenix", "Robin", "Sam", "Charlie", "Finley", "Kendall", "Jamie",
    "Emerson", "Harley", "Indigo", "Jesse", "Jules", "Kai",
    "Lake", "Lane", "Lennon", "Oakley", "Presley", "Remy", "Rory",
    "Scout", "Shiloh", "Sutton", "Tatum", "Teagan",
    "Devon", "Ellery", "Fallon", "Francis",
    "Gray", "Harbor", "Haven", "Hollis", "Honor",
    "Merritt", "Milan", "Monroe", "Noel", "North",
    "Ocean", "Pax", "Poet", "Prairie", "Rain", "Rebel", "Remi",
    "Sage", "Salem", "Sloane", "Sol", "Story", "Sunday",
    "Sunny", "Sunniva", "Tempest", "True", "Vale", "Vesper", "Winter",
    "Zephyr",
    # East Asian names (Chinese, Japanese, Korean, Vietnamese)
    "Hiroshi", "Kenji", "Akira", "Sakura", "Haruki", "Yuki", "Daichi",
    "Mei", "Lin", "Wei", "Jing", "Bo", "Tao", "Ling",
    "Xinyi", "Junho", "Minjun", "Soyeon", "Haeun", "Jihoon", "Yuna",
    "Linh", "Minh", "Bao", "Thanh",
    # South Asian names
    "Aarav", "Arjun", "Vihaan", "Ishaan", "Kabir", "Reyansh", "Aanya",
    "Saanvi", "Diya", "Priya", "Anika", "Meera", "Aditi", "Anushka",
    "Kavya",
    # Middle Eastern names (Arabic, Persian, Hebrew)
    "Amir", "Omar", "Zaid", "Karim", "Hassan", "Tariq", "Fatima",
    "Aisha", "Yasmin", "Mariam", "Cyrus", "Darius", "Roya", "Eitan",
    "Noa",
    # African names (Swahili, Yoruba, Igbo, Akan)
    "Kwame", "Kofi", "Jelani", "Chidi", "Amara", "Zuri", "Ayo",
    "Femi", "Sade", "Imani",
    # Southeast Asian and Pacific names
    "Dewi", "Indra", "Putri", "Niran", "Apsara", "Anh",
    "Moana", "Keanu", "Nalu",
    # Mesoamerican
    "Itzel",
]

assert len(NAMES) == len(set(NAMES)) == 500, (
    f"NAMES must contain 500 unique entries; got {len(NAMES)} ({len(set(NAMES))} unique)"
)
assert len(NOUNS) == len(set(NOUNS)), "NOUNS must be unique"
assert len(ADJECTIVES) == len(set(ADJECTIVES)), "ADJECTIVES must be unique"
