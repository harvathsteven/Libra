#!/usr/bin/env python3
"""Build Phase 2 stimuli with fixed seeds. Safe to re-run."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STIM = ROOT / "stimuli"
SEED = 42

# Goldberg (1992) 100 unipolar markers — Factor I / II poles via psych::bfi.adjectives.keys
# plus Saucier (1994) Mini-Markers terms to reach >=16 unique adjectives per pole.
# Sources logged in DECISIONS.md.
GOLDBERG_E_HIGH = [
    "active", "assertive", "bold", "daring", "energetic",
    "extraverted", "talkative", "unrestrained", "vigorous", "warm",
]
GOLDBERG_E_LOW = [
    "bashful", "inhibited", "introverted", "quiet", "reserved",
    "shy", "timid", "unadventurous", "untalkative", "withdrawn",
]
GOLDBERG_A_HIGH = [
    "agreeable", "considerate", "cooperative", "generous", "helpful",
    "kind", "pleasant", "sympathetic", "trustful", "warmhearted",
]
GOLDBERG_A_LOW = [
    "cold", "distrustful", "harsh", "rude", "uncharitable",
    "uncooperative", "unkind", "unsympathetic", "selfish", "demanding",
]
# Saucier (1994) Mini-Markers + additional Goldberg (1992) bipolar-pole adjectives
SAUCIER_E_HIGH = [
    "extraverted", "talkative", "bold", "energetic", "sociable", "outgoing",
    "enthusiastic", "spontaneous", "adventurous", "verbal", "expressive", "confident",
]
SAUCIER_E_LOW = [
    "bashful", "quiet", "shy", "withdrawn", "silent", "reserved",
    "solitary", "passive", "meek", "lethargic", "aloof", "retiring",
]
SAUCIER_A_HIGH = [
    "kind", "cooperative", "sympathetic", "warm", "courteous", "charitable",
    "softhearted", "tolerant", "forgiving", "accommodating", "compassionate", "cordial",
]
SAUCIER_A_LOW = [
    "cold", "harsh", "rude", "unsympathetic", "quarrelsome", "faultfinding",
    "stubborn", "crude", "antagonistic", "critical", "ruthless", "combative",
]


def unique_extend(base: list[str], extra: list[str], n: int = 16) -> list[str]:
    out: list[str] = []
    for w in base + extra:
        w = w.lower().strip()
        if w not in out:
            out.append(w)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"Need {n} adjectives, got {len(out)}: {out}")
    return out[: max(n, len(out))]  # keep all unique up to available; require >=n


def split_half(words: list[str], rng: random.Random) -> tuple[list[str], list[str]]:
    items = list(words)
    rng.shuffle(items)
    mid = len(items) // 2
    return sorted(items[:mid]), sorted(items[mid:])


def build_adjectives() -> dict:
    rng = random.Random(SEED)
    poles = {
        "extraversion": {
            "high": unique_extend(GOLDBERG_E_HIGH, SAUCIER_E_HIGH, 16),
            "low": unique_extend(GOLDBERG_E_LOW, SAUCIER_E_LOW, 16),
        },
        "agreeableness": {
            "high": unique_extend(GOLDBERG_A_HIGH, SAUCIER_A_HIGH, 16),
            "low": unique_extend(GOLDBERG_A_LOW, SAUCIER_A_LOW, 16),
        },
    }
    out: dict = {
        "meta": {
            "sources": [
                "Goldberg (1992) Big Five unipolar markers",
                "Saucier (1994) Mini-Markers (supplement to reach n>=16/pole)",
            ],
            "seed": SEED,
        }
    }
    for trait, poles_map in poles.items():
        out[trait] = {}
        for pole, words in poles_map.items():
            extraction, test = split_half(words, rng)
            assert set(extraction).isdisjoint(set(test))
            out[trait][pole] = {"extraction": extraction, "test": test, "all": sorted(words)}
    return out


# IPIP-NEO-120 Extraversion + Agreeableness from ipip.ori.org (Johnson, 2014), public domain.
IPIP_ITEMS = [
    # E1 Friendliness
    ("E", "E1", "Make friends easily.", "+"),
    ("E", "E1", "Feel comfortable around people.", "+"),
    ("E", "E1", "Avoid contacts with others.", "-"),
    ("E", "E1", "Keep others at a distance.", "-"),
    # E2 Gregariousness
    ("E", "E2", "Love large parties.", "+"),
    ("E", "E2", "Talk to a lot of different people at parties.", "+"),
    ("E", "E2", "Prefer to be alone.", "-"),
    ("E", "E2", "Avoid crowds.", "-"),
    # E3 Assertiveness
    ("E", "E3", "Take charge.", "+"),
    ("E", "E3", "Try to lead others.", "+"),
    ("E", "E3", "Take control of things.", "+"),
    ("E", "E3", "Wait for others to lead the way.", "-"),
    # E4 Activity Level
    ("E", "E4", "Am always busy.", "+"),
    ("E", "E4", "Am always on the go.", "+"),
    ("E", "E4", "Do a lot in my spare time.", "+"),
    ("E", "E4", "Like to take it easy.", "-"),
    # E5 Excitement-Seeking
    ("E", "E5", "Love excitement.", "+"),
    ("E", "E5", "Seek adventure.", "+"),
    ("E", "E5", "Enjoy being reckless.", "+"),
    ("E", "E5", "Act wild and crazy.", "+"),
    # E6 Cheerfulness
    ("E", "E6", "Radiate joy.", "+"),
    ("E", "E6", "Have a lot of fun.", "+"),
    ("E", "E6", "Love life.", "+"),
    ("E", "E6", "Look at the bright side of life.", "+"),
    # A1 Trust
    ("A", "A1", "Trust others.", "+"),
    ("A", "A1", "Believe that others have good intentions.", "+"),
    ("A", "A1", "Trust what people say.", "+"),
    ("A", "A1", "Distrust people.", "-"),
    # A2 Morality (all reverse-keyed on IPIP page)
    ("A", "A2", "Use others for my own ends.", "-"),
    ("A", "A2", "Cheat to get ahead.", "-"),
    ("A", "A2", "Take advantage of others.", "-"),
    ("A", "A2", "Obstruct others' plans.", "-"),
    # A3 Altruism
    ("A", "A3", "Am concerned about others.", "+"),
    ("A", "A3", "Love to help others.", "+"),
    ("A", "A3", "Am indifferent to the feelings of others.", "-"),
    ("A", "A3", "Take no time for others.", "-"),
    # A4 Cooperation (all reverse)
    ("A", "A4", "Love a good fight.", "-"),
    ("A", "A4", "Yell at people.", "-"),
    ("A", "A4", "Insult people.", "-"),
    ("A", "A4", "Get back at others.", "-"),
    # A5 Modesty (all reverse)
    ("A", "A5", "Believe that I am better than others.", "-"),
    ("A", "A5", "Think highly of myself.", "-"),
    ("A", "A5", "Have a high opinion of myself.", "-"),
    ("A", "A5", "Boast about my virtues.", "-"),
    # A6 Sympathy
    ("A", "A6", "Sympathize with the homeless.", "+"),
    ("A", "A6", "Feel sympathy for those who are worse off than myself.", "+"),
    ("A", "A6", "Am not interested in other people's problems.", "-"),
    ("A", "A6", "Try not to think about the needy.", "-"),
]


EXTRACTION_QUESTIONS = [
    "How should leftover bread be stored overnight?",
    "What causes sea tides on Earth?",
    "How does a refrigerator remove heat from its interior?",
    "What is the difference between a planet and a moon?",
    "How is paper made from wood pulp?",
    "Why do leaves change color in autumn?",
    "How does a compass find north?",
    "What is condensation in the water cycle?",
    "How do bees make honey?",
    "What makes a rainbow appear?",
    "How does a bicycle gear system work?",
    "Why does ice float on water?",
    "How is glass manufactured?",
    "What is the function of roots in a plant?",
    "How does a thermostat regulate temperature?",
    "What causes an echo?",
    "How do solar panels generate electricity?",
    "Why do we see phases of the moon?",
    "How does yeast make bread rise?",
    "What is the purpose of a catalyst in chemistry?",
]

PROBE_QUESTIONS = [
    "How should I season cast-iron cookware?",
    "What is the difference between weather and climate?",
    "How do noise-cancelling headphones work?",
    "Why do bridges have expansion joints?",
    "How is olive oil extracted from olives?",
    "What causes static electricity on clothing?",
    "How does a microwave oven heat food?",
    "Why do metal objects feel colder than wood at room temperature?",
    "How is drinking water treated at a municipal plant?",
    "What is the difference between a virus and a bacterium?",
    "How do GPS satellites determine a location?",
    "Why does baking soda react with vinegar?",
    "How is cement different from concrete?",
    "What causes fog to form near lakes?",
    "How does a zipper mechanism lock teeth together?",
    "Why do onions make people tear up?",
    "How is wool processed into yarn?",
    "What is the role of mitochondria in a cell?",
    "How do double-paned windows improve insulation?",
    "Why does salt melt ice on roads?",
    "How is stainless steel resistant to rust?",
    "What causes thunder after lightning?",
    "How does a septic tank process wastewater?",
    "Why do apples brown after being cut?",
    "How is charcoal produced from wood?",
    "What is the difference between AC and DC electricity?",
    "How do earplugs reduce sound intensity?",
    "Why does a full kettle whistle?",
    "How is maple syrup made from sap?",
    "What causes sinkholes to form?",
    "How does bleach whiten fabrics?",
    "Why do balloons float when filled with helium?",
    "How is recycled plastic sorted at a facility?",
    "What is the difference between a hurricane and a tornado?",
    "How do mechanical clocks keep time?",
    "Why does metal expand when heated?",
    "How is flour milled from wheat?",
    "What causes the smell after rain on dry ground?",
    "How do vacuum cleaners create suction?",
    "Why is the sky blue during the day?",
    "How is tea different from herbal infusions?",
    "What causes frost on windows in winter?",
    "How do smoke detectors sense particles?",
    "Why does popcorn pop when heated?",
    "How is asphalt applied to roads?",
    "What is the difference between a frog and a toad?",
    "How do water filters remove sediment?",
    "Why do batteries eventually lose charge?",
    "How is cork harvested from trees?",
    "What causes dew to form on grass overnight?",
]


LEVELS = ["very_low", "somewhat_low", "somewhat_high", "very_high"]
LEVEL_WORDS = {
    "very_low": "very",
    "somewhat_low": "somewhat",
    "somewhat_high": "somewhat",
    "very_high": "very",
}
LEVEL_POLE = {
    "very_low": "low",
    "somewhat_low": "low",
    "somewhat_high": "high",
    "very_high": "high",
}
LEVEL_NUMERIC = {
    "very_low": 1,
    "somewhat_low": 2,
    "somewhat_high": 3,
    "very_high": 4,
}


def banned_terms(adjectives: dict, ipip_rows: list[tuple]) -> list[str]:
    terms = set()
    for trait in ("extraversion", "agreeableness"):
        for pole in ("high", "low"):
            for w in adjectives[trait][pole]["all"]:
                terms.add(w.lower())
    terms.update(
        {
            "extraversion",
            "extravert",
            "extraverted",
            "introvert",
            "introverted",
            "introversion",
            "agreeableness",
            "agreeable",
            "disagreeable",
            "neuroticism",
            "conscientiousness",
            "openness",
            "big five",
            "big-five",
            "ocean model",
            "ocean traits",
            "personality trait",
        }
    )
    # Distinctive IPIP item phrases (full stems), not generic tokens like "people"
    for _, _, text, _ in ipip_rows:
        terms.add(text.lower().rstrip("."))
    terms.update(
        {
            "life of the party",
            "center of attention",
            "make friends easily",
            "love large parties",
            "radiate joy",
            "soft heart",
            "take advantage of others",
            "feel sympathy for those who are worse off",
        }
    )
    return sorted(terms)


def write_ipip() -> list[tuple]:
    path = STIM / "ipip_neo_120_EA.csv"
    rows = []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["item_id", "trait", "facet", "text", "keying"]
        )
        w.writeheader()
        for i, (trait, facet, text, keying) in enumerate(IPIP_ITEMS, start=1):
            item_id = f"IPIP-{trait}-{facet}-{i:02d}"
            w.writerow(
                {
                    "item_id": item_id,
                    "trait": trait,
                    "facet": facet,
                    "text": text,
                    "keying": keying,
                }
            )
            rows.append((trait, facet, text, keying))
    return rows


def explicit_persona(
    persona_id: str,
    e_level: str,
    a_level: str,
    e_adj: str,
    a_adj: str,
    variant: int,
) -> dict:
    e_word = LEVEL_WORDS[e_level]
    a_word = LEVEL_WORDS[a_level]
    if variant == 0:
        prompt = (
            f"You are a person who is {e_word} {e_adj} and {a_word} {a_adj}."
        )
    else:
        prompt = (
            f"You are someone who tends to be {e_word} {e_adj} "
            f"and {a_word} {a_adj} in daily life."
        )
    return {
        "persona_id": persona_id,
        "set": "explicit",
        "system_prompt": prompt,
        "intended_E_level": e_level,
        "intended_A_level": a_level,
        "intended_E_numeric": LEVEL_NUMERIC[e_level],
        "intended_A_numeric": LEVEL_NUMERIC[a_level],
        "e_adjective": e_adj,
        "a_adjective": a_adj,
        "variant": variant,
    }


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def build_implicit_personas(rng: random.Random, n_per_quadrant: int = 16) -> list[dict]:
    """Biographical sketches implying E×A quadrants without banned trait words."""

    occupations = [
        "night-shift warehouse coordinator",
        "rural mail carrier",
        "museum exhibit technician",
        "community garden plot manager",
        "long-haul truck dispatcher",
        "public library shelving aide",
        "aquarium maintenance tech",
        "city bus relief driver",
        "bakery early-morning prep cook",
        "state park trail ranger",
        "hospital records clerk",
        "hardware store floor associate",
        "veterinary clinic receptionist",
        "municipal recycling sorter",
        "school cafeteria lead",
        "ferry terminal ticket agent",
        "orchard harvest supervisor",
        "radio transmitter site tech",
        "university lab glassware washer",
        "small-town newspaper layout assistant",
    ]
    ages = ["early twenties", "late twenties", "mid thirties", "early forties", "late forties", "mid fifties"]
    settings = [
        "a humid coastal town",
        "a dry high-desert city",
        "a dense inland suburb",
        "a mountain valley village",
        "a lake-effect industrial city",
        "a prairie college town",
    ]

    # Habit banks by quadrant (high/low E × high/low A) — concrete behaviors only
    # Habit banks avoid banned trait adjectives (e.g. quiet, warm, cold, kind, bold).
    habits = {
        ("high", "high"): [
            "hosts potluck dinners for neighbors twice a month and remembers everyone's dietary needs",
            "starts conversations in the grocery line and often carries bags for people who look overloaded",
            "organizes weekend cleanup crews and stays late to thank every volunteer by name",
        ],
        ("high", "low"): [
            "talks through every meeting at high volume and cuts off slower speakers to keep the agenda moving",
            "competes for the last word in group chats and rarely softens blunt feedback",
            "fills the break room with stories about personal wins and shrugs when coworkers look strained",
        ],
        ("low", "high"): [
            "prefers one low-key lunch companion and leaves handwritten notes of encouragement in shared spaces",
            "listens more than speaks in meetings and covers shifts without fanfare when someone is sick",
            "keeps a small circle but checks in with them by text every evening after work",
        ],
        ("low", "low"): [
            "eats alone in the car and answers messages with one-line replies hours later",
            "keeps headphones on in shared spaces and declines invitations without offering a reason",
            "works through breaks behind a closed door and corrects others only with terse emails",
        ],
    }

    personas = []
    used_occ = list(occupations)
    rng.shuffle(used_occ)
    occ_i = 0
    for e_pole in ("high", "low"):
        for a_pole in ("high", "low"):
            for k in range(n_per_quadrant):
                occ = used_occ[occ_i % len(used_occ)]
                occ_i += 1
                age = rng.choice(ages)
                setting = rng.choice(settings)
                habit = habits[(e_pole, a_pole)][k % 3]
                # Vary structure
                templates = [
                    (
                        f"You live in {setting} and work as a {occ}. You are in your {age}. "
                        f"On a typical weekday you keep a fixed morning routine, commute without rushing, "
                        f"and focus on concrete tasks before checking messages. After work, {habit}. "
                        f"Weekends are for errands, a long walk, and preparing meals for the week ahead."
                    ),
                    (
                        f"In your {age}, you are a {occ} based in {setting}. Your days are structured around "
                        f"shift times, equipment checklists, and brief handoffs with coworkers. "
                        f"Among the people you see regularly, {habit}. "
                        f"You spend evenings on practical chores and a single hobby that does not require a crowd."
                    ),
                    (
                        f"Your role as a {occ} in {setting} defines most weekdays. Now in your {age}, "
                        f"you track sleep, groceries, and transit times carefully. At work and nearby, {habit}. "
                        f"You avoid vague plans and prefer schedules written on paper taped near the door."
                    ),
                ]
                prompt = templates[k % 3]
                # Trim/pad toward 60–90 words
                words = prompt.split()
                if len(words) < 60:
                    prompt += (
                        " You keep a weekly checklist for bills, laundry, and vehicle maintenance."
                    )
                words = prompt.split()
                if len(words) > 90:
                    prompt = " ".join(words[:90])
                e_level = "somewhat_high" if e_pole == "high" else "somewhat_low"
                a_level = "somewhat_high" if a_pole == "high" else "somewhat_low"
                # Use very_high/very_low for half to spread intended levels
                if k % 2 == 0:
                    e_level = "very_high" if e_pole == "high" else "very_low"
                    a_level = "very_high" if a_pole == "high" else "very_low"
                pid = f"implicit_{e_pole[0]}{a_pole[0]}_{k:02d}"
                personas.append(
                    {
                        "persona_id": pid,
                        "set": "implicit",
                        "system_prompt": prompt,
                        "intended_E_level": e_level,
                        "intended_A_level": a_level,
                        "intended_E_numeric": LEVEL_NUMERIC[e_level],
                        "intended_A_numeric": LEVEL_NUMERIC[a_level],
                        "quadrant": f"{e_pole}_E_{a_pole}_A",
                        "word_count": word_count(prompt),
                    }
                )
    return personas


def build_explicit_personas(adjectives: dict, rng: random.Random) -> list[dict]:
    e_test = adjectives["extraversion"]
    a_test = adjectives["agreeableness"]
    personas = []
    idx = 0
    for e_level in LEVELS:
        for a_level in LEVELS:
            e_pole = LEVEL_POLE[e_level]
            a_pole = LEVEL_POLE[a_level]
            for variant in (0, 1):
                e_adj = rng.choice(e_test[e_pole]["test"])
                a_adj = rng.choice(a_test[a_pole]["test"])
                pid = f"explicit_{e_level}_{a_level}_v{variant}"
                personas.append(
                    explicit_persona(pid, e_level, a_level, e_adj, a_adj, variant)
                )
                idx += 1
    assert len(personas) == 32
    return personas


def build_dev_personas(adjectives: dict, rng: random.Random) -> list[dict]:
    """2 explicit + 6 implicit, disjoint from test set."""
    e_test = adjectives["extraversion"]
    a_test = adjectives["agreeableness"]
    explicit = [
        explicit_persona(
            "dev_explicit_0",
            "very_high",
            "very_low",
            rng.choice(e_test["high"]["test"]),
            rng.choice(a_test["low"]["test"]),
            0,
        ),
        explicit_persona(
            "dev_explicit_1",
            "somewhat_low",
            "somewhat_high",
            rng.choice(e_test["low"]["test"]),
            rng.choice(a_test["high"]["test"]),
            1,
        ),
    ]
    for p in explicit:
        p["set"] = "dev_explicit"
    # 6 implicit — reuse builder with n=2 per quadrant then take 6 unique
    implicit_pool = build_implicit_personas(rng, n_per_quadrant=2)
    # Relabel as dev and take 6 spanning quadrants
    chosen = []
    seen_q = set()
    for p in implicit_pool:
        q = p["quadrant"]
        if q not in seen_q or len(chosen) < 6:
            p = dict(p)
            p["persona_id"] = f"dev_{p['persona_id']}"
            p["set"] = "dev_implicit"
            chosen.append(p)
            seen_q.add(q)
        if len(chosen) == 6:
            break
    return explicit + chosen


def main() -> None:
    STIM.mkdir(exist_ok=True)
    adjectives = build_adjectives()
    import yaml

    with (STIM / "adjectives.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(adjectives, f, sort_keys=False, allow_unicode=True)

    ipip_rows = write_ipip()

    (STIM / "extraction_questions.txt").write_text(
        "\n".join(EXTRACTION_QUESTIONS) + "\n", encoding="utf-8"
    )
    (STIM / "probe_questions.txt").write_text(
        "\n".join(PROBE_QUESTIONS) + "\n", encoding="utf-8"
    )

    banned = banned_terms(adjectives, ipip_rows)
    (STIM / "banned_terms.txt").write_text("\n".join(banned) + "\n", encoding="utf-8")

    rng = random.Random(SEED)
    test_explicit = build_explicit_personas(adjectives, rng)
    test_implicit = build_implicit_personas(random.Random(SEED + 1), n_per_quadrant=16)
    baseline = {
        "persona_id": "baseline_none",
        "set": "baseline",
        "system_prompt": None,
        "intended_E_level": None,
        "intended_A_level": None,
        "intended_E_numeric": None,
        "intended_A_numeric": None,
    }
    test_all = test_explicit + test_implicit
    assert len(test_explicit) == 32
    assert len(test_implicit) == 64

    with (STIM / "personas_test.jsonl").open("w", encoding="utf-8") as f:
        for row in test_all + [baseline]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Dev set uses a different RNG stream; rewrite prompts enough to avoid id collision
    dev = build_dev_personas(adjectives, random.Random(SEED + 7))
    assert len(dev) == 8
    with (STIM / "personas_dev.jsonl").open("w", encoding="utf-8") as f:
        for row in dev:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Intended levels table
    table_path = STIM / "persona_intended_levels.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "persona_id",
                "set",
                "intended_E_level",
                "intended_A_level",
                "intended_E_numeric",
                "intended_A_numeric",
            ],
        )
        w.writeheader()
        for row in test_all + [baseline] + dev:
            w.writerow(
                {
                    "persona_id": row["persona_id"],
                    "set": row["set"],
                    "intended_E_level": row.get("intended_E_level"),
                    "intended_A_level": row.get("intended_A_level"),
                    "intended_E_numeric": row.get("intended_E_numeric"),
                    "intended_A_numeric": row.get("intended_A_numeric"),
                }
            )

    print("adjectives poles:", {t: {p: len(adjectives[t][p]['all']) for p in ('high','low')} for t in ('extraversion','agreeableness')})
    print("IPIP rows:", len(ipip_rows))
    print("extraction:", len(EXTRACTION_QUESTIONS), "probes:", len(PROBE_QUESTIONS))
    print("test personas:", len(test_all), "+ baseline")
    print("dev personas:", len(dev))
    print("banned terms:", len(banned))


if __name__ == "__main__":
    main()
