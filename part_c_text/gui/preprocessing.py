
import re


# Text cleaning patterns

HTML = re.compile(r"<[^>]+>")
URL = re.compile(r"http\S+|www\.\S+")
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2764\u2B50\u2705]")
NON_TEXT = re.compile(r"[^\w\s]|_", re.UNICODE)
DIGITS = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
REPEAT = re.compile(r"(.)\1{2,}")

_HEX2LET = str.maketrans("0123456789abcdef", "ghijklmnopqrstuv")


# Arabic normalization

def base_norm(s):
    s = re.sub("[إأآا]", "ا", s)
    return s.replace("ى", "ي").replace("ة", "ه")


LETTER_MAP = str.maketrans({
    "ث": "ت",
    "ذ": "د",
    "ظ": "ض",
    "ؤ": "و",
    "ڤ": "ف",
    "چ": "ج",
    "گ": "ج",
    "ڨ": "ق"
})


def char_norm(s, letters):
    s = base_norm(s)
    return s.translate(LETTER_MAP) if letters else s


# Stopwords

MSA_STOP = set("""
في من على الى إلى عن مع هذا هذه ذلك تلك هنا هناك هو هي هم هن انا انت انتم نحن أنا أنت التي الذي الذين
ما ماذا لماذا متى كيف اين أين اي أي كل بعض غير بعد قبل حتى اذا إذا لو ان أن إن انه أنه انها أنها كان كانت
يكون تكون كانوا قد لقد ثم او أو و ف ب ل ك لدى لدي عند عندما حيث بين خلال منذ فقط ايضا أيضا
علي عليه عليها عليهم منه منها منهم له لها لهم به بها بهم فيه فيها فيهم يا اما أما إما لكن لكنه لكنها بل ولا ولكن
""".split())

EGY_STOP = set("""
ده دي دا دة دى ديه دول دوول كده كدا كدة ازاي إزاي ايه إيه ايش ليه ليش فين امتى إمتى مين مين ازيك
عشان علشان عشانن اللي اللى لسه لسا لسة برضه برضو برده بقى بقي طب
انتي انتو احنا هما هم احنا دلوقتي دلوقت دلوقتى حاجه حاجة حاجات شوية شويه واحد واحدة حد
فى فيه فيها ف ع علي عليه عليها عليهم معاه معاها معاهم منه منها عنده عندها عندهم عندي عندك
اهو اهي اهم ايوه ايوا لما لسا هو هي ان إن انا انت انك انه انها يلا بتاع بتاعت بتاعه
كان كانت كانوا بقا بقت هيكون يبقى يبقي لو لو سواء
""".split())

KEEP = {
    "لا", "لم", "لن", "ليس", "مش", "ما", "غير", "بدون",
    "مفيش", "مافيش", "مو", "اوي", "خالص", "جدا", "كتير", "قوي"
}


# Dialect spelling variants

_LEX = {
    "مفيش": "مافيش مفيش مافي مفيهوش مافيهوش مفيهاش مافيهاش مفيييش",
    "عايز": "عاوز عايز عاوزه عايزه عاوزين عايزين عايزة عاوزة",
    "اوي": "اوي اووي قوي اوى قوى",
    "كتير": "كتير كتيير كثير",
    "عشان": "علشان عشان عالشان عشانن",
    "ليه": "ليه ليش لييه ليهه",
    "كده": "كده كدا كدة كدي",
    "لسه": "لسه لسا لسة لسه",
    "برضه": "برضه برضو برده برضة",
    "وحش": "وحش وحشه وحشة وحشين",
    "حلو": "حلو حلوه حلوة حلوين",
    "سيء": "سيء سيئ سئ سيئه سيئة سء سيئين",
    "زباله": "زباله زبالة زبالا",
    "روعه": "روعه روعة روووعه",
    "ممتاز": "ممتاز ممتازه ممتازة ممتازين",
    "رائع": "رائع رائعه رائعة",
    "جميل": "جميل جميله جميلة جميلين",
    "كويس": "كويس كويسه كويسة كويسين",
    "تمام": "تمام تمااام",
    "دلوقتي": "دلوقتي دلوقت دلوقتى",
    "ايوه": "ايوه ايوا",
    "شغال": "شغال شغاله شغالة",
    "بطيء": "بطيء بطئ بطيئ بطيئه بطيئة"
}

LEXICON = {
    base_norm(v): base_norm(k)
    for k, variants in _LEX.items()
    for v in variants.split()
}

NEG_CIRCUM = re.compile(r"^(?:ما|م)(.{2,}?)ش$")

DIAL_PREFIX = [
    (re.compile(r"^(?:بي|هي)(.{3,})$"), r"ي\1"),
    (re.compile(r"^(?:بت|هت)(.{3,})$"), r"ت\1"),
    (re.compile(r"^(?:بن|هن)(.{3,})$"), r"ن\1"),
    (re.compile(r"^(?:با|ها)(.{3,})$"), r"ا\1")
]

AL_PREFIX = re.compile(r"^(?:وال|بال|كال|فال|لل|ال)(.{3,})$")


# Stopword configuration

def make_stop(cfg):
    base = (MSA_STOP | EGY_STOP) if cfg["egy_stop"] else MSA_STOP

    return {
        char_norm(w, cfg["letters"])
        for w in base - KEEP
    } - {
        char_norm(w, cfg["letters"])
        for w in KEEP
    }


# Main preprocessing function

def make_cleaner(cfg):
    stop = make_stop(cfg)

    def clean(text):
        t = str(text).lower()

        t = HTML.sub(" ", t)
        t = URL.sub(" ", t)

        t = DIACRITICS.sub("", t).replace("ـ", "")

        if cfg["emoji"]:
            t = EMOJI.sub(
                lambda m: " emo" + format(ord(m.group()), "x").translate(_HEX2LET) + " ",
                t
            )

        t = NON_TEXT.sub(" ", t)
        t = DIGITS.sub(" ", t)
        t = REPEAT.sub(r"\1", t)

        t = base_norm(t)
        toks = t.split()

        if cfg["lex"]:
            toks = [LEXICON.get(w, w) for w in toks]

        if cfg["letters"]:
            toks = [w.translate(LETTER_MAP) for w in toks]

        if cfg["neg"]:
            out = []

            for w in toks:
                m = NEG_CIRCUM.match(w) if (
                    w not in ("مفيش", "مش", "مافيش") and len(w) >= 4
                ) else None

                out += ["مش", m.group(1)] if m else [w]

            toks = out

        toks = [w for w in toks if w not in stop and len(w) > 1]

        if cfg["prefix"]:
            new = []

            for w in toks:
                if len(w) >= 5 and not w.startswith("emo"):
                    for rx, rep in DIAL_PREFIX:
                        if rx.match(w):
                            w = rx.sub(rep, w)
                            break

                new.append(w)

            toks = new

        if cfg["stem"]:
            new = []

            for w in toks:
                if not w.startswith("emo") and w.isascii() is False:
                    m = AL_PREFIX.match(w)

                    if m:
                        w = m.group(1)

                    if len(w) >= 5 and w[0] == "و":
                        w = w[1:]

                    if len(w) >= 4 and w.endswith("ه"):
                        w = w[:-1]

                new.append(w)

            toks = new

        return " ".join(w for w in toks if len(w) > 1)

    return clean
