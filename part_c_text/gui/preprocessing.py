
import re

ARABIC_STOPWORDS = set("""
في من على الى إلى عن مع هذا هذه ذلك تلك هنا هناك هو هي هم هن انا انت انتم نحن
أنا أنت التي الذي الذين اللذان اللتان ما ماذا لماذا متى كيف اين أين اي أي كل بعض
غير بعد قبل حتى اذا إذا لو ان أن إن انه أنه انها أنها كان كانت يكون تكون كانوا
قد لقد ثم او أو و ف ب ل ك لدى لدي عند عندما حيث بين خلال منذ فقط ايضا أيضا
هذا هاذا هاد دي ده دا دى ديه كده كدا ايه إيه ليه ازاي فين امتى مين اللي
علي عليه عليها عليهم منه منها منهم له لها لهم به بها بهم فيه فيها فيهم
يا اما أما إما لكن لكنه لكنها بل ولا ولكن اللى الى الي
""".split())

KEEP = {"لا", "لم", "لن", "ليس", "مش", "ما", "غير", "بدون", "مفيش", "ماعندي"}

ARABIC_STOPWORDS -= KEEP

HTML = re.compile(r"<[^>]+>")
URL = re.compile(r"http\S+|www\.\S+")
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_TEXT = re.compile(r"[^\w\s]|_", re.UNICODE)
DIGITS = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
REPEAT = re.compile(r"(.)\1{2,}")


def _norm(w):
    w = re.sub("[إأآا]", "ا", w)
    return w.replace("ى", "ي").replace("ة", "ه")


NORM_STOP = {_norm(w) for w in ARABIC_STOPWORDS}


def clean(text: str) -> str:
    t = str(text).lower()
    t = HTML.sub(" ", t)
    t = URL.sub(" ", t)
    t = DIACRITICS.sub("", t).replace("ـ", "")
    t = NON_TEXT.sub(" ", t)
    t = DIGITS.sub(" ", t)
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = REPEAT.sub(r"\1", t)

    tokens = [word for word in t.split() if word not in NORM_STOP]

    return " ".join(tokens)
