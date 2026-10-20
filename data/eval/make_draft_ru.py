"""DRAFT evaluation questions (Ru, Labor Code) proposed by the agent.

gold/twin are CANDIDATES found by searching the corpus; the owner finalizes them
by reading the code (CLAUDE.md 5.2). Twin must be a different chunk with the
opposite polarity on the same topic; empty twin -> excluded from Twin-above-Gold.
Split (dev/test) is assigned after finalization.

Run:  python data/eval/make_draft_ru.py  -> data/eval/queries_draft_ru.jsonl + .tsv
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent

# (slice, question, gold candidates, twin candidates, note)
Q = [
# --- negation-sensitive: the answer flips if retrieval confuses allowed / forbidden
("negation", "Может ли работодатель уволить беременную работницу по сокращению штата?",
 ["labor:54"], ["labor:52:1"], "52:1 — основания расторжения по инициативе работодателя (в т.ч. сокращение штата)"),
("negation", "Можно ли уволить работника по инициативе работодателя, пока он находится на больничном?",
 ["labor:54"], ["labor:52:1"], "54 п.1 — запрет с исключениями; в 52:1 пп.20 — увольнение за неявку >2 мес. по болезни"),
("negation", "Можно ли привлечь беременную работницу к работе в выходной или праздничный день?",
 ["labor:85"], ["labor:86"], "85 — запрет для беременных; 86 — случаи привлечения без согласия"),
("negation", "Может ли работодатель при производственной необходимости временно перевести работника на другую работу без его согласия?",
 ["labor:41"], ["labor:38"], "ответ «да, до 3 месяцев»; twin 38 — перевод только с согласия (проверить формулировку в 38)"),
("negation", "Вправе ли работодатель требовать при приёме на работу документы, которых нет в перечне Трудового кодекса?",
 ["labor:32:6"], ["labor:32:1"], "32:1 — перечень необходимых документов"),
("negation", "Можно ли принять на работу по совместительству человека, которому ещё нет 18 лет?",
 ["labor:26:2"], ["labor:32:4"], "twin слабый: 32:4 — документы для договора о совместительстве"),
("negation", "Можно ли направить на работу вахтовым методом беременную женщину на сроке 20 недель?",
 ["labor:135:3"], [], "естественного twin нет (разрешение для иных работников в том же чанке)"),
("negation", "Может ли работодатель отозвать из ежегодного отпуска работника, которому 17 лет?",
 ["labor:95"], [], "разрешение «только с согласия» и запрет для несовершеннолетних в одном чанке"),
("negation", "Допускается ли применение к работнику дисциплинарного взыскания, не предусмотренного Трудовым кодексом?",
 ["labor:64"], [], "искать twin в 65:* (порядок применения взысканий)"),
("negation", "Может ли испытательный срок для рядового работника составлять четыре месяца?",
 ["labor:36"], [], "для руководителей до 6 месяцев — тот же чанк"),
("negation", "Оплачивается ли работнику время простоя, который произошёл по его вине?",
 ["labor:112"], [], "оплата простоя по вине работодателя — тот же чанк"),
# --- general: the answer does not depend on polarity
("general", "Какова продолжительность основного ежегодного оплачиваемого отпуска?", ["labor:88"], [], ""),
("general", "С какого возраста можно заключить трудовой договор?", ["labor:31"], [], ""),
("general", "Какова нормальная продолжительность рабочего времени в неделю?", ["labor:68"], [], ""),
("general", "Какие документы нужны для заключения трудового договора?", ["labor:32:1"], [], ""),
("general", "За какой срок работник должен предупредить работодателя об увольнении по собственному желанию?", ["labor:56"], [], ""),
("general", "Какое время считается ночным по Трудовому кодексу?", ["labor:76"], [], ""),
("general", "Как оплачивается работа в ночное время?", ["labor:110"], [], ""),
("general", "Сколько длится отпуск по беременности и родам?", ["labor:99:2"], [], ""),
("general", "Какие документы подтверждают трудовую деятельность работника?", ["labor:35"], [], ""),
("general", "Что такое вахтовый метод работы?", ["labor:135:1"], [], ""),
("general", "Что происходит, если работодатель не уведомил работника о результатах испытательного срока до его окончания?", ["labor:37"], [], ""),
]


def main():
    rows = []
    for i, (sl, q, gold, twin, note) in enumerate(Q, 1):
        rows.append({"qid": f"ru_lab_{i:03d}", "pair_id": f"lab_{i:03d}", "lang": "ru", "slice": sl,
                     "question": q, "gold_ids": gold, "twin_ids": twin, "split": "", "notes": note,
                     "status": "draft_owner_finalize"})
    with open(HERE / "queries_draft_ru.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(HERE / "queries_draft_ru.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["qid", "slice", "question", "gold_ids", "twin_ids", "notes", "owner_ok", "owner_comment"])
        for r in rows:
            w.writerow([r["qid"], r["slice"], r["question"], " ".join(r["gold_ids"]), " ".join(r["twin_ids"]),
                        r["notes"], "", ""])
    n = sum(r["slice"] == "negation" for r in rows)
    print(f"{len(rows)} drafts: {n} negation ({sum(bool(r['twin_ids']) for r in rows)} with twin), {len(rows) - n} general")


if __name__ == "__main__":
    main()
