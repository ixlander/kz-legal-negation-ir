const CELL = {
  "neg-prohibit": "ответ «нет»: норма запрещает",
  "neg-permit": "ответ «да»: норма разрешает, рядом есть запрет",
  "neg-explicit": "в вопросе есть отрицание",
  "gen-polar": "ответ «да», обратной нормы нет",
};
const lawPair = c => `<div class="cols"><div class="law"><span class="t">${esc(c.id)} · ${esc(c.title)}</span>${esc(c.text)}</div>
  <div class="law"><span class="t">${esc(c.title_kz)}</span>${esc(c.text_kz)}</div></div>`;
const SECTIONS = [
  { id: "questions", tab: "Вопросы v2", items: DATA.questions,
    intro: "Новые черновики вопросов по Трудовому и Налоговому кодексам (написал Claude, казахский текст не проверен). Все вопросы закрытые («можно ли…»). <b>Gold</b> — пункт, который отвечает на вопрос. <b>Twin</b> — пункт на ту же тему с обратным смыслом; если такого в кодексе нет, поле пустое. Метка справа говорит, каким задуман вопрос: проверьте, что она соответствует тексту gold. Id пункта: <code>labor:54:2</code> = Трудовой кодекс, статья 54, пункт 2. Что проверить: 1) вопрос понятен и на него отвечает именно gold; 2) казахская версия спрашивает то же, что русская, и звучит так, как спросил бы обычный человек, а не как цитата из кодекса; 3) twin действительно говорит обратное. Правьте прямо в полях (несколько id — через пробел).",
    choices: [["ok", "Годится", "ok"], ["fixed", "Исправил(а)", "fix"], ["drop", "Выкинуть", "bad"]],
    body: x => `<div class="head"><span class="id">${esc(x.qid)} · ${x.code === "tax" ? "Налоговый" : "Трудовой"}</span><span class="chip">${esc(CELL[x.cell] || x.cell)}</span></div>
      <div class="rows">${row("Вопрос (ru)", area(x.key, "question", x.question, 2))}
      ${row("Вопрос (kz)", area(x.key, "question_kz", x.question_kz, 2))}
      ${row("Gold", area(x.key, "gold", x.gold.map(c => c.id).join(" "), 1))}
      ${row("Twin", area(x.key, "twin", x.twin.map(c => c.id).join(" "), 1))}</div>
      <div class="rows">${x.gold.map(c => row("Текст gold", lawPair(c))).join("")}${x.twin.map(c => row("Текст twin", lawPair(c))).join("")}</div>` },
];
