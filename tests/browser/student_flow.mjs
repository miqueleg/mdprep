/*
 * Drives the input builder the way a student would: upload a real structure,
 * answer each question by its decision id, take the manifest that falls out.
 *
 *   node student.mjs <baseUrl> <structure.pdb> <chrome> <answers.json>
 */
import fs from "node:fs";
import { launchChrome, reportAndExit } from "./launch.mjs";

const [baseUrl, structurePath, chromePath, answersPath] = process.argv.slice(2);
const ANSWERS = JSON.parse(fs.readFileSync(answersPath, "utf8"));
const report = { asked: [], unanswered: [], errors: [] };

let browser;
try {
  browser = await launchChrome(chromePath);
} catch (error) {
  reportAndExit(report, error);
}

try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => report.errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") report.errors.push(`console: ${m.text()}`); });
  page.on("response", (r) => { if (r.status() >= 400) report.errors.push(`http ${r.status()}: ${r.url()}`); });

  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(
    () => document.getElementById("load-status").textContent.startsWith("Ready"),
    { timeout: 300000, polling: 1000 });

  await (await page.$("#pdb-file")).uploadFile(structurePath);
  // 6NIE's substrate is a free lysine deposited as LYS A 301, which mdprep
  // refuses: tleap resolves residues by name and would build it from the
  // protein template. The page offers the rename rather than renaming behind
  // the user, so a student clicks here, and so does this.
  await page.waitForFunction(
    () =>
      ["rename-panel", "question-panel"].some(
        (id) => !document.getElementById(id).classList.contains("hidden"),
      ),
    { timeout: 180000, polling: 500 });
  report.renamed = await page.evaluate(
    () => !document.getElementById("rename-panel").classList.contains("hidden"));
  if (report.renamed) {
    report.renameMessage = await page.$eval("#rename-message", (n) => n.textContent);
    await page.click("#rename-apply");
  }
  await page.waitForFunction(
    () => !document.getElementById("question-panel").classList.contains("hidden"),
    { timeout: 180000, polling: 500 });

  report.plan = await page.evaluate(() => ({
    decisions: window.mdprepState.plan.decision_count,
    blocking: window.mdprepState.plan.blocking_count,
  }));

  let lastIndex = -1;
  for (let i = 0; i < 200; i += 1) {
    const done = await page.evaluate(
      () => !document.getElementById("result-panel").classList.contains("hidden"));
    if (done) break;
    // Identify the question from what the page is showing, rather than
    // re-deriving which decision is visible: duplicating the page's own
    // condition logic here is how this driver silently answers the wrong one.
    const q = await page.evaluate(() => {
      const text = document.getElementById("question-text").textContent;
      const match = window.mdprepState.plan.decisions.find((d) => d.question === text);
      return { id: match ? match.id : null, text };
    });
    if (!q.id) { report.unanswered.push(`no decision matches: ${q.text}`); }
    const answer = ANSWERS[q.id];
    const note = await page.$eval("#blocking-note", (n) => n.textContent).catch(() => "");
    const prefilled = await page.evaluate(() => {
      const input = document.getElementById("answer-input");
      return input ? input.value : null;
    });
    // A yes/no or a choice is a set of radios: which one the page selected
    // before the student touched it is that question's default or proposal.
    const selected = await page.evaluate(() => {
      // A checklist is a set of boxes: the ones the page ticked before the
      // student touched it are its default.
      const boxes = [...document.querySelectorAll('input[name="answer-multi"]')];
      if (boxes.length) return boxes.filter((box) => box.checked).map((box) => box.value).join(",");
      const checked = document.querySelector('input[name="answer"]:checked');
      return checked ? checked.value : null;
    });
    report.asked.push({
      id: q.id,
      answered: answer !== undefined ? String(answer) : null,
      note,
      prefilled,
      selected,
    });
    if (answer !== undefined) {
      const ok = await page.evaluate((value) => {
        // A checklist answer names the boxes to tick; yes and no tick all or none.
        const boxes = [...document.querySelectorAll('input[name="answer-multi"]')];
        if (boxes.length) {
          const token = value.trim().toLowerCase();
          const wanted = ["true", "yes", "all"].includes(token)
            ? boxes.map((box) => box.value)
            : ["false", "no", "none", ""].includes(token)
              ? []
              : value.split(",").map((item) => item.trim());
          if (!wanted.every((item) => boxes.some((box) => box.value === item))) return false;
          boxes.forEach((box) => { box.checked = wanted.includes(box.value); });
          return true;
        }
        const input = document.getElementById("answer-input");
        if (input) { input.value = value; return true; }
        const radios = [...document.querySelectorAll('input[name="answer"]')];
        const hit = radios.find((r) => r.value === value);
        if (hit) { hit.checked = true; return true; }
        return false;
      }, String(answer));
      if (!ok) report.unanswered.push(`${q.id}: could not set ${answer}`);
    }
    await page.click("#next");
    await new Promise((r) => setTimeout(r, 120));
    // A blocking question with no answer leaves the page where it is, and
    // the loop would otherwise spin until its limit with nothing to show.
    const moved = await page.evaluate(() => window.mdprepState.index);
    if (moved === lastIndex) {
      const error = await page.$eval("#answer-error", (n) => n.textContent);
      report.unanswered.push(`stuck on ${q.id}: ${error || "no value accepted"}`);
      break;
    }
    lastIndex = moved;
  }

  report.manifest = await page.$eval("#manifest-yaml", (n) => n.textContent).catch(() => null);
  report.warnings = await page.$eval("#result-warnings", (n) => n.textContent).catch(() => "");
  report.ok = Boolean(report.manifest);
} catch (error) {
  report.ok = false;
  report.failure = error.message;
} finally {
  await browser.close();
}
reportAndExit(report);
