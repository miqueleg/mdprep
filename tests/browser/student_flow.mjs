/*
 * Drives the input builder the way a student would: upload a real structure,
 * answer each question by its decision id, take the manifest that falls out.
 *
 *   node student.mjs <baseUrl> <structure.pdb> <chrome> <answers.json>
 */
import fs from "node:fs";
import puppeteer from "puppeteer-core";

const [baseUrl, structurePath, chromePath, answersPath] = process.argv.slice(2);
const ANSWERS = JSON.parse(fs.readFileSync(answersPath, "utf8"));
const report = { asked: [], unanswered: [], errors: [] };

const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: "new",
  args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
});

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
  await page.waitForFunction(
    () => !document.getElementById("question-panel").classList.contains("hidden"),
    { timeout: 180000, polling: 500 });

  report.plan = await page.evaluate(() => ({
    decisions: window.mdprepState.plan.decision_count,
    blocking: window.mdprepState.plan.blocking_count,
  }));

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
    report.asked.push({ id: q.id, answered: answer !== undefined ? String(answer) : null });
    if (answer !== undefined) {
      const ok = await page.evaluate((value) => {
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
process.stdout.write(JSON.stringify(report, null, 1));
