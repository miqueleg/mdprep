/*
 * The "Load SyrB2" button, for a student with no structure of their own.
 * RCSB is answered from the vendored deposition, so the test needs no network
 * and says which URL the page asked for.
 *
 *   node example_button.mjs <baseUrl> <2FCT.pdb> <chrome>
 */
import fs from "node:fs";
import { launchChrome, reportAndExit } from "./launch.mjs";

const [baseUrl, depositionPath, chromePath] = process.argv.slice(2);
const DEPOSITION = fs.readFileSync(depositionPath, "utf8");
const report = { rcsbRequests: [], errors: [] };

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
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    const url = request.url();
    if (url.startsWith("https://files.rcsb.org/")) {
      report.rcsbRequests.push(url);
      // RCSB allows any origin; a stand-in that does not is refused by the
      // page's fetch before the test sees anything.
      const headers = { "Access-Control-Allow-Origin": "*" };
      if (url.endsWith("/2FCT.pdb")) {
        request.respond({ status: 200, contentType: "text/plain", headers, body: DEPOSITION });
      } else {
        request.respond({ status: 404, contentType: "text/plain", headers, body: "not here" });
      }
      return;
    }
    request.continue();
  });

  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(
    () => document.getElementById("load-status").textContent.startsWith("Ready"),
    { timeout: 300000, polling: 1000 });

  report.buttonText = await page.$eval("#load-example", (n) => n.textContent.trim());
  await page.click("#load-example");
  await page.waitForFunction(
    () =>
      ["rename-panel", "question-panel"].some(
        (id) => !document.getElementById(id).classList.contains("hidden"),
      ),
    { timeout: 180000, polling: 500 });

  report.loaded = await page.evaluate(() => ({
    pdbId: window.mdprepState.pdbId,
    structureName: window.mdprepState.structureName,
    decisions: window.mdprepState.plan.decision_count,
    idField: document.getElementById("pdb-id").value,
    questionShown: !document.getElementById("question-panel").classList.contains("hidden"),
  }));
  report.ok = true;
} catch (error) {
  report.ok = false;
  report.failure = error.message;
} finally {
  await browser.close();
}
reportAndExit(report);
