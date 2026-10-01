/*
 * Drives the input builder in a real headless browser and prints a JSON
 * report on stdout. test_web_builder_browser.py runs this and checks the
 * manifest it produces against the mdprep schema.
 *
 *   node tests/browser/smoke.mjs <baseUrl> <structure.pdb>
 */

import puppeteer from "puppeteer-core";

const [baseUrl, structurePath, chromePath] = process.argv.slice(2);
const report = { steps: [], errors: [] };

const browser = await puppeteer.launch({
  executablePath: chromePath,
  headless: "new",
  args: ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
});

try {
  const page = await browser.newPage();
  page.on("pageerror", (error) => report.errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      report.errors.push(`console: ${message.text()}`);
    }
  });
  // A 404 for anything, including a favicon, means the site is incomplete.
  page.on("response", (response) => {
    if (response.status() >= 400) {
      report.errors.push(`http ${response.status()}: ${response.url()}`);
    }
  });

  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });

  // Pyodide downloads its runtime and the mdprep wheel on first load.
  await page.waitForFunction(
    () => document.getElementById("load-status").textContent.startsWith("Ready"),
    { timeout: 300000, polling: 1000 },
  );
  report.status = await page.$eval("#load-status", (n) => n.textContent);

  await (await page.$("#pdb-file")).uploadFile(structurePath);
  await page.waitForFunction(
    () => !document.getElementById("question-panel").classList.contains("hidden"),
    { timeout: 120000, polling: 500 },
  );

  report.plan = await page.evaluate(() => ({
    decisions: window.mdprepState.plan.decision_count,
    blocking: window.mdprepState.plan.blocking_count,
    ligands: window.mdprepState.plan.findings.ligands,
  }));
  report.canvas = await page.evaluate(() => {
    const canvas = document.querySelector("#viewer canvas");
    return canvas ? { width: canvas.width, height: canvas.height } : null;
  });

  // Walk the questions, supplying a value only where the engine refuses to
  // default. Captions are recorded so the test can check what was framed.
  for (let i = 0; i < 60; i += 1) {
    const done = await page.evaluate(
      () => !document.getElementById("result-panel").classList.contains("hidden"),
    );
    if (done) break;
    const question = await page.evaluate(() => ({
      text: document.getElementById("question-text").textContent,
      blocking: document.getElementById("blocking-note").textContent !== "",
      hasInput: Boolean(document.getElementById("answer-input")),
      radios: document.querySelectorAll('input[name="answer"]').length,
      caption: document.getElementById("viewer-caption").textContent,
    }));
    report.steps.push({ text: question.text, caption: question.caption });
    if (question.blocking) {
      if (question.hasInput) {
        await page.$eval("#answer-input", (node) => {
          node.value = "0";
        });
      } else if (question.radios) {
        await page.evaluate(() => {
          document.querySelector('input[name="answer"]').checked = true;
        });
      }
    }
    await page.click("#next");
    await new Promise((resolve) => setTimeout(resolve, 150));
  }

  report.manifest = await page.$eval("#manifest-yaml", (n) => n.textContent);
  report.colabUpload = await page.$eval("#colab-command", (n) => n.textContent);
  report.colabSetup = await page.$eval("#colab-setup", (n) => n.textContent);
  report.toolchainNote = await page.$eval("#toolchain-note", (n) => n.textContent);

  // The other handoff path: a structure fetched from RCSB is re-fetched by
  // Colab, so no bundle has to change hands.
  report.colabRcsb = await page.evaluate(() => {
    window.mdprepState.pdbId = "1AKI";
    window.mdprepState.structureName = "1AKI.pdb";
    const render = window.renderColabCellForTest;
    render(window.mdprepLastResultForTest);
    return document.getElementById("colab-command").textContent;
  });

  // The bundle the upload path depends on.
  report.bundleNames = await page.evaluate(async () => {
    const bytes = window.storedZipBytesForTest([
      ["x.pdb", "END\n"],
      ["system.yaml", "project:\n"],
    ]);
    return bytes.length;
  });

  report.ok = true;
} catch (error) {
  report.ok = false;
  report.failure = error.message;
} finally {
  await browser.close();
}

process.stdout.write(JSON.stringify(report));
