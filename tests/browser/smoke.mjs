/*
 * Drives the input builder in a real headless browser and prints a JSON
 * report on stdout. test_web_builder_browser.py runs this and checks the
 * manifest it produces against the mdprep schema.
 *
 *   node tests/browser/smoke.mjs <baseUrl> <structure.pdb>
 */

import fs from "node:fs";
import { launchChrome, reportAndExit } from "./launch.mjs";

/*
 * 3Dmol sizes its canvas from the container, and the sizing can land a frame
 * or two after the panel appears, so reading it the moment the questions show
 * up catches a 0x0 canvas about half the time. Poll until it has real
 * dimensions; on timeout report what is there, so the canvas check fails on
 * its own rather than taking the whole run down with it.
 */
async function canvasSize(page, timeout = 30000) {
  const read = () => {
    const canvas = document.querySelector("#viewer canvas");
    return canvas ? { width: canvas.width, height: canvas.height } : null;
  };
  try {
    await page.waitForFunction(
      () => {
        const canvas = document.querySelector("#viewer canvas");
        return Boolean(canvas && canvas.width > 0 && canvas.height > 0);
      },
      { timeout, polling: 50 },
    );
  } catch {
    // Fall through to whatever the page has.
  }
  return page.evaluate(read);
}

const [baseUrl, structurePath, chromePath, answersPath] = process.argv.slice(2);
const ANSWERS = answersPath ? JSON.parse(fs.readFileSync(answersPath, "utf8")) : {};
const report = { steps: [], errors: [] };

let browser;
try {
  browser = await launchChrome(chromePath);
} catch (error) {
  reportAndExit(report, error);
}

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
  report.canvas = await canvasSize(page);

  // Walk the questions, supplying a value only where the engine refuses to
  // default. Captions are recorded so the test can check what was framed.
  //
  // "0" is not a valid answer to every numeric question -- a spin
  // multiplicity has to be at least 1 -- so a caller can name values by
  // decision id. Without that this used to re-submit a rejected answer until
  // it hit the loop limit, which is why no metal structure was ever driven
  // here.
  let lastIndex = await page.evaluate(() => window.mdprepState.index);
  for (let i = 0; i < 80; i += 1) {
    const done = await page.evaluate(
      () => !document.getElementById("result-panel").classList.contains("hidden"),
    );
    if (done) break;
    const question = await page.evaluate(() => {
      const text = document.getElementById("question-text").textContent;
      const match = window.mdprepState.plan.decisions.find((d) => d.question === text);
      return {
        id: match ? match.id : null,
        text,
        blocking: document.getElementById("blocking-note").textContent !== "",
        hasInput: Boolean(document.getElementById("answer-input")),
        radios: document.querySelectorAll('input[name="answer"]').length,
        caption: document.getElementById("viewer-caption").textContent,
      };
    });
    report.steps.push({ text: question.text, caption: question.caption });
    const named = ANSWERS[question.id];
    if (named !== undefined) {
      await page.evaluate((value) => {
        const input = document.getElementById("answer-input");
        if (input) { input.value = value; return; }
        const radios = [...document.querySelectorAll('input[name="answer"]')];
        const hit = radios.find((r) => r.value === value);
        if (hit) hit.checked = true;
      }, String(named));
    } else if (question.blocking) {
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
    // A rejected answer leaves the page where it was. Stop and say which
    // question, rather than spinning to the loop limit with nothing to show.
    const moved = await page.evaluate(() => window.mdprepState.index);
    if (moved === lastIndex) {
      const error = await page
        .$eval("#answer-error", (n) => n.textContent)
        .catch(() => "");
      throw new Error(
        `stuck on ${question.id || question.text}: ${error || "no value accepted"}`,
      );
    }
    lastIndex = moved;
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

reportAndExit(report);
