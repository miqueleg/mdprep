/*
 * The viewer when the browser's graphics give way: a structure loaded again,
 * the WebGL context reset, the card out of memory, a browser that keeps
 * losing it, and the Redraw button. Prints a JSON report;
 * test_web_builder_viewer.py checks it.
 *
 * 3Dmol 2.5 draws through one WebGL context it keeps on an offscreen canvas
 * (viewer.renderer.context); the canvas on the page only receives its frames.
 * The failures are forced on that context.
 *
 *   node viewer_recovery.mjs <baseUrl> <structure.pdb> <same structure, renamed> <chrome>
 */
import { launchChrome, reportAndExit } from "./launch.mjs";

const [baseUrl, structurePath, againPath, chromePath] = process.argv.slice(2);
const report = { errors: [] };

let browser;
try {
  browser = await launchChrome(chromePath);
} catch (error) {
  reportAndExit(report, error);
}

// What the viewer is drawing now, read from 3Dmol's own state.
const viewerState = () => {
  const s = window.mdprepState;
  const context = s.viewer && s.viewer.renderer && s.viewer.renderer.context;
  const protein = s.viewer
    ? s.viewer.getModel().selectedAtoms({}).find((atom) => !atom.hetflag)
    : null;
  return {
    hasViewer: Boolean(s.viewer),
    canvases: document.querySelectorAll("#viewer canvas").length,
    lost: context ? context.isContextLost() : null,
    geometries: s.viewer ? s.viewer.renderer.info.memory.geometries : null,
    recoveries: s.viewerRecoveries,
    light: s.lightViewer,
    proteinStyle: protein ? Object.keys(protein.style).sort().join(",") : null,
    caption: document.getElementById("viewer-caption").textContent,
  };
};

const loseContext = () => {
  window.__lostContext = window.mdprepState.viewer.renderer.context;
  window.__beforeLoss = window.mdprepState.viewer;
  window.__lostContext.getExtension("WEBGL_lose_context").loseContext();
};

try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => report.errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") report.errors.push(`console: ${m.text()}`);
  });
  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(
    () => document.getElementById("load-status").textContent.startsWith("Ready"),
    { timeout: 300000, polling: 1000 });

  const upload = async (path) => {
    await page.evaluate(() => { window.__lastViewer = window.mdprepState.viewer; });
    await (await page.$("#pdb-file")).uploadFile(path);
    await page.waitForFunction(
      () => window.mdprepState.viewer && window.mdprepState.viewer !== window.__lastViewer,
      { timeout: 180000, polling: 100 });
  };
  const rebuilt = () => page.waitForFunction(
    () => window.mdprepState.viewer && window.mdprepState.viewer !== window.__beforeLoss,
    { timeout: 20000, polling: 100 });

  await upload(structurePath);
  report.loaded = await page.evaluate(viewerState);

  // The same structure again, as a student retries: the meshes the last
  // viewer was drawing are deleted, not left to garbage collection. (One
  // replaced by a restyle before that is 3Dmol's to collect.)
  report.lastViewerBefore = await page.evaluate(() => {
    const v = window.mdprepState.viewer;
    const meshes = [];
    const walk = (o) => { if (!o) return; if (o.geometry) meshes.push(o); (o.children || []).forEach(walk); };
    walk(v.getModel().renderedMolObj);
    return { geometries: v.renderer.info.memory.geometries, drawn: meshes.length };
  });
  await upload(againPath);
  report.reloaded = await page.evaluate(viewerState);
  report.lastViewerGeometries = await page.evaluate(
    () => window.__lastViewer.renderer.info.memory.geometries);

  // The browser resets the context, as after a GPU driver hiccup or a laptop
  // switching graphics cards: the viewer rebuilds itself on a new one.
  await page.evaluate(loseContext);
  await rebuilt();
  report.afterReset = await page.evaluate(viewerState);
  report.newContext = await page.evaluate(
    () => window.mdprepState.viewer.renderer.context !== window.__lostContext);

  // The card runs out of memory drawing the next question: the protein is
  // drawn as lines from then on, and the caption says why.
  await page.evaluate(() => {
    window.__beforeLoss = window.mdprepState.viewer;
    const context = window.mdprepState.viewer.renderer.context;
    const real = context.getError.bind(context);
    let once = true;
    context.getError = () => {
      if (once) {
        once = false;
        return context.OUT_OF_MEMORY;
      }
      return real();
    };
  });
  await page.click("#next");
  await rebuilt();
  report.afterMemory = await page.evaluate(viewerState);

  // A browser that keeps losing it: rebuilt three times in all, then the
  // viewer stops and says what to do, and the questions go on.
  await page.evaluate(loseContext);
  await rebuilt();
  await page.evaluate(loseContext);
  await page.waitForFunction(() => window.mdprepState.viewerRecoveries > 3, { timeout: 20000, polling: 100 });
  await new Promise((r) => setTimeout(r, 600));
  report.gaveUp = await page.evaluate(viewerState);
  await page.click("#next");
  await page.evaluate(() => window.dispatchEvent(new Event("resize")));
  await new Promise((r) => setTimeout(r, 300));
  report.questionsGoOn = await page.evaluate(() => window.mdprepState.index);

  // Redraw by hand, as a student would once the card is free again.
  await page.click("#redraw-viewer");
  await page.waitForFunction(() => Boolean(window.mdprepState.viewer), { timeout: 20000, polling: 100 });
  report.redrawn = await page.evaluate(viewerState);
  report.ok = true;
} catch (error) {
  report.ok = false;
  report.failure = error.message;
} finally {
  await browser.close();
}
reportAndExit(report);
