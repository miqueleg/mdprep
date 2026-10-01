/*
 * mdprep input builder.
 *
 * The decision logic is NOT reimplemented here. Pyodide runs the real mdprep
 * package, so `plan_decisions` and `build_manifest` behave exactly as they do
 * on the command line and the page cannot drift from the manifest schema.
 * This file only loads a structure, renders the plan, and draws the viewer.
 */

const PYODIDE_PACKAGES = ["pydantic", "pyyaml", "micropip"];
// build.py writes vendor/wheel.json because micropip insists on a PEP 427
// filename, which carries the version and therefore cannot be hard-coded.
const WHEEL_INDEX_URL = "vendor/wheel.json";

// How the generated Colab cell installs mdprep. The released PyPI wheel is
// still 0.2.0, which predates the stage-2 NaN and large-PDB fixes, so the
// generated cell installs from the repository instead. Switch this to
// "mdprep" once a release carrying them is published.
const MDPREP_INSTALL = '"mdprep @ git+https://github.com/miqueleg/mdprep@master"';
const COLAB_NOTEBOOK_URL =
  "https://colab.research.google.com/github/miqueleg/mdprep/blob/master/examples/mdprep_interactive_colab.ipynb";

const state = {
  pyodide: null,
  plan: null, // the decision plan, as plain JS objects
  answers: {}, // decision id -> raw answer string
  order: [], // ids of decisions asked so far, for Back
  index: 0,
  structureText: null,
  structureName: null,
  pdbId: null, // set only when fetched from RCSB
  viewer: null,
};

const el = (id) => document.getElementById(id);
const show = (id) => el(id).classList.remove("hidden");
const hide = (id) => el(id).classList.add("hidden");

function setStatus(message, isError = false) {
  const node = el("load-status");
  node.textContent = message;
  node.classList.toggle("error", isError);
}

/* ---------------------------------------------------------------- runtime */

async function bootPython() {
  setStatus("Starting the Python runtime (this takes a few seconds)…");
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(PYODIDE_PACKAGES);
  const index = await fetch(WHEEL_INDEX_URL).then((response) => {
    if (!response.ok) {
      throw new Error(
        `vendor/wheel.json is missing (HTTP ${response.status}). ` +
          "Run `python web/build.py` before serving the page.",
      );
    }
    return response.json();
  });
  const micropip = pyodide.pyimport("micropip");
  // deps=False: every dependency the planning path needs is already loaded
  // above as a prebuilt Pyodide package, and mdprep's numpy/typer/rich
  // dependencies are only used by paths that never run in a browser.
  // callKwargs is the Pyodide way to pass Python keyword arguments; a plain
  // object would arrive as a positional argument instead.
  await micropip.install.callKwargs(`vendor/${index.wheel}`, { deps: false });
  pyodide.runPython(`
import json
from mdprep.config.decisions import plan_decisions, build_manifest, DecisionError
from mdprep.workflows.wizard import render_manifest_yaml
from mdprep.config.decisions import unresolved_mcpb_sites
from mdprep.workflows.runtime import conda_packages_for, micromamba_setup_commands

_PLAN = {}

def mdprep_plan(text, name):
    with open(name, "w") as handle:
        handle.write(text)
    plan = plan_decisions(name)
    _PLAN["plan"] = plan
    return json.dumps(plan.to_dict())

def mdprep_build(answers_json, mdprep_spec):
    plan = _PLAN["plan"]
    answers = json.loads(answers_json)
    try:
        sites = list(unresolved_mcpb_sites(plan, answers))
        text = render_manifest_yaml(plan, answers, mcpb_sites=tuple(sites))
        manifest = build_manifest(plan, answers)
    except DecisionError as exc:
        return json.dumps({"ok": False, "error": str(exc)})
    return json.dumps({
        "ok": True,
        "yaml": text,
        "mcpb_sites": sites,
        # Only the packages this particular run will actually invoke.
        "packages": conda_packages_for(manifest),
        "setup": micromamba_setup_commands(manifest, mdprep_spec=mdprep_spec),
    })
`);
  state.pyodide = pyodide;
  setStatus(`Ready (mdprep ${index.version}). Load a structure to begin.`);
}

/* -------------------------------------------------------------- structure */

async function loadStructure(text, name, pdbId = null) {
  if (!state.pyodide) {
    setStatus("The Python runtime is still starting; try again in a moment.", true);
    return;
  }
  setStatus("Reading the structure…");
  state.structureText = text;
  state.structureName = name;
  state.pdbId = pdbId;
  try {
    const planFn = state.pyodide.globals.get("mdprep_plan");
    const planJson = planFn(text, name);
    planFn.destroy();
    state.plan = JSON.parse(planJson);
  } catch (error) {
    setStatus(`mdprep could not read that structure: ${error.message}`, true);
    return;
  }
  state.answers = {};
  state.order = [];
  state.index = 0;
  setStatus(`Loaded ${name}.`);
  renderFindings();
  initViewer();
  show("findings-panel");
  show("question-panel");
  hide("result-panel");
  renderCurrentQuestion();
}

function renderFindings() {
  const f = state.plan.findings;
  const rows = [
    ["Atoms", f.total_atoms],
    ["Residues", f.total_residues],
    ["Protein residues", f.protein_residues],
    ["Waters", f.water_residues],
    ["Chains", f.chains.join(", ") || "none"],
    ["Ligands", f.ligands.join(", ") || "none"],
    ["Metal ions", f.metal_ions.join(", ") || "none"],
    ["Histidines", f.histidines.length ? `${f.histidines.length}` : "none"],
    ["Possible disulfides", f.possible_disulfides.join("; ") || "none"],
  ];
  el("findings").innerHTML =
    `<table>${rows
      .map(([k, v]) => `<tr><th>${k}</th><td>${escapeHtml(String(v))}</td></tr>`)
      .join("")}</table>` +
    (f.structure_warnings.length
      ? `<p class="warn">${f.structure_warnings.map(escapeHtml).join("<br>")}</p>`
      : "");
}

/* ----------------------------------------------------------------- viewer */

function initViewer() {
  const container = el("viewer");
  container.innerHTML = "";
  state.viewer = $3Dmol.createViewer(container, { backgroundColor: "white" });
  state.viewer.addModel(state.structureText, "pdb");
  state.viewer.setStyle({}, { cartoon: { color: "spectrum" } });
  state.viewer.zoomTo();
  state.viewer.render();
}

// 3Dmol selects on chain/resi/resn, which is exactly what a ResidueTarget
// carries, so the engine's targets drive the viewer with no translation.
function targetSelector(target) {
  const selector = { resi: target.resid, resn: target.resname };
  if (target.chain) selector.chain = target.chain;
  return selector;
}

function focusOn(decision) {
  const viewer = state.viewer;
  if (!viewer) return;
  viewer.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.55 } });

  const caption = [];
  if (decision.focus === "targets") {
    const selectors = decision.targets.map(targetSelector);
    if (decision.target_resnames.length) {
      selectors.push({ resn: decision.target_resnames });
    }
    selectors.forEach((selector) => {
      viewer.setStyle(selector, {
        stick: { radius: 0.22, colorscheme: "greenCarbon" },
        sphere: { radius: 0.45 },
        cartoon: { color: "green" },
      });
      viewer.addStyle(selector, { sphere: { radius: 0.35 } });
    });
    if (selectors.length) {
      // Frame the subject, not the whole highlighted set, so a metal does not
      // get lost inside its surroundings.
      viewer.zoomTo(selectors[0]);
      caption.push(
        decision.targets.length
          ? `Highlighted: ${decision.targets
              .map((t) => `${t.resname}${t.resid}`)
              .join(", ")}`
          : `Highlighted: all ${decision.target_resnames.join("/")} residues`,
      );
    }
  } else if (decision.focus === "structure") {
    viewer.setStyle({}, { cartoon: { color: "spectrum" } });
    viewer.zoomTo();
    caption.push("Showing the whole structure.");
  } else {
    viewer.setStyle({}, { cartoon: { color: "spectrum" } });
    viewer.zoomTo();
  }
  viewer.render();
  el("viewer-caption").textContent = caption.join(" ");
}

/* -------------------------------------------------------------- questions */

// Mirrors Decision.is_applicable: a decision is asked only when every
// condition on it is satisfied by the answers already given.
function isApplicable(decision, answers) {
  return decision.when.every((condition) => {
    const value = answers[condition.decision_id];
    return value !== undefined && condition.values.includes(String(value));
  });
}

function applicableDecisions() {
  const answers = {};
  const result = [];
  for (const decision of state.plan.decisions) {
    if (!isApplicable(decision, answers)) continue;
    result.push(decision);
    const given = state.answers[decision.id];
    answers[decision.id] =
      given !== undefined && given !== "" ? given : decision.default;
  }
  return result;
}

function currentDecision() {
  const decisions = applicableDecisions();
  return decisions[state.index] || null;
}

function renderCurrentQuestion() {
  const decisions = applicableDecisions();
  const decision = decisions[state.index];
  if (!decision) {
    buildManifest();
    return;
  }
  el("question-text").textContent = decision.question;
  el("question-why").textContent = decision.why;
  el("question-evidence").innerHTML = decision.evidence
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
  el("answer-error").textContent = "";

  el("blocking-note").textContent = decision.requires_user_input
    ? "This cannot be guessed from the structure, so there is no default."
    : "";
  el("blocking-note").classList.toggle("visible", decision.requires_user_input);

  const total = decisions.length;
  el("progress-text").textContent = `Question ${state.index + 1} of ${total}`;
  el("progress-bar").style.width = `${((state.index + 1) / total) * 100}%`;
  el("back").disabled = state.index === 0;
  el("next").textContent = state.index === total - 1 ? "Finish" : "Next";

  renderAnswerInput(decision);
  focusOn(decision);
}

function renderAnswerInput(decision) {
  const area = el("answer-area");
  const existing = state.answers[decision.id];
  const value = existing !== undefined ? existing : decision.default;
  area.innerHTML = "";

  if (decision.kind === "choice") {
    decision.choices.forEach((choice) => {
      const id = `choice-${choice.value}`;
      const wrapper = document.createElement("label");
      wrapper.className = "choice";
      wrapper.innerHTML =
        `<input type="radio" name="answer" id="${id}" value="${escapeHtml(choice.value)}"` +
        `${String(value) === choice.value ? " checked" : ""}>` +
        `<span><strong>${escapeHtml(choice.label)}</strong>` +
        `<em>${escapeHtml(choice.detail)}</em></span>`;
      area.appendChild(wrapper);
    });
    return;
  }
  if (decision.kind === "boolean") {
    [
      ["true", "Yes"],
      ["false", "No"],
    ].forEach(([raw, label]) => {
      const wrapper = document.createElement("label");
      wrapper.className = "choice";
      wrapper.innerHTML =
        `<input type="radio" name="answer" value="${raw}"` +
        `${String(value) === raw ? " checked" : ""}>` +
        `<span><strong>${label}</strong></span>`;
      area.appendChild(wrapper);
    });
    return;
  }
  const input = document.createElement("input");
  input.type = decision.kind === "text" ? "text" : "number";
  if (decision.kind === "integer") input.step = "1";
  if (decision.kind === "number") input.step = "any";
  if (decision.minimum !== null) input.min = decision.minimum;
  if (decision.maximum !== null) input.max = decision.maximum;
  input.id = "answer-input";
  input.value = value === null ? "" : value;
  if (decision.requires_user_input && existing === undefined) input.value = "";
  area.appendChild(input);
  input.focus();
}

function readAnswer() {
  const checked = document.querySelector('input[name="answer"]:checked');
  if (checked) return checked.value;
  const input = el("answer-input");
  return input ? input.value.trim() : "";
}

function onNext() {
  const decision = currentDecision();
  if (!decision) return;
  const raw = readAnswer();
  if (raw === "") {
    if (decision.requires_user_input || decision.default === null) {
      el("answer-error").textContent =
        "This question has no default. Please give a value.";
      return;
    }
    state.answers[decision.id] = decision.default;
  } else {
    state.answers[decision.id] = raw;
  }
  state.index += 1;
  renderCurrentQuestion();
}

function onBack() {
  if (state.index === 0) return;
  state.index -= 1;
  renderCurrentQuestion();
}

/* ----------------------------------------------------------------- result */

function buildManifest() {
  const payload = {};
  applicableDecisions().forEach((decision) => {
    const given = state.answers[decision.id];
    if (given !== undefined && given !== "") payload[decision.id] = given;
  });
  const buildFn = state.pyodide.globals.get("mdprep_build");
  const resultJson = buildFn(JSON.stringify(payload), MDPREP_INSTALL);
  buildFn.destroy();
  const result = JSON.parse(resultJson);

  if (!result.ok) {
    el("answer-error").textContent = result.error;
    state.index = Math.max(0, applicableDecisions().length - 1);
    renderCurrentQuestion();
    return;
  }

  el("manifest-yaml").textContent = result.yaml;
  el("result-warnings").innerHTML = result.mcpb_sites.length
    ? `<p class="warn">Metal site(s) ${result.mcpb_sites
        .map(escapeHtml)
        .join(", ")} were set to bonded MCPB. Those need external QM work and
        are deliberately not in the manifest &mdash; add them by hand following
        docs/metals.md.</p>`
    : "";
  window.mdprepLastResultForTest = result;
  renderColabCell(result);
  hide("question-panel");
  show("result-panel");
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function renderColabCell(result) {
  const manifestName = "system.yaml";
  const manifestYaml = result.yaml;

  // Cell 1: build a runtime. mdprep shells out to tleap and antechamber, which
  // pip cannot provide but conda-forge can, so this is a real install rather
  // than a prerequisite the user has to arrange themselves. Only the packages
  // this manifest will actually invoke are included, which keeps the solve as
  // small as the job allows.
  const setup = ["%%bash", ...result.setup].join("\n");

  // Cell 2: get the inputs in place and run.
  let intro;
  let runLines;
  if (state.pdbId) {
    intro =
      `Your structure is in the PDB, so this re-downloads ${state.pdbId} by ID. ` +
      "Nothing needs uploading.";
    runLines = [
      "import pathlib, urllib.request",
      "",
      "urllib.request.urlretrieve(",
      `    "https://files.rcsb.org/download/${state.pdbId}.pdb", "${state.structureName}"`,
      ")",
      `pathlib.Path("${manifestName}").write_text("""`,
      manifestYaml.trimEnd(),
      '""")',
      "",
      RUN_COMMAND(manifestName),
    ];
  } else {
    intro =
      "Your structure is not in the PDB, so download the bundle above and " +
      "upload it when this cell asks. That is what guarantees Colab prepares " +
      "the same file this page planned.";
    runLines = [
      "import zipfile",
      "from google.colab import files",
      "",
      "uploaded = files.upload()   # choose mdprep-bundle.zip",
      'zipfile.ZipFile(next(iter(uploaded))).extractall(".")',
      "",
      RUN_COMMAND(manifestName),
    ];
  }

  el("colab-setup").textContent = setup;
  el("colab-intro").textContent = intro;
  el("colab-command").textContent = runLines.join("\n");
  el("toolchain-note").textContent =
    `This installs ${result.packages.join(", ")} from conda-forge, which is a ` +
    "large download, so the first cell takes several minutes. It only has to " +
    "run once per Colab session, and only the packages this manifest actually " +
    "uses are installed.";
}

// mdprep lives in the micromamba environment, not the notebook's Python, so it
// has to be invoked through it.
function RUN_COMMAND(manifestName) {
  return (
    "!MAMBA_ROOT_PREFIX=/content/mamba-root /content/bin/micromamba run " +
    `-n mdprep-runtime mdprep prepare ${manifestName}`
  );
}

async function downloadBundle() {
  // A plain stored (uncompressed) zip, written by hand so the page needs no
  // zip library: structure and manifest must travel together or Colab can end
  // up preparing a different file from the one that was planned here.
  const files = [
    [state.structureName, state.structureText],
    ["system.yaml", el("manifest-yaml").textContent],
    ["README.txt", bundleReadme()],
  ];
  const blob = makeStoredZip(files);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "mdprep-bundle.zip";
  link.click();
  URL.revokeObjectURL(url);
}

function bundleReadme() {
  return [
    "mdprep input bundle",
    "",
    `structure: ${state.structureName}`,
    "manifest:  system.yaml",
    "",
    "Upload this zip to Colab, extract it, and run:",
    "    mdprep prepare system.yaml",
    "",
    "The manifest references the structure by the name above, so keep them",
    "together in the same directory.",
  ].join("\n");
}

/* ------------------------------------------------------------------ utils */

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = original), 1500);
  } catch {
    /* clipboard blocked; the text is on screen to select manually */
  }
}

/* ------------------------------------------------------------------ wiring */

function wire() {
  el("fetch-pdb").addEventListener("click", async () => {
    const id = el("pdb-id").value.trim().toUpperCase();
    if (!/^[0-9A-Z]{4,8}$/.test(id)) {
      setStatus("Enter a PDB ID such as 1AKI.", true);
      return;
    }
    setStatus(`Downloading ${id} from RCSB…`);
    try {
      const response = await fetch(`https://files.rcsb.org/download/${id}.pdb`);
      if (!response.ok) throw new Error(`RCSB returned ${response.status}`);
      await loadStructure(await response.text(), `${id}.pdb`, id);
    } catch (error) {
      setStatus(`Could not download ${id}: ${error.message}`, true);
    }
  });

  el("pdb-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    await loadStructure(await file.text(), file.name, null);
  });

  el("next").addEventListener("click", onNext);
  el("back").addEventListener("click", onBack);
  el("restart").addEventListener("click", () => {
    state.index = 0;
    hide("result-panel");
    show("question-panel");
    renderCurrentQuestion();
  });
  el("copy-manifest").addEventListener("click", (event) =>
    copyText(el("manifest-yaml").textContent, event.target),
  );
  el("copy-colab").addEventListener("click", (event) =>
    copyText(el("colab-command").textContent, event.target),
  );
  el("copy-setup").addEventListener("click", (event) =>
    copyText(el("colab-setup").textContent, event.target),
  );
  el("download-bundle").addEventListener("click", downloadBundle);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !el("question-panel").classList.contains("hidden")) {
      event.preventDefault();
      onNext();
    }
  });
}

wire();
bootPython().catch((error) => {
  setStatus(`Could not start the Python runtime: ${error.message}`, true);
});

// Exposed for the browser smoke test in tests/browser/smoke.mjs.
window.mdprepState = state;
window.renderColabCellForTest = renderColabCell;
window.storedZipBytesForTest = storedZipBytes;
