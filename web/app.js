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
// The official notebook: it takes the bundle this page writes, prepares and
// equilibrates the system, and shows every step of it.
const COLAB_NOTEBOOK_URL =
  "https://colab.research.google.com/github/miqueleg/mdprep/blob/master/examples/mdprep_colab.ipynb";

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

// Prediction writes mmCIF -- Boltz emits ModelCIF, the RCSB serves mmCIF --
// so the builder takes either. mdprep chooses its reader by extension too,
// which is why the name is carried through rather than the contents sniffed.
const structureFormat = (name) =>
  /\.(cif|mmcif)$/i.test(name || "") ? "cif" : "pdb";
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
from mdprep.config.decisions import (
    plan_decisions,
    build_manifest,
    DecisionError,
    ReservedResidueNameError,
)
from mdprep.workflows.wizard import render_manifest_yaml
from mdprep.config.decisions import (
    answer_conflicts,
    unresolved_mcpb_sites,
    ligands_needing_hydrogens,
    proposed_answers,
    proposal_explanations,
)
from mdprep.structure.reader import read_structure
from mdprep.structure.rename import rename_residues
from mdprep.structure.writer import render_pdb
from mdprep.workflows.runtime import (
    conda_packages_for,
    manifest_wants_gxtb,
    micromamba_run_command,
    micromamba_setup_commands,
)

_PLAN = {}

def mdprep_plan(text, name):
    with open(name, "w") as handle:
        handle.write(text)
    try:
        plan = plan_decisions(name)
    except ReservedResidueNameError as exc:
        # Not an error to show and forget: the page can offer the rename,
        # because mdprep knows exactly which residues and what to call them.
        # It is still a refusal -- nothing proceeds until the user accepts.
        return json.dumps({"blocked": exc.to_dict()})
    _PLAN["plan"] = plan
    payload = plan.to_dict()
    # The structure as mdprep read it, for the viewer to draw. Handing it the
    # uploaded file instead draws the names the file came with, while the
    # questions ask about the names mdprep chose -- so a renamed ligand or
    # metal highlighted nothing at all, including on "are these the atoms
    # coordinating this metal?", the one question asking the student to
    # confirm what they can see.
    payload["structure_pdb"] = render_pdb(read_structure(plan.structure_path))
    return json.dumps(payload)

def mdprep_rename(text, name, renames_json):
    """Apply the rename the user accepted and hand back the new structure.

    The rewrite goes through mdprep's own reader and writer, so the text the
    page then plans, shows and ships is exactly what mdprep will read. Doing
    it in JavaScript would be a second implementation of the PDB column
    layout, and this project has already been bitten by having two.
    """
    with open(name, "w") as handle:
        handle.write(text)
    structure = read_structure(name)
    renamed = rename_residues(structure, json.loads(renames_json))
    return render_pdb(renamed)


def mdprep_proposals(answers_json):
    """Answers that follow from earlier ones, such as a capped model charge.

    The arithmetic comes back with each one so the page can show it; a number
    nobody can check is not much better than no number.
    """
    plan = _PLAN["plan"]
    answers = json.loads(answers_json)
    values = proposed_answers(plan, answers)
    why = proposal_explanations(plan, answers)
    return json.dumps({key: {"value": value, "why": why.get(key, "")}
                       for key, value in values.items()})


def mdprep_conflicts(answers_json):
    """A refusal the answers so far already make certain, and its question."""
    conflict = answer_conflicts(_PLAN["plan"], json.loads(answers_json))
    if conflict is None:
        return "null"
    return json.dumps({"error": str(conflict), "decision": conflict.decision_id})


def _readable_build_error(exc):
    """One line a student can act on, out of whatever was raised."""

    name = type(exc).__name__
    if name == "ValidationError":
        # pydantic prints a block per failure; the message is the useful part.
        parts = []
        for error in getattr(exc, "errors", lambda: [])():
            location = ".".join(str(piece) for piece in error.get("loc", ()))
            message = str(error.get("msg", "")).removeprefix("Value error, ")
            parts.append(f"{location}: {message}" if location else message)
        if parts:
            return "This combination of answers is not a valid manifest. " + "; ".join(parts[:3])
    return f"{name}: {exc}"


def mdprep_build(answers_json, mdprep_spec):
    plan = _PLAN["plan"]
    answers = json.loads(answers_json)
    try:
        sites = list(unresolved_mcpb_sites(plan, answers))
        hydrogenless = list(ligands_needing_hydrogens(plan, answers))
        text = render_manifest_yaml(plan, answers, mcpb_sites=tuple(sites))
        manifest = build_manifest(plan, answers)
    except DecisionError as exc:
        # The question to go back to, when mdprep can name it.
        return json.dumps({"ok": False, "error": str(exc),
                           "decision": getattr(exc, "decision_id", None)})
    except Exception as exc:
        # Anything else, and there is plenty: the manifest is validated by
        # pydantic on the way out, and a ValidationError is not a
        # DecisionError. Three charge methods and two metal answers reach
        # one, and catching only DecisionError meant it escaped Python,
        # escaped onNext, and left Finish doing nothing at all -- no
        # message, no manifest, the same question forever.
        return json.dumps({
            "ok": False,
            "error": _readable_build_error(exc),
        })
    return json.dumps({
        "ok": True,
        "yaml": text,
        "mcpb_sites": sites,
        "ligands_needing_hydrogens": hydrogenless,
        # Only the packages this particular run will actually invoke.
        "packages": conda_packages_for(manifest),
        "setup": micromamba_setup_commands(manifest, mdprep_spec=mdprep_spec),
        "run": micromamba_run_command(manifest, "system.yaml"),
        "needs_gxtb": manifest_wants_gxtb(manifest)[0],
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
  // The file is not adopted until it has been read. Setting it first and
  // returning on failure left the rejected file's name on a plan built from
  // the previous one, and the bundle went out holding one structure while
  // its manifest named the other -- which Colab cannot run, and which looks
  // like nothing went wrong, because the red status is on a panel the
  // student has already scrolled past.
  const previous = {
    text: state.structureText,
    name: state.structureName,
    pdbId: state.pdbId,
  };
  let plan;
  try {
    const planFn = state.pyodide.globals.get("mdprep_plan");
    const planJson = planFn(text, name);
    planFn.destroy();
    plan = JSON.parse(planJson);
  } catch (error) {
    state.structureText = previous.text;
    state.structureName = previous.name;
    state.pdbId = previous.pdbId;
    setStatus(readableLoadError(error), true);
    return;
  }
  if (plan.blocked) {
    // mdprep will not build a ligand whose residue name a force field owns,
    // and will not rename it behind the user's back either. Refuse the file,
    // say why, and offer the rename as something the user decides to do.
    state.structureText = previous.text;
    state.structureName = previous.name;
    state.pdbId = previous.pdbId;
    offerResidueRename(plan.blocked, text, name);
    return;
  }
  hide("rename-panel");
  const atoms = ((plan.findings || {}).total_atoms) || 0;
  if (!atoms) {
    // A text file named .pdb parses to nothing and used to go all the way
    // to a manifest and a set of Colab cells.
    state.structureText = previous.text;
    state.structureName = previous.name;
    state.pdbId = previous.pdbId;
    setStatus(
      `${name} contains no atoms. If it is an mmCIF, give it a .cif name; ` +
        "otherwise check that the file is a structure.",
      true,
    );
    return;
  }
  state.structureText = text;
  state.structureName = name;
  state.pdbId = pdbId;
  state.plan = plan;
  state.viewerText = plan.structure_pdb || null;
  // A new structure is a new session: nothing from the last one may leak
  // into this bundle, including whether it was already downloaded.
  state.downloaded = false;
  state.answers = {};
  state.order = [];
  state.index = 0;
  setStatus(`Loaded ${name}.`);
  renderFindings();
  // The panels come out of hiding first: 3Dmol sizes its canvas from the
  // container, and a container that is still display:none measures zero, so
  // creating the viewer before the panel is visible leaves a 0x0 canvas until
  // a later resize happens to fix it.
  show("findings-panel");
  show("question-panel");
  initViewer();
  hide("result-panel");
  renderCurrentQuestion();
}

/*
 * A residue name the Amber force fields already define cannot be built, and
 * mdprep used to paper over it by renaming the ligand for the build only --
 * so the prepared system described a residue under a name the input never
 * used, and every later reader had to know about the substitution.
 *
 * It refuses now. But the structure this matters most for is 6NIE, whose
 * substrate is a free lysine recorded as LYS 301, fetched here by PDB ID;
 * telling a student to go and edit a megabyte of PDB is where the workflow
 * would end. So the refusal comes with the rename attached, as a thing they
 * choose: nothing is loaded until they do, and what they get afterwards is a
 * structure that really is named that, all the way into Colab.
 */
function offerResidueRename(blocked, text, name) {
  setStatus("This structure cannot be prepared as it is.", true);
  const residues = blocked.residues || [];
  const targets = [...new Set(residues.map((item) => item.resname))];
  const becomes = [
    ...new Set(residues.map((item) => `${item.resname} \u2192 ${item.new_resname}`)),
  ];
  el("rename-message").textContent = blocked.message;
  el("rename-detail").textContent =
    `${residues.length} residue${residues.length === 1 ? "" : "s"}: ` +
    residues.map((item) => item.display).join(", ") +
    `. Every copy of ${targets.join(" and ")} listed here is renamed; the ` +
    "protein's own residues of the same name are not touched.";
  const button = el("rename-apply");
  button.textContent = `Rename ${becomes.join(", ")} and load`;
  button.onclick = () => applyResidueRename(residues, text, name);
  show("rename-panel");
  hide("findings-panel");
  hide("question-panel");
  hide("result-panel");
}

async function applyResidueRename(residues, text, name) {
  setStatus("Renaming\u2026");
  let renamed;
  try {
    const renameFn = state.pyodide.globals.get("mdprep_rename");
    renamed = renameFn(text, name, JSON.stringify(residues));
    renameFn.destroy();
  } catch (error) {
    setStatus(readableLoadError(error), true);
    return;
  }
  hide("rename-panel");
  // The PDB id goes with it. Colab re-downloads by id when there is one, and
  // that would fetch the deposition again -- the one that was just refused.
  // Without an id the page hands over the bundle instead, which carries the
  // structure the rename produced.
  await loadStructure(renamed, name, null);
}

function renderFindings() {
  const f = state.plan.findings;
  const rows = [
    ["Atoms", f.total_atoms],
    ["Residues", f.total_residues],
    ["Protein residues", f.protein_residues],
    ["Waters", f.water_residues],
    ["Chains", f.chains.join(", ") || "none"],
    ["Copies of one protein", copiesSummary(f.oligomer)],
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

// Which protein chains are copies of one protein: what decides whether the
// page will offer to parameterise them once.
function copiesSummary(oligomer) {
  const groups = ((oligomer && oligomer.groups) || []).filter((group) => group.length > 1);
  if (!groups.length) return "none";
  return groups
    .map((group) => {
      const renumbered = group
        .slice(1)
        .some((chain) => oligomer.alignments[chain] && oligomer.alignments[chain].numbering_differs);
      return `chains ${group.join(", ")}` + (renumbered ? " (numbered differently)" : "");
    })
    .join("; ");
}

/*
 * A Pyodide failure arrives as a whole Python traceback, twenty lines of it,
 * with the one useful sentence last. A student cannot act on that, and it is
 * what they get for a truncated file or an mmCIF saved as .pdb -- both
 * ordinary mistakes. Show them the last line, which is the message mdprep
 * actually wrote, and keep the rest in the console for whoever wants it.
 */
function readableLoadError(error) {
  const text = String((error && error.message) || error || "").trimEnd();
  console.error(text);
  const lines = text.split("\n").filter((line) => line.trim());
  const last = lines[lines.length - 1] || "the file could not be read";
  const cleaned = last.replace(/^\w*(Error|Exception):\s*/, "");
  return `mdprep could not read that structure: ${cleaned}`;
}

/* ----------------------------------------------------------------- viewer */

function initViewer() {
  const container = el("viewer");
  container.innerHTML = "";
  state.viewer = $3Dmol.createViewer(container, { backgroundColor: "white" });
  // 3Dmol parses the coordinates itself and needs to be told which format,
  // so this follows the same extension the Python side dispatched on.
  // Always PDB here: this is mdprep's own rendering of what it parsed, so
  // the chain, resname and resid the selectors use are the ones on screen.
  state.viewer.addModel(state.viewerText || state.structureText,
                        state.viewerText ? "pdb" : structureFormat(state.structureName));
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

// Chain letters included: in an oligomer AKG1 is in every copy, and a
// prediction gives each ligand a chain of its own.
function targetList(targets) {
  return targets
    .map((t) => `${t.chain ? `${t.chain}:` : ""}${t.resname}${t.resid}`)
    .join(", ");
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
          ? `Highlighted: ${targetList(decision.targets)}`
          : `Highlighted: all ${decision.target_resnames.join("/")} residues`,
      );
    }
  } else if (decision.focus === "chains") {
    // Which chains to keep, and which copies match: the whole assembly, each
    // chain its own colour, with the residues the question is about picked
    // out in every copy at once.
    viewer.setStyle({}, { cartoon: { colorscheme: "chain" } });
    // Drawn larger than a single site's: the whole assembly is in view.
    decision.targets.map(targetSelector).forEach((selector) => {
      viewer.setStyle(selector, {
        stick: { radius: 0.4, colorscheme: "greenCarbon" },
        sphere: { radius: 1.0, colorscheme: "greenCarbon" },
      });
    });
    viewer.zoomTo();
    caption.push("Each chain in its own colour.");
    if (decision.targets.length) {
      caption.push(`Highlighted: ${targetList(decision.targets)}.`);
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
// condition in `when` is satisfied by the answers already given, and at
// least one in `when_any` if it has any -- a copy's ligand, say, is asked
// when its chain is kept alone or when copies are not to share parameters.
function isApplicable(decision, answers) {
  const met = (condition) => {
    const value = answers[condition.decision_id];
    return value !== undefined && condition.values.includes(String(value));
  };
  const anyOf = decision.when_any || [];
  return decision.when.every(met) && (anyOf.length === 0 || anyOf.some(met));
}

function applicableDecisions() {
  const answers = {};
  const result = [];
  for (const decision of state.plan.decisions) {
    // A browser has no filesystem path the user could usefully give, so
    // these keep their default and never become a question here.
    if (decision.needs_local_filesystem) continue;
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

  // Worked out before the note, because a question that has a proposal must
  // not also be told it has no default.
  const proposal =
    state.answers[decision.id] === undefined ? proposalFor(decision) : null;
  state.proposal = proposal;

  const note = proposal
    ? `Worked out from your earlier answers: ${proposal.why}. Change it if that is not this site.`
    : decision.requires_user_input
      ? "This cannot be guessed from the structure, so there is no default."
      : "";
  el("blocking-note").textContent = note;
  el("blocking-note").classList.toggle("visible", note !== "");

  const total = decisions.length;
  el("progress-text").textContent = `Question ${state.index + 1} of ${total}`;
  el("progress-bar").style.width = `${((state.index + 1) / total) * 100}%`;
  el("back").disabled = state.index === 0;
  el("next").textContent = state.index === total - 1 ? "Finish" : "Next";

  renderAnswerInput(decision);
  focusOn(decision);
}

// Some answers follow from earlier ones -- a capped QM model's charge is the
// sum of the metal, the donors and the cofactors, every one of which has
// already been asked for. Worked out in Python so the arithmetic lives in one
// place, and offered the way a default is: filled in, and changeable.
function proposalFor(decision) {
  if (!state.pyodide || !state.plan) return null;
  try {
    const fn = state.pyodide.globals.get("mdprep_proposals");
    const proposals = JSON.parse(fn(JSON.stringify(state.answers)));
    fn.destroy();
    return proposals[decision.id] ?? null;
  } catch {
    // A proposal is a convenience; failing to compute one must not stop the
    // question being asked.
    return null;
  }
}

function renderAnswerInput(decision) {
  const area = el("answer-area");
  const existing = state.answers[decision.id];
  const proposed = state.proposal ? state.proposal.value : null;
  const value =
    existing !== undefined
      ? existing
      : proposed !== null
        ? proposed
        : decision.default;
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
  input.value = value === null || value === undefined ? "" : value;
  // A blocking question is shown empty so its default cannot be accepted by
  // reflex -- but a proposal is not a default. It was computed from answers
  // this user already gave, the arithmetic is on screen beside it, and
  // blanking it would mean asking them to retype their own sum.
  if (decision.requires_user_input && existing === undefined && proposed === null) {
    input.value = "";
  }
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
    const fallback = (state.proposal ? state.proposal.value : null) ?? decision.default;
    if (fallback === null || fallback === undefined) {
      el("answer-error").textContent =
        "This question has no default. Please give a value.";
      return;
    }
    state.answers[decision.id] = fallback;
  } else {
    state.answers[decision.id] = raw;
  }
  // Some answers rule the build out at once -- a second bonded metal site
  // that would need an MCPB fit of its own. Say so here, at the question that
  // did it, rather than at Finish after a spin question nobody can answer.
  const conflict = conflictAt(decision);
  if (conflict) {
    el("answer-error").textContent = conflict;
    return;
  }
  state.index += 1;
  renderCurrentQuestion();
}

function conflictAt(decision) {
  if (!state.pyodide) return null;
  try {
    const fn = state.pyodide.globals.get("mdprep_conflicts");
    const result = JSON.parse(fn(JSON.stringify(state.answers)));
    fn.destroy();
    return result && result.decision === decision.id ? result.error : null;
  } catch {
    // A convenience: the same refusal still comes from Finish.
    return null;
  }
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
  let result;
  try {
    const resultJson = buildFn(JSON.stringify(payload), MDPREP_INSTALL);
    result = JSON.parse(resultJson);
  } catch (error) {
    // Belt and braces for the Python side's own catch-all: whatever goes
    // wrong, the student gets a sentence rather than a button that stopped
    // responding.
    console.error(error);
    result = { ok: false, error: readableLoadError(error).replace(
      "mdprep could not read that structure: ", "") };
  } finally {
    buildFn.destroy();
  }

  if (!result.ok) {
    // renderCurrentQuestion clears answer-error on its way in, so setting it
    // first meant the message was wiped before anything was painted: Finish
    // did nothing at all, forever, with no text anywhere. Jump to the
    // question the engine is complaining about rather than the last one --
    // it can be twenty questions back -- and write the message after the
    // render, not before.
    const decisions = applicableDecisions();
    let offending = decisions.findIndex((decision) => decision.id === result.decision);
    if (offending < 0) {
      offending = decisions.findIndex((decision) => result.error.includes(decision.id));
    }
    state.index = offending >= 0 ? offending : Math.max(0, decisions.length - 1);
    renderCurrentQuestion();
    el("answer-error").textContent = result.error;
    return;
  }

  el("manifest-yaml").textContent = result.yaml;
  const warnings = [];
  if (result.mcpb_sites.length) {
    warnings.push(`<p class="warn">Metal site(s) ${result.mcpb_sites
      .map(escapeHtml)
      .join(", ")} were set to bonded MCPB. Those need external QM work and
      are deliberately not in the manifest &mdash; add them by hand following
      docs/metals.md.</p>`);
  }
  if ((result.ligands_needing_hydrogens || []).length) {
    // Not asked as a question: a SMILES typed into a form cannot be checked
    // against the residue it is for, and a wrong one silently changes the
    // chemistry. It is reported so it can be added and reviewed in the file.
    warnings.push(`<p class="warn">This structure resolved no hydrogens on
      ${result.ligands_needing_hydrogens.map(escapeHtml).join(", ")}. mdprep
      will not guess a ligand's protonation, so add a <code>smiles:</code> to
      each of those blocks &mdash; or hydrogens to the input &mdash; before
      running. See docs/ligands.md.</p>`);
  }
  el("result-warnings").innerHTML = warnings.join("");
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
      // The suffix that worked here, not an assumed .pdb: a large entry is
      // mmCIF only, and the cell would otherwise fetch a URL that 404s.
      `    "https://files.rcsb.org/download/${state.structureName}", "${state.structureName}"`,
      ")",
      `pathlib.Path("${manifestName}").write_text("""`,
      manifestYaml.trimEnd(),
      '""")',
      "",
      `!${result.run}`,
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
      `!${result.run}`,
    ];
  }

  el("open-colab").href = COLAB_NOTEBOOK_URL;
  el("colab-setup").textContent = setup;
  el("colab-intro").textContent = intro;
  el("colab-command").textContent = runLines.join("\n");
  el("toolchain-note").textContent =
    `This installs ${result.packages.join(", ")} from conda-forge, which is a ` +
    "large download, so the first cell takes several minutes. It only has to " +
    "run once per Colab session, and only the packages this manifest actually " +
    "uses are installed." +
    (result.needs_gxtb
      ? " Your metal site uses a g-xTB Hessian, so the cell also downloads " +
        "the g-xTB build of xtb from the Grimme group's releases and checks " +
        "it against the published checksum: conda-forge carries only the " +
        "GFN-only xtb, which cannot do this calculation."
      : "");
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
  state.downloaded = true;
  el("download-note").textContent =
    "Downloaded. Extract the zip and run: mdprep prepare system.yaml";
}

function bundleReadme() {
  return [
    "mdprep input bundle",
    "",
    `structure: ${state.structureName}`,
    "manifest:  system.yaml",
    "",
    "Extract this zip and run, from the directory it extracted into:",
    "    mdprep prepare system.yaml",
    "",
    "The manifest references the structure by the name above, so keep them",
    "together in the same directory.",
    "",
    "Output goes to the directory named under project.output_dir, relative",
    "to wherever you run the command. Change it there if you want it",
    "somewhere else.",
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
      // Large entries -- ribosomes, capsids -- are mmCIF only, and the PDB
      // download 404s for them. That is indistinguishable from a wrong ID,
      // and this page reads mmCIF perfectly well, so ask for it before
      // giving up. Colab re-fetches by the same name, so the suffix that
      // worked here is the one recorded.
      let response = await fetch(`https://files.rcsb.org/download/${id}.pdb`);
      let suffix = "pdb";
      if (!response.ok) {
        response = await fetch(`https://files.rcsb.org/download/${id}.cif`);
        suffix = "cif";
      }
      if (!response.ok) {
        throw new Error(
          response.status === 404
            ? `RCSB has no entry ${id}`
            : `RCSB returned ${response.status}`,
        );
      }
      await loadStructure(await response.text(), `${id}.${suffix}`, id);
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
    // Going back to question one with every answer still held is not
    // starting over: a student who clicks this to try different chemistry
    // has to press Next through twenty stale answers, and the project name
    // still says the last structure. Keep the structure, drop the answers.
    state.answers = {};
    state.order = [];
    state.index = 0;
    state.downloaded = false;
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
    if (event.key !== "Enter") return;
    // Enter in the PDB-ID box fetches that entry. The handler is on the
    // document, so without this it answered whatever chemistry question was
    // on screen behind the box, accepting its default, and the ID was never
    // fetched at all.
    if (event.target && event.target.id === "pdb-id") {
      event.preventDefault();
      el("fetch-pdb").click();
      return;
    }
    if (!el("question-panel").classList.contains("hidden")) {
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
// Everything lives in this tab. Once a manifest exists there is something
// to lose, so closing without downloading asks first.
window.addEventListener("beforeunload", (event) => {
  const built = !el("result-panel").classList.contains("hidden");
  if (built && !state.downloaded) {
    event.preventDefault();
    event.returnValue = "";
  }
});

window.mdprepState = state;
window.renderColabCellForTest = renderColabCell;
window.storedZipBytesForTest = storedZipBytes;
