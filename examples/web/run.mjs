// The web example's cases in Pyodide under Node (examples/web/README.md).
//
//   npm install && node run.mjs                  # torchnative from PyPI, through micropip
//   node run.mjs --wheel path/to/torchnative-*-pyemscripten_2026_0_wasm32.whl
//
// The same Pyodide index.html loads in a browser (314.0.8, CPython 3.14, platform tag
// pyemscripten_2026_0). The page and this script run basics.py's cases and compare them
// with reference.json, written by reference.py with upstream torch. Exit status: 0 when
// every case agrees, 1 when one disagrees or raises, 2 when torch cannot be installed or
// imported. Nothing here stubs a module before `import torch`: what the wheel needs, it
// must carry.
import { readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const at = args.indexOf("--wheel");
const wheel = at >= 0 ? args[at + 1] : null;
if (at >= 0 && !wheel) {
  console.error("--wheel needs a path");
  process.exit(2);
}

const py = await loadPyodide();
await py.loadPackage("micropip");
py.FS.writeFile("/home/pyodide/basics.py", readFileSync(join(here, "basics.py")));
let spec = "torchnative";
if (wheel) {
  spec = `emfs:/tmp/${basename(wheel)}`;
  py.FS.writeFile(`/tmp/${basename(wheel)}`, readFileSync(wheel));
}
py.globals.set("SPEC", spec);
py.globals.set("REFERENCE", readFileSync(join(here, "reference.json"), "utf8"));
try {
  await py.runPythonAsync(`
import micropip
await micropip.install(SPEC, pre=True)
print("INSTALLED " + str(micropip.list()["torchnative"].version) + " from " + SPEC)
import torch
`);
} catch (err) {
  console.log(`IMPORT_FAILED ${String(err.message ?? err).trim().split("\n").slice(-3).join(" | ")}`);
  process.exit(2);
}
const code = await py.runPythonAsync(`
import sys
sys.path.insert(0, "/home/pyodide")
import basics
basics.main(REFERENCE)
`);
process.exit(code);
