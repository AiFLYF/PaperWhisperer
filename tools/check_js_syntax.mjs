// Parse every front-end ES module to catch syntax errors before they reach a browser.
//
//   node --experimental-vm-modules tools/check_js_syntax.mjs templates/static/js
//
// `--experimental-vm-modules` is required: `vm.SourceTextModule` is the only
// parser that understands `import`/`export` without executing the module.

import { readdirSync, statSync } from 'node:fs';
import { join, resolve, extname } from 'node:path';
import { pathToFileURL } from 'node:url';
import vm from 'node:vm';

if (typeof vm.SourceTextModule !== 'function') {
  console.error('vm.SourceTextModule unavailable — rerun with --experimental-vm-modules');
  process.exit(2);
}

const root = resolve(process.argv[2] ?? 'templates/static/js');

function collect(dir) {
  const found = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) found.push(...collect(full));
    else if (extname(full) === '.js') found.push(full);
  }
  return found;
}

const files = collect(root).sort();
let failed = 0;

for (const file of files) {
  try {
    // Link callbacks are never invoked: nothing imports this context, we only
    // want the parse to fail on a syntax error.
    new vm.SourceTextModule(await import('node:fs').then((fs) => fs.readFileSync(file, 'utf8')), {
      identifier: pathToFileURL(file).href,
    });
    console.log(`OK   ${file.slice(root.length + 1)}`);
  } catch (error) {
    failed += 1;
    console.error(`FAIL ${file.slice(root.length + 1)}: ${error.message}`);
  }
}

console.log(`\n${files.length - failed}/${files.length} modules parsed`);
process.exit(failed === 0 ? 0 : 1);
