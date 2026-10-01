# PaperWhisperer — Front-end design system

What the interface is made of, where each piece lives, and the rules that keep
it from drifting. This is the reference for the CSS layers; if a rule here and a
rule in the code disagree, the code is wrong.

No framework, no build step, no npm, no new dependency. Thirteen `<link>`ed
stylesheets and the cascade.

---

## 1. The cascade

Stylesheets load in this order, and the order **is** the architecture. A later
file may restyle a selector an earlier file also styles, but the codebase is
being held to the stronger rule below: it should not.

| # | File | Owns |
|---|------|------|
| 1 | `tokens.css` | Every colour, size, radius, duration. Nothing else defines a value. |
| 2 | `base.css` | Reset, element defaults, a11y helpers, focus, forced-colours. |
| 3 | `layout.css` | `.container`, nav header, cards, section titles, theme toggle. |
| 4 | `controls.css` | Inputs, drop zone, file meta, checkboxes, the primary button. |
| 5 | `feedback.css` | The file-identity strip, parse status, the reparse form. |
| 6 | `workspace.css` | Result section, the Mermaid canvas. |
| 7 | `papers.css` | Search results, recommendations, the reading queue, signal rows. |
| 8 | `chat.css` | Ask section, the transcript, the answer verdict. |
| 9 | `export.css` | Export preview, workspace guidance, the action button. |
| 10 | `reading.css` | All generated prose: paragraphs, headings, tables, maths, code. |
| 11 | `states.css` | Every run state: progress, empty, error, cancel, failure, clock. |
| 12 | `chrome.css` | Keyframes, footer, back-to-top, the language switcher. |
| 13 | `responsive.css` | Reduced motion, coarse pointer, breakpoints. |

`reading.css` and `states.css` load **after** the files whose selectors they
restyle. This is deliberate and load-bearing: they replace older, looser
definitions of the same selectors, and if either moves above `workspace.css` the
superseded rules start winning again. `tools/check_frontend.py` asserts
`tokens.css` is first; the rest of the order is a convention that is easy to
break silently, so the reasoning is written down here rather than left implicit.

### The one rule

**A thing is defined once.** Two definitions of the same selector is worse than
either one alone, because the winner is decided by load order — a fact nobody
reads at 3am. When a new file supersedes an old rule, the old rule is
**deleted**, not left to be overridden. Files that have already given up a
selector say so in place, so the next person does not "restore" it:

> `workspace.css` §18 and §19 now say only *moved to reading.css* / *moved to
> states.css*, and list the values that used to differ.

The check that enforces this is a selector-collision script (see §8).

---

## 2. The vocabulary

149 custom properties, all in `tokens.css`, which is the only file allowed to
contain a raw colour or a raw pixel size.

**Space** — `--space-0` … `--space-12`, a 4px base. Layouts pick steps, never
sizes. Negative optical offsets are `calc(-1 * var(--space-N))`.

**Type** — one ramp, two voices. `--font-stack` is the workbench (chrome, labels,
meta) and stays compact sans. `--font-reading` is a book serif and is used for
*generated prose only* — analysis, answers, cited passages, abstracts — because
that output is an academic paper being read. Prose gets a real measure
(`--measure: 68ch`) and tabular figures on anything numeric.

**Colour** — five families, and a colour belongs to exactly one:

| family | meaning |
|---|---|
| surface | what you see through to, or sit on |
| text | what you read, ranked |
| border | where one surface ends |
| accent | the product's voice (teal), a second voice (amber), a cool third (blue) |
| state | success / warning / danger / info — meaning about *the run*, not the page |

Two rules follow and they are what make the colour mean something:

- A **state** colour is never decoration. A red banner always means something
  failed.
- An **accent** colour is never a state. Teal never means "error".

### The tint ramp

The most-used effect in the product — an accent hue at low alpha — used to be
written inline as `rgba(14, 125, 110, 0.14)` and thirty variations on it across
six files. Worse, `14,125,110` was the teal *before* the palette was darkened
for contrast, so those literals were not even the colour the tokens name: the
product was running **two teals at once**, and a glow drawn from the old one sat
next to a button filled with the new one.

Tints are therefore a ramp, four steps, declared once per hue:

```
--{hue}-tint-subtle   a wash — backgrounds and rules
--{hue}-soft          the default fill for a selected or tinted surface
--{hue}-tint          a marked or hovered edge
--{hue}-tint-deep     emphasis and glow
```

The alphas are fixed. An ad-hoc `0.18` is not a decision anyone can defend, and
it is exactly how two files end up disagreeing about what "highlighted" looks
like. A tint whose step does not exist is a missing token, not a new literal.

### Contrast

Every text and state colour was measured against its own surface; the ratios are
recorded in the token file next to the value. Three were failing and were fixed:
`--text-faint` (2.99 → 4.72:1), `--warning-color` (3.64 → 5.62:1),
`--success-color` (4.22 → 5.23:1). `--primary-color` was deepened to
`#0a6a5d` because the old teal only cleared 4.5:1 against the card and most of
the UI sits on the body, which is lighter.

---

## 3. States

The happy path is the easy half. One idea holds this together: **the product
streams six sections concurrently over minutes, so the interface is almost always
in a partial state.** Partial is the normal case and gets the most careful
treatment, not the least.

### Progress — position, never a percentage

A parse takes between seconds and minutes and its duration is not knowable in
advance, so the indicator shows **position**: which of four stages
(`upload → analyze → render → ready`), and how many are done. A reader can count
the filled nodes instead of reading the sentence.

A percentage is deliberately absent. A bar that guesses is a bar that is lying,
and a reader who watches a fabricated 60% sit still for two minutes learns to
distrust every number the interface shows — including the real ones. The
connector ahead of the current stage carries an indeterminate march so work is
visibly in flight, and it never reaches the end, so it cannot be misread as
completion.

The one knowable measure is the **elapsed clock** (`.progress-elapsed`), which
`dom.js` rewrites once a second in tabular figures so the digits do not reflow
the line beneath them. The cancel button is amber, not red: a cancelled run is
not a failed run, and it is the only amber control in the product.

### The four failures

`runtime.js` resolves every failure to one of four kinds, and they are not
decoration on one "error" — the reader's next move differs in each:

| kind | meaning | shape | actions |
|---|---|---|---|
| `pending` | work is still happening — **not a failure** | solid, info, live dot | none |
| `retryable` | failed, and the same request may work again | solid, amber | retry |
| `terminal` | failed and repeating it fails identically | solid, red | dismiss |
| `unreachable` | the server could not be reached | **dashed**, red | retry, dismiss |

`pending` gets no buttons: a control that does nothing is worse than no control.
`terminal` deliberately gets no retry button, because a greyed-out "重试" teaches
the reader the button is decorative.

`unreachable` is separated from `terminal` by **shape as well as hue** — a
dashed border against a solid one. The cause is the connection, not the request,
and a shape difference survives a monochrome print and a reader with no colour
perception. An unknown error code is treated as `terminal`, never `retryable`: a
new refusal is far likelier than a transient blip, and the raw code is shown so
an operator has something to act on.

### Other states

- **Awaiting** — before the first section event lands, every container is
  genuinely empty. Left bare, a column of blank cards cannot tell "still
  working" from "returned nothing". Each empty section container draws a quiet
  rule and an "awaiting analysis" note. `#exportPreview` is excluded: it has its
  own copy.
- **Partial** — a section that failed while the rest of the document succeeded
  gets a solid red rule, loud enough to read and quiet enough that a reader does
  not think the whole run failed.
- **Empty vs unavailable** — in the recommendation signals, `unavailable` and
  `empty` cannot look alike: they are separate states with separate copy, and
  `data-status` is the state, never a count, so a 0 on an unavailable signal
  cannot borrow the styling of a 0 on an empty one.
- **Success** — the status chip stops pulsing. The pulse is a liveness signal
  and something that has finished is not live.
- **Offline** — amber, not red, and the dot keeps breathing: a machine with no
  network has not done anything wrong, and every action the reader took is still
  queued. `config.js` declares an `offline` tone; it is styled so that value does
  not depend on which module happens to run.

---

## 4. Reading

Dense academic content is what usually looks worst, because it is generated,
unpredictable, and arrives as raw markdown. Two rules govern it.

**Prose is capped; data is not.** Running text is held to `--measure` and
centred, because past ~70ch a serif at reading size loses the return sweep.
Tables, code blocks and equations keep the full card width, because a column
narrow enough to read is a column too narrow for a seven-column results table.
The cap is applied per block, never to the container.

**Tables are the hardest content in the product** and the one most often left as
an unstyled border grid. They get the full treatment: a sticky header that stays
put while a wide table scrolls sideways (which is the whole reason to scroll
one), `tabular-nums` so a column of figures is scannable, `overflow-wrap` so a
long DOI wraps inside its cell rather than widening the table, and a last-row
rule that stops.

`data-cell="R2C3"` coordinates from the backend are **not** rendered as visible
text. It would be tempting, and it would be wrong: the injected string joins the
accessibility tree and the copy buffer, so copying a results table out of this
product would produce `R2C30.8421`. Coordinates reach the reader through the
evidence panel's target rows instead, which is where a human would look.

A cell the locator named is emphasised with a **corner notch** — a shape, not
just a colour — so it survives a monochrome print.

**Headings are signposts, not display elements.** Margins are asymmetric: tight
to the text a heading introduces, well clear of the text it follows. Sub-heads
carry a hairline, which does the work of separating "the method" from "the
results" in a long document where heading size alone is too weak to.

**Blockquotes are not italic.** Italic on a serif is right for an excerpt and
wrong for a definition, and this product has both. The distinction is carried by
the rule and the serif instead — a citation that reads as an aside is not a
citation.

**Maths.** KaTeX is injected after sanitisation, so display equations are
guaranteed to arrive and the two things that actually break in a narrow card get
attention: an equation wider than its container scrolls rather than shrinking
into illegibility, and the fraction line is forced to the current colour. Inline
maths is pinned to `1em` so it does not blow a paragraph's line spacing apart.

**Code is data, not prose** — mono, never wrapping mid-token, and a code block is
the one place a horizontal scroll is the right answer, because reflowing code
changes what it means.

---

## 5. Theme

Dark mode is a re-declaration of the same roles, not an inversion. Hues are held
and only lightness moves, so "teal" means the same thing in both themes; text
contrast is measured against `--bg-card` in the dark theme rather than reusing
the light numbers. The one deliberate inversion is declared once, in one place:

> `--text-on-accent-fill` is white in light and near-black in dark, because the
> accent fill is a light mint in dark and white on it sits at roughly 1.6:1.

That is the only reason any colour flips, and it is why the filled button reads
`var(--text-on-accent-fill)` rather than a literal `#ffffff`.

The tint ramp is re-declared per theme: a wash that reads on paper has to be
lighter on a dark desk or it vanishes. The step *names* are identical in both, so
no layer ever needs to know which theme is active.

---

## 6. Accessibility

Accessibility is part of the design here, not a pass at the end.

- **One focus ring**, always the same width, always drawn *outside* the element
  so it is never clipped by an `overflow: hidden` ancestor, and a `box-shadow`
  rather than an `outline` so a control is still identifiable as focused while it
  is also `:disabled` or `:checked`. `select` was added to the list — the reparse
  parser and the export format are `<select>` elements and were being left out,
  and a control with no focus ring is a control a keyboard reader cannot find.
- **Forced colours.** A box-shadow is not painted at all in Windows High
  Contrast, so there the ring switches to an `outline`, which the OS honours.
  This is only safe because every colour-signalled state is *also* signalled by a
  border, a rule or a shape — the done-node, the tone rules, cited cells, claim
  presentation, the answer verdict. That redundancy is the requirement, not a
  nicety.
- **Reduced motion.** One block, last, so it wins. The universal reset is there,
  but the permanent loops are also **named** — brand pulse, status dot, connector
  sweep, shimmer, pending-notice dot, heartbeat — so a reader who has asked for
  no motion sees none of them, and so the list stays reviewable: a new loop has to
  be a deliberate line in the file.
- **Focus order** is document order; the skip links are the first tab stop and
  reach upload, search and the workspace directly. `tabindex="-1"` appears only
  on programmatic focus targets and on controls that are genuinely hidden.
- **Live regions** are polite for status, assertive only for the error banner.
  A notice is a status update about work the reader initiated and is watching;
  interrupting a screen reader mid-sentence for something they already know is
  happening is a regression, not an improvement.

---

## 7. Honesty in the pixels

Where the product cannot do something, the interface says so in a way that reads
as designed rather than as a missing feature. These are load-bearing, not copy
that can be trimmed.

- **The page map is a schematic, not a rendered page.** It is built from the
  parser's own coordinates and styled to look like one: muted blocks, a dashed
  page outline, a highlight that reads as an overlay. A visible
  `非原始渲染图` badge says so in *words* as well, because colour and position are
  not a disclaimer a reader can act on — if they believe they are looking at the
  paper, the citation they trust is the paper's, and it may not be.
- **"No visual targeting" is not "this failed."** It is a separate state from the
  error, because the passage resolved and is shown below; only the on-page
  locator is unavailable. Collapsing the two would report a limitation of the
  *paper* where the real answer is a limitation of the *parse*.
- **The answer verdict** states what the product will and will not vouch for,
  under the answer it qualifies. `insufficient_evidence` is deliberately **not
  red** — "this is not backed by the document" is a true and useful answer, and
  painting it as an error would teach the reader that the confident answers are
  the safe ones and this one is broken. It is quiet, because quiet is what
  "nothing here is presented as the paper's claim" should look like.
- **Degraded retrieval is stated.** If retrieval fell back, the verdict says so,
  including when a locator came from re-derived plain text rather than a real
  parse, so a character offset and a page number are never rendered as though
  they were the same kind of thing.
- **The export hint is written before the download,** not left to be discovered
  by opening the file months later.

The verdict is also set in sans, small and rule-bounded while the answer beneath
it is serif and measured — because it is a qualification *of* the answer, not
another paragraph the model wrote.

---

## 8. Checks

Run before any commit:

```bash
python3 tools/check_frontend.py
node --experimental-vm-modules tools/check_js_syntax.mjs templates/static/js
```

Both must stay green. `check_frontend.py` asserts every referenced asset exists,
no stylesheet is listed twice, `tokens.css` is first, the module graph is
acyclic, required containers exist, and every network call goes through
`api.js`.

### Things a stylesheet test cannot catch

Three invariants hold this file together and none of them are enforced by a test,
so they are checked by hand or by a script:

1. **No selector is defined twice with different values.** The winner is decided
   by load order, and nothing about a duplicate is an error to the browser.
2. **No raw colour or pixel size outside `tokens.css`.** Grep for
   `#[0-9a-f]{3,8}`, `rgba?(`, `font-size: Npx`, and `padding|margin|gap` with a
   `Npx`. The only sanctioned exceptions are the two `#000 0 0` mask gradients
   (a technical mask, not a colour) and the `margin: -1px` in `.sr-only`, which
   is the clip's own compensation and has to cancel the 1px the element is
   measured with.
3. **Every `var(--x)` resolves to a token that is actually declared.** A typo
   resolves to nothing and silently inherits, which is the same failure mode as
   a stylesheet that fails to load.

---

## 9. The one piece of styling left in the template

The evidence panel, the evidence view and the mobile bottom sheet still live in
a `<style>` block in `index.html`. This is not a preference: three tests in
`tests/test_frontend_evidence.py` assert against the **contents** of that
element — the panel's positioning, the `1080px` sheet breakpoint (mirrored by
`EVIDENCE_SHEET_MEDIA_QUERY` in `config.js`), the sheet's scroll containment, the
absence of a body scroll lock, and the absence of a `::before`/`::after` rule
that could fabricate a quotation mark around a paraphrase.

Moving them into `static/css/` requires changing that test in the same commit.
Everything that was parked in the same block and was *not* test-pinned — claim
provenance, parse status, the export format control, the recommendation signal
rows, the language switcher — has already been moved into the cascade, into the
file that owns each surface.

---

## 10. What the JS agents must adopt

The CSS will not be changed to accommodate a new class name later. If a module
needs a new state, add the class here in the same commit.

- **A new run state belongs in `states.css`**, using the existing four failure
  kinds or the tone vocabulary. Do not open a second failure surface.
- **`data-tone` carries state, never a count.** A `0` on an unavailable signal
  and a `0` on an empty one must not be able to share a style.
- **Signals that are carried in words must stay in words.** `data-confidence`,
  `data-status`, `data-kind` and the visible sentence are redundant on purpose;
  the sentence is what survives a stylesheet that fails to load, a forced-colours
  mode, and a screen reader. Do not delete the sentence and keep the attribute.
- **A state that is new to the product needs a shape, not just a hue.** Solid vs
  dashed vs a corner notch is what makes it survive monochrome. If the only
  difference is a colour, it is not finished.
- **State class names are hyphenated and unprefixed within their layer**
  (`runtime-notice-*`, `section-state-*`, `answer-verdict-*`, `signal-*`). The
  `.runtime-notice` and `.section-state` base classes are shared with the JS
  module of the same name, so they may not be renamed.
- **`--evidence-panel-width` and the `1080px` breakpoint are a contract with
  `config.js`.** `EVIDENCE_SHEET_MEDIA_QUERY` and the stylesheet media query must
  agree or the sheet will trap focus in a column the reader can still read past.
