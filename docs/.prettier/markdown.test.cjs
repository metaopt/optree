// Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { resolve } = require("node:path");
const test = require("node:test");

function runPrettier(input, filepath, ...args) {
  return execFileSync("prettier", ["--stdin-filepath", filepath, ...args], {
    cwd: resolve(__dirname, "../.."),
    encoding: "utf8",
    input,
  });
}

function markdown(lines) {
  return [...lines, ""].join("\n");
}

function indent(width, ...lines) {
  const prefix = " ".repeat(width);
  return lines.map((line) => line && prefix + line);
}

function firstMdxListItem(source) {
  const ast = JSON.parse(runPrettier(source, "docs/example.mdx", "--debug-print-ast"));
  return ast.children[0].children[0];
}

/** Check the formatted text and return the input and both passes for AST assertions. */
function check(extension, input, expected = input) {
  const filepath = `docs/example.${extension}`;
  const source = markdown(input);
  const first = runPrettier(source, filepath);
  assert.equal(first, markdown(expected));
  const second = runPrettier(first, filepath);
  assert.equal(second, first);
  return [source, first, second];
}

test("apply the plugin to filenames recognized as Markdown or MDX", () => {
  const input = markdown(["1. Item", "   ```text", "   code", "   ```"]);
  const expected = markdown(["1. Item", "    ```text", "    code", "    ```"]);
  for (const filename of [
    "README",
    "example.MD",
    "example.MDX",
    "example.mdown",
    "example.markdown",
  ]) {
    assert.equal(runPrettier(input, filename), expected, filename);
  }
});

for (const extension of ["md", "mdx"]) {
  test(`list fence indentation in .${extension}`, () => {
    const input = [
      "- Parent",
      "    - Child",
      ...indent(6, "~~~python", "if ready:", "    run()", "~~~"),
      "",
      ...indent(2, "~~~text", "belongs to Parent", "~~~"),
      "",
      "1. Step",
      "   - Substep",
      "",
      ...indent(3, "~~~js", "const x=1", "~~~"),
      "",
      "~~~text",
      "- literal bullet",
      "~~~",
    ];
    const expected = [
      "- Parent",
      "  - Child",
      ...indent(4, "```python", "if ready:", "    run()", "```"),
      "",
      ...indent(2, "```text", "belongs to Parent", "```"),
      "",
      "1. Step",
      "   - Substep",
      "",
      ...indent(4, "```js", "const x = 1;", "```"),
      "",
      "```text",
      "- literal bullet",
      "```",
    ];
    check(extension, input, expected);
  });

  test(`round fence indentation when every ordered list item is a fence in .${extension}`, () => {
    const first = ["first line", "  indented line", "```"];
    const second = ["second item", "```"];
    check(
      extension,
      ["1. ```text", ...indent(3, ...first), "2. ```text", ...indent(3, ...second)],
      ["1.  ```text", ...indent(4, ...first), "2.  ```text", ...indent(4, ...second)],
    );
  });

  test(`round a later fence while preserving the first item's nesting in .${extension}`, () => {
    const first = ["1. ```text", "   example", "   ```", "   - Child", ""];
    const input = [...first, "2. ```text", "   second", "   ```"];
    const expected = [...first, "2.  ```text", "    second", "    ```"];
    check(extension, input, expected);
  });

  test(`align fences in mixed ordered lists in .${extension}`, () => {
    check(extension, ["10. First", "1. ```", "   second", "   ```", "1. Third"]);
    check(extension, ["1. ```text", "   first", "   ```", "2. Text"]);
    check(
      extension,
      ["9. ```txt", "   value", "   ```", "10. Text"],
      ["9.  ```txt", "    value", "    ```", "10. Text"],
    );
  });

  test(`preserve indented continuations alongside fences in .${extension}`, () => {
    function sample(first, later, continuation) {
      return [
        "1. Step",
        "",
        ...indent(first, "```txt", "x", "```"),
        "",
        ...indent(later, "```txt", "later", "```"),
        "",
        "    <!-- note -->",
        "",
        "2. Text\\",
        ...indent(continuation, "line"),
        "",
        ...indent(first, "```txt", "y", "```"),
      ];
    }
    check(extension, sample(3, 3, 4), extension === "mdx" ? sample(3, 4, 4) : sample(4, 4, 3));
  });

  test(`preserve the parent fence following a nested list in .${extension}`, () => {
    const outer = ["```text", "outer", "```"];
    const nextItem = ["", "    <!-- note -->", "", "2. - Substep", ""];
    const input = [
      "1. - ```text",
      ...indent(5, "nested", "```"),
      "",
      ...indent(3, ...outer),
      ...nextItem,
      ...indent(3, ...outer),
    ];
    const expected = [
      "1. -  ```text",
      ...indent(6, "nested", "```"),
      "",
      ...indent(extension === "mdx" ? 3 : 4, ...outer),
      ...nextItem,
      ...indent(4, ...outer),
    ];
    check(extension, input, expected);
  });

  test(`an ignored fence in .${extension} retains its indentation`, () => {
    const input = [
      "1. Item",
      "",
      "   <!-- prettier-ignore -->",
      "   ```js",
      "   const x=1",
      "   ```",
    ];
    check(extension, input);
  });

  test(`a tilde fence can contain a backtick fence in .${extension}`, () => {
    const inner = ["```python", "print(1)", "```"];
    check(
      extension,
      ["1. Example", "", ...indent(3, "~~~text", ...inner, "~~~")],
      ["1. Example", "", ...indent(4, "````text", ...inner, "````")],
    );
  });

  test(`round embedded Markdown fences exactly once in .${extension}`, () => {
    const code = ["```python", "print(1)", "```"];
    check(
      extension,
      ["1. Outer", "", ...indent(3, "~~~markdown", "1. Inner", "", ...indent(3, ...code), "~~~")],
      ["1. Outer", "", ...indent(4, "````markdown", "1. Inner", "", ...indent(4, ...code), "````")],
    );

    const following = ["", "   Paragraph in the outer item"];
    check(
      extension,
      ["1. ~~~~md", "   1. ~~~py", ...indent(6, "print(1)", "~~~"), "   ~~~~", ...following],
      ["1. ````md", "   1.  ```py", ...indent(7, "print(1)", "```"), "   ````", ...following],
    );
  });

  test(`use the core formatter's printed marker widths in .${extension}`, () => {
    check(
      extension,
      ["9. Nine", ...indent(3, "~~~", "nine", "~~~"), "2. Ten", ...indent(3, "~~~", "ten", "~~~")],
      ["9. Nine", ...indent(4, "```", "nine", "```"), "10. Ten", ...indent(4, "```", "ten", "```")],
    );
  });

  test(`round code and display math indentation inside lists in .${extension}`, () => {
    function blocks(width, mdxBodyWidth = width) {
      const equation = "\\frac{x_i}{2}";
      return [
        ...indent(width, "```", equation, "", "  + y", "```"),
        "",
        ...indent(width, "$$", equation),
        "",
        ...indent(extension === "mdx" ? mdxBodyWidth : width, "  + y"),
        ...indent(width, "$$"),
      ];
    }
    function sample(ordered, nested) {
      return [
        ...blocks(0),
        "",
        "- Item",
        "",
        ...blocks(2),
        "",
        "1. Inline $x$.",
        "",
        ...blocks(ordered, 3),
        "",
        "   Last.",
        "",
        "---",
        "",
        "1. Parent",
        "   - Child",
        "",
        ...blocks(nested, 5),
      ];
    }
    check(extension, sample(3, 5), sample(4, 6));
  });

  test(`retain Prettier's implicit closing fences in .${extension}`, () => {
    for (const fence of ["```", "$$"]) {
      const lines = ["1. Parent", "", "   Intro.", "", "    " + fence, "    x"];
      check(extension, lines, [...lines, "    " + fence]);
    }
  });

  test(`other blocks keep Prettier's odd indentation in .${extension}`, () => {
    const input = [
      "1. Item",
      "",
      "   Paragraph",
      "",
      "   \\[", // Prettier does not support brackets in math mode
      "   x",
      "   \\]",
      "",
      "   > Quote",
      "",
      "---",
      "",
      "- Example",
      "",
      "      indented code",
    ];
    check(extension, input);
  });
}

test("preserve indentation after a blank line in MDX math at even fence indents", () => {
  const input = [" $$", "x", "", " y", " $$"];
  const expected = ["$$", "x", "", " y", "$$"];
  check("mdx", input, expected);
  check("mdx", ["- item", "", ...indent(2, ...input)], ["- item", "", ...indent(2, ...expected)]);
});

test("preserve indentation after a blank line when rounding ordered-list MDX math", () => {
  for (const [following, y] of [
    [[], "     y"],
    [["", "   More."], "    y"],
  ]) {
    const input = ["1. Step", "", "    $$", "   x", "", "    y", "    $$", ...following];
    const expected = ["1. Step", "", "    $$", "    x", "", y, "    $$", ...following];
    check("mdx", input, expected);
  }
});

test("preserve the parsed MDX equation when rounding a fence before a paragraph", () => {
  function checkEquation(equation, formattedBody) {
    const input = ["1. Step", "", "   $$", ...indent(3, ...equation), "   $$", "", "   More."];
    const expected = ["1. Step", "", "    $$", ...formattedBody, "    $$", "", "   More."];
    for (const source of check("mdx", input, expected)) {
      const math = firstMdxListItem(source).children.find((node) => node.type === "math");
      assert.equal(math.value, equation.join("\n"));
    }
  }

  checkEquation(["x", "", "y"], ["    x", "", "   y"]);
  checkEquation(["x", "", "", "  y"], ["    x", "", "", "      y"]);
  checkEquation([], []);
});

test("keep the full MDX equation inside a nested bullet on both passes", () => {
  const input = ["1. - Child", "", "      $$", "      x", "", "      y", "      $$"];
  for (const source of check("mdx", input)) {
    const child = firstMdxListItem(source).children[0].children[0];
    assert.equal(child.children[1].value, ["x", "", "y"].join("\n"));
  }
});

test("preserve MDX math on delimiter lines with an implicit closer", () => {
  const input = ["1. Item", "", "     $$ x", "     y", "     $$ x"];
  const expected = ["1. Item", "", "    $$", "    x", "    y", "    $$ x", "    $$"];
  check("mdx", input, expected);
});

test("preserve Prettier's trailing blank lines in MDX math on the first pass", () => {
  const input = ["1. Item", "", "     $$", "     x", "", "     $$"];
  const expected = ["1. Item", "", "    $$", "    x", "", "", "    $$"];
  check("mdx", input, expected);
});

test("preserve nested MDX math with a tab after the opening delimiter", () => {
  const input = ["1. Item", "", "     $$\t", "", "     x", "", "     $$"];
  const expected = ["1. Item", "", "    $$", "", "", "     x", "", "", "    $$"];
  check("mdx", input, expected);
});

test("format nested MDX math with a tab before the closing delimiter", () => {
  const item = ["1. Item", "   with math.", ""];
  const input = [...item, "    $$", "    x", "   \t$$"];
  const expected = [...item, "    $$", "    x", "", "    $$"];
  // Prettier adds another trailing blank line on its second pass for this closing tab.
  assert.equal(runPrettier(markdown(input), "docs/example.mdx"), markdown(expected));
});

test("preserve fences immediately after MDX task checkboxes", () => {
  const input = [
    "1. [x] ~~~text",
    "   ~~~",
    "",
    "2. [ ] ~~~text",
    "   code",
    "     indented",
    "   ~~~",
  ];
  const expected = [
    "1. [x] ```text",
    "",
    "   ```",
    "",
    "2. [ ] ```text",
    "   code",
    "     indented",
    "   ```",
  ];
  check("mdx", input, expected);
});

test("keep a following paragraph inside a nested MDX task", () => {
  for (const [markers, width, following] of [
    ["1. - [x] ", 6, []],
    ["1. 1. - [ ] ", 10, []],
    ["1. - [x] ", 5, ["", "   Outer paragraph."]],
  ]) {
    const input = [
      markers + "~~~text",
      ...indent(width, "code", "  indented", "~~~", "", "Continue here."),
      ...following,
    ];
    const expected = input.map((line) => line.replaceAll("~~~", "```"));
    check("mdx", input, expected);
  }
});
