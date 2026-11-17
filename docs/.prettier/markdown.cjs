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

/**
 * Prettier plugin for Markdown and MDX that adjusts fenced code and display math indentation while
 * preserving list nesting.
 * @module
 */

// Both pre-commit and CI run the pinned Prettier CLI.
const { parsers, printers } = require.main.require("prettier/plugins/markdown");
const { builders, printer: docPrinter, utils } = require.main.require("prettier/doc");
const markdownPrinter = printers.mdast;

const roundFence = Symbol("round fence indentation");
const formattedMarkdown = Symbol("formatted Markdown");

/** Identify the blocks eligible for indentation rounding. */
function isFencedBlock(node) {
  return node?.type === "math" || (node?.type === "code" && !node.isIndented);
}

/** Mark a printed fence for rounding unless padding would change its list item. */
function markFence(path, options, doc) {
  if (!isFencedBlock(path.node) || markdownPrinter.hasPrettierIgnore(path)) return doc;

  const item = path.parent?.type === "listItem" ? path.parent : null;
  const leading = item && path.isFirst;
  const mdx = options.parser === "mdx";
  if (leading) {
    // A leading MDX task fence needs only the list marker's indent on continuation lines.
    if (mdx && item.checked != null) return builders.dedent(doc);
    // Padding the marker can detach following blocks from this item.
    if (item.children.length > 1) return doc;
  }
  const math = mdx && path.node.type === "math" && !leading ? path.node.value : null;
  return builders.label({ [roundFence]: true, item, math }, doc);
}

/**
 * Return the width of a string alignment prefix, or zero for other document nodes.
 * Prettier uses these prefixes for list markers, block quotes, and footnotes.
 */
function prefixWidth(doc) {
  return doc.type === "align" && typeof doc.n === "string" ? doc.n.length : 0;
}

/** Round each labeled fence's indentation up to a multiple of 2 spaces in place. */
function roundFenceIndents(doc) {
  let indent = 0;
  utils.traverseDoc(
    doc,
    (part) => {
      // An embedded Markdown document has already processed its own fences.
      if (part.label === formattedMarkdown) return false;
      indent += prefixWidth(part);
      if (!part.label?.[roundFence]) return;

      if (indent % 2 === 1) {
        const { math } = part.label;
        part.contents =
          math === null ? [" ", builders.align(" ", part.contents)] : padMdxMath(math);
      }
      return false;
    },
    (part) => {
      indent -= prefixWidth(part);
    },
  );
}

/** Check a Markdown subtree, including its root. */
function containsNode(node, predicate) {
  return predicate(node) || node.children?.some((child) => containsNode(child, predicate));
}

/** Return the list indent MDX consumes, or zero when there are no continuation lines. */
function mdxContinuationIndent(doc, options) {
  const { formatted } = docPrinter.printDocToString(doc, options);
  const lines = formatted
    .split("\n")
    .slice(1)
    .filter((line) => line.trim());
  if (lines.length === 0) return 0;
  return lines.every((line) => /^(?: {4}|\t)/.test(line)) ? 4 : 3;
}

/** Only a fence that occupies its whole item can change the printed list marker's width. */
function soleItemFence(item, doc) {
  if (item.children.length !== 1 || !isFencedBlock(item.children[0])) return;
  return utils.findInDoc(doc, (part) => (part.label?.item === item ? part.label : undefined));
}

/** Preserve the continuation indent that MDX infers for a single-digit list marker. */
function preserveMdxIndent(item, itemDoc, options) {
  const [marker, contents] = itemDoc;
  if (!/^\d\. $/.test(marker)) return;

  const first = item.children[0];
  if (
    first?.type === "list" &&
    containsNode(first, isFencedBlock) &&
    mdxContinuationIndent(itemDoc, options) === 4
  ) {
    // MDX consumes four columns before the nested list's own indentation begins.
    contents.n = "    ";
  }
  if (item.children.length === 1 || !item.children.some(isFencedBlock)) return;

  const labels = [];
  const withoutFences = utils.mapDoc(itemDoc, (part) => {
    if (part.label?.item !== item) return part;
    labels.push(part.label);
    return "";
  });
  if (!labels.length) return;
  const remainingIndent = mdxContinuationIndent(withoutFences, options);
  if (remainingIndent === 0) {
    // With only rounded fences on continuation lines, MDX consumes all four columns as list indent.
    // The core math document already contains the equation's relative indentation.
    for (const label of labels) label.math = null;
  } else if (remainingIndent === 4) {
    // Leave one fence at column three if rounding would also shift prose, HTML, or hard breaks.
    labels[0][roundFence] = false;
  }
}

/** Keep rounding from changing how Prettier parses an ordered list on the next pass. */
function preserveListIndentation(list, doc, options) {
  // Core prints each item as [marker, contents], separated by line breaks.
  const itemDocs = doc.filter((part) => typeof part[0] === "string");
  const items = itemDocs.map((itemDoc, index) => {
    const item = list.children[index];
    if (options.parser === "mdx") preserveMdxIndent(item, itemDoc, options);
    return { item, marker: itemDoc[0], soleFence: soleItemFence(item, itemDoc) };
  });

  // Core infers alignment from the first two markers. The second can shrink to `1.` after `10.`.
  const [first, second] = items;
  if (!first?.soleFence && !second?.soleFence) return;
  const canAlign = items.every(
    ({ item, marker, soleFence }) =>
      soleFence ||
      ((marker.trimEnd().length + 1) % options.tabWidth === 0 &&
        !containsNode(item, (node) => node.type === "list")),
  );
  if (canAlign) return;
  if (first.soleFence && !list.isAligned) first.soleFence[roundFence] = false;
  if (second?.soleFence && list.start >= 10 && Number.parseInt(second.marker, 10) === 1) {
    second.soleFence[roundFence] = false;
  }
}

/** Pad an MDX math fence without adding the padding to the parsed equation. */
function padMdxMath(value) {
  const lines = value ? value.split("\n") : [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i] === "") {
      // MDX reads a blank line and the next line together. Leave both unpadded.
      i++;
    } else {
      lines[i] = " " + lines[i];
    }
  }
  return builders.join(builders.hardline, [" $$", ...lines, " $$"]);
}

/** Print with Prettier, then round the marked fences once all list and quote prefixes are known. */
function printMarkdown(path, options, print, args) {
  const { node } = path;
  const printChild = (...childArgs) => markFence(path, options, print(...childArgs));
  const doc = markdownPrinter.print(path, options, printChild, args);

  if (node.type === "list" && node.ordered) preserveListIndentation(node, doc, options);
  if (!path.isRoot) return doc;
  roundFenceIndents(doc);
  return builders.label(formattedMarkdown, doc);
}

module.exports = {
  parsers: { markdown: parsers.markdown, mdx: parsers.mdx },
  printers: { mdast: { ...markdownPrinter, print: printMarkdown } },
};
