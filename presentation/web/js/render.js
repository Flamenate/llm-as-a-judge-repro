/* global mermaid */

import { applyFrenchSpacing } from "./parser.js";

export const SLIDE_WIDTH = 1920;
export const SLIDE_HEIGHT = 1080;

const FOOTER_TEXT = "Justice or Prejudice? — étude de reproductibilité";
const TEXT_MIN_PX = 18;
const TEXT_HEAVY_CHARS = 900;
const TABLE_MIN_PX = 14;

let mermaidCounter = 0;

export function initMermaid() {
  mermaid.initialize({
    startOnLoad: false,
    theme: "neutral",
    securityLevel: "strict",
    fontFamily: "Inter, system-ui, sans-serif",
    themeVariables: { fontSize: "22px" },
    flowchart: {
      htmlLabels: true,
      curve: "basis",
      padding: 18,
      subGraphTitleMargin: { top: 8, bottom: 14 },
    },
  });
}

function element(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function layoutOf(slide) {
  if (slide.isTitle) return "layout-title";
  return slide.visuals.length ? "layout-two" : "layout-full";
}

function buildVisual(visual) {
  if (visual.type === "image") {
    const box = element("figure", "visual visual-image");
    const img = document.createElement("img");
    img.src = visual.src;
    img.alt = visual.alt || "";
    img.decoding = "async";
    box.append(img);
    return box;
  }
  if (visual.type === "table") {
    return element("div", "visual visual-table", visual.html);
  }
  const box = element("div", "visual visual-mermaid");
  box.dataset.code = visual.code;
  return box;
}

const VERDICT_CLASSES = [
  ["non comparable", "verdict-na"],
  ["en partie", "verdict-partial"],
  ["oui", "verdict-yes"],
  ["non", "verdict-no"],
];

/* Colour table cells that open with a bold "Oui", "En partie", "Non"... */
function markVerdicts(root) {
  for (const cell of root.querySelectorAll("td")) {
    const lead = cell.querySelector("strong");
    if (!lead || cell.firstElementChild !== lead) continue;
    const word = lead.textContent.trim().toLowerCase();
    const match = VERDICT_CLASSES.find(([prefix]) => word === prefix);
    if (match) cell.classList.add(match[1]);
  }
}

export function buildSlide(slide, total) {
  const section = element("section", `slide ${layoutOf(slide)}`);
  section.dataset.number = slide.number;

  if (slide.isTitle) {
    section.append(element("div", "title-block", slide.textHtml));
    section.append(element("div", "title-meta", "ICLR 2025 · Ye et al."));
    applyFrenchSpacing(section);
    return section;
  }

  const header = element("header", "slide-header");
  header.append(element("span", "slide-kicker", `Diapositive ${slide.number}`));
  header.append(element("h1", "slide-title", slide.title));
  section.append(header);

  const body = element("div", "slide-body");
  const text = element("div", "slide-text", slide.textHtml);
  if (text.textContent.length > TEXT_HEAVY_CHARS) section.classList.add("text-heavy");
  body.append(text);
  if (slide.visuals.length) {
    const visuals = element("div", "slide-visuals");
    if (slide.visuals.every((visual) => visual.type === "table")) {
      visuals.classList.add("only-tables");
    }
    slide.visuals.forEach((visual) => visuals.append(buildVisual(visual)));
    body.append(visuals);
  }
  section.append(body);

  const footer = element("footer", "slide-footer");
  footer.append(element("span", "", FOOTER_TEXT));
  footer.append(element("span", "slide-count", `${slide.number} / ${total}`));
  section.append(footer);
  applyFrenchSpacing(section);
  markVerdicts(section);
  return section;
}

async function renderMermaid(root) {
  const boxes = [...root.querySelectorAll(".visual-mermaid[data-code]")];
  for (const box of boxes) {
    mermaidCounter += 1;
    try {
      const { svg } = await mermaid.render(`mermaid-${mermaidCounter}`, box.dataset.code);
      box.innerHTML = svg;
      const node = box.querySelector("svg");
      node.removeAttribute("height");
      node.style.maxWidth = "100%";
    } catch (error) {
      box.textContent = `Diagramme invalide : ${error.message}`;
    }
    delete box.dataset.code;
  }
}

function overflows(node) {
  return node.scrollHeight > node.clientHeight + 1 || node.scrollWidth > node.clientWidth + 1;
}

function shrinkUntilFits(container, target, minPx) {
  let size = parseFloat(getComputedStyle(target).fontSize);
  while (overflows(container) && size > minPx) {
    size -= 1;
    target.style.fontSize = `${size}px`;
  }
}

function fitSlide(section) {
  const text = section.querySelector(".slide-text");
  if (text) shrinkUntilFits(text, text, TEXT_MIN_PX);
  const visuals = section.querySelector(".slide-visuals");
  if (visuals) {
    for (const table of visuals.querySelectorAll(".visual-table")) {
      shrinkUntilFits(visuals, table, TABLE_MIN_PX);
    }
  }
}

function imagesLoaded(root) {
  const images = [...root.querySelectorAll("img")];
  return Promise.all(
    images.map((img) =>
      img.complete
        ? Promise.resolve()
        : new Promise((resolve) => {
            img.addEventListener("load", resolve, { once: true });
            img.addEventListener("error", resolve, { once: true });
          })
    )
  );
}

/* Slides must be laid out (not display: none) when this runs. */
export async function finishDeck(deck) {
  await document.fonts.ready;
  await renderMermaid(deck);
  await imagesLoaded(deck);
  deck.querySelectorAll(".slide").forEach(fitSlide);
}

export function fitStage(stage, viewport = window) {
  const width = viewport.innerWidth;
  const height = viewport.innerHeight;
  const scale = Math.min(width / SLIDE_WIDTH, height / SLIDE_HEIGHT);
  const left = (width - SLIDE_WIDTH * scale) / 2;
  const top = (height - SLIDE_HEIGHT * scale) / 2;
  stage.style.transform = `translate(${left}px, ${top}px) scale(${scale})`;
  return scale;
}
