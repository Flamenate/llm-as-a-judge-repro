/* global marked */

export const MARKDOWN_URL = "../presentation_fr.md";

const SLIDE_HEADING = /^## Diapositive (\d+)\s*[—–-]\s*(.+)$/gm;
const SECTION_HEADING = /^### (Texte|Visuels|Discours)\s*$/gm;

function configureMarked(baseUrl) {
  marked.use({
    gfm: true,
    renderer: {
      image(href, title, text) {
        const src = new URL(href, baseUrl).href;
        const titleAttr = title ? ` title="${title}"` : "";
        return `<img src="${src}" alt="${text}"${titleAttr}>`;
      },
    },
  });
}

function stripSeparators(markdown) {
  return markdown.replace(/^\s*---\s*$/gm, "").trim();
}

function splitSections(body) {
  const sections = { Texte: "", Visuels: "", Discours: "" };
  const matches = [...body.matchAll(SECTION_HEADING)];
  matches.forEach((match, i) => {
    const start = match.index + match[0].length;
    const end = i + 1 < matches.length ? matches[i + 1].index : body.length;
    sections[match[1]] = stripSeparators(body.slice(start, end));
  });
  return sections;
}

function renderTokens(tokens, links) {
  const list = [...tokens];
  list.links = links;
  return marked.parser(list);
}

function isOnlyCode(token) {
  const parts = (token.tokens || []).filter(
    (part) => !(part.type === "text" && !part.text.trim())
  );
  return parts.length > 0 && parts.every((part) => part.type === "codespan");
}

/* Images, Mermaid blocks and tables go on the slide; prose goes to the presenter. */
function splitVisuals(markdown, baseUrl) {
  const tokens = marked.lexer(markdown);
  const visuals = [];
  const notes = [];
  for (const token of tokens) {
    if (token.type === "code" && token.lang === "mermaid") {
      visuals.push({ type: "mermaid", code: token.text });
    } else if (token.type === "table") {
      visuals.push({ type: "table", html: renderTokens([token], tokens.links) });
    } else if (token.type === "paragraph") {
      const images = (token.tokens || []).filter((part) => part.type === "image");
      if (images.length) {
        for (const image of images) {
          visuals.push({
            type: "image",
            src: new URL(image.href, baseUrl).href,
            alt: image.text,
          });
        }
      } else if (!isOnlyCode(token)) {
        notes.push(token);
      }
    } else if (token.type !== "space") {
      notes.push(token);
    }
  }
  return {
    visuals,
    notesHtml: notes.length ? renderTokens(notes, tokens.links) : "",
  };
}

const NBSP = "\u00a0";
const SPACE_BEFORE = / ([;:!?%»])/g;
const SPACE_AFTER = /(«) /g;

/* French typography: keep punctuation on the same line as the preceding word. */
export function applyFrenchSpacing(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      return node.parentElement.closest("code, pre, svg, .visual-mermaid")
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT;
    },
  });
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    node.nodeValue = node.nodeValue
      .replace(SPACE_BEFORE, `${NBSP}$1`)
      .replace(SPACE_AFTER, `$1${NBSP}`);
  }
}

export async function loadSlides(markdownUrl = MARKDOWN_URL) {
  const baseUrl = new URL(markdownUrl, window.location.href);
  const response = await fetch(baseUrl, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Impossible de charger ${baseUrl} (${response.status})`);
  }
  const source = (await response.text()).replace(/\r\n?/g, "\n");
  configureMarked(baseUrl);

  const headings = [...source.matchAll(SLIDE_HEADING)];
  return headings.map((match, i) => {
    const start = match.index + match[0].length;
    const end = i + 1 < headings.length ? headings[i + 1].index : source.length;
    const sections = splitSections(source.slice(start, end));
    const { visuals, notesHtml } = splitVisuals(sections.Visuels, baseUrl);
    const isTitle = i === 0;
    return {
      number: Number(match[1]),
      title: match[2].trim(),
      isTitle,
      textHtml: marked.parse(sections.Texte, { breaks: isTitle }),
      visuals,
      notesHtml,
      speechHtml: marked.parse(sections.Discours),
    };
  });
}
