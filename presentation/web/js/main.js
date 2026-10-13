import { loadSlides } from "./parser.js";
import { buildSlide, finishDeck, fitStage, initMermaid } from "./render.js";
import { handleNavigationKey, openChannel, slideFromHash } from "./sync.js";

const params = new URLSearchParams(window.location.search);
const EMBED = params.has("embed");
const CONTROLS_IDLE_MS = 2200;
const SWIPE_PX = 60;

const stage = document.getElementById("stage");
const progress = document.getElementById("progress");
const controls = document.getElementById("controls");
const blackout = document.getElementById("blackout");
const status = document.getElementById("status");

let slides = [];
let sections = [];
let current = 0;
let channel = null;

document.body.classList.toggle("embed", EMBED);

function clamp(index) {
  return Math.min(Math.max(index, 0), slides.length - 1);
}

function render() {
  sections.forEach((section, i) => {
    section.classList.toggle("active", i === current);
    section.classList.toggle("past", i < current);
    section.setAttribute("aria-hidden", i === current ? "false" : "true");
  });
  [...progress.children].forEach((segment, i) => {
    segment.classList.toggle("done", i <= current);
    segment.classList.toggle("current", i === current);
  });
  document.title = `${current + 1}/${slides.length} — ${slides[current].title}`;
  const hash = `#${current + 1}`;
  if (window.location.hash !== hash) {
    history.replaceState(null, "", hash);
  }
}

function goTo(index, { broadcast = true } = {}) {
  const target = clamp(index);
  const changed = target !== current;
  current = target;
  render();
  if (broadcast && changed && channel) {
    channel.send({ type: "goto", index: current });
  }
}

function setBlack(on, { broadcast = true } = {}) {
  blackout.classList.toggle("on", on);
  if (broadcast && channel) channel.send({ type: "black", on });
}

const actions = {
  next: () => goTo(current + 1),
  previous: () => goTo(current - 1),
  goTo: (index) => goTo(index),
  toggleBlack: () => setBlack(!blackout.classList.contains("on")),
};

function openPresenter() {
  window.open(
    `presenter.html#${current + 1}`,
    "calm-presenter",
    "width=1400,height=860"
  );
}

function toggleFullscreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    document.documentElement.requestFullscreen();
  }
}

function buildProgress() {
  progress.innerHTML = "";
  slides.forEach((slide, i) => {
    const segment = document.createElement("button");
    segment.type = "button";
    segment.className = "segment";
    segment.title = `${slide.number}. ${slide.title}`;
    segment.addEventListener("click", (event) => {
      event.stopPropagation();
      goTo(i);
    });
    progress.append(segment);
  });
}

function bindKeyboard() {
  const digits = { value: "" };
  window.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "f" || event.key === "F") {
      toggleFullscreen();
    } else if (event.key === "p" || event.key === "P") {
      openPresenter();
    } else if (!handleNavigationKey(event, actions, digits)) {
      return;
    }
    event.preventDefault();
  });
}

function bindMouse() {
  document.getElementById("viewport").addEventListener("click", (event) => {
    if (event.target.closest("button, a")) return;
    if (window.getSelection().toString()) return;
    if (event.clientX < window.innerWidth * 0.25) {
      actions.previous();
    } else {
      actions.next();
    }
  });

  controls.querySelector("[data-action=prev]").addEventListener("click", actions.previous);
  controls.querySelector("[data-action=next]").addEventListener("click", actions.next);
  controls.querySelector("[data-action=presenter]").addEventListener("click", openPresenter);
  controls.querySelector("[data-action=fullscreen]").addEventListener("click", toggleFullscreen);

  let idleTimer = null;
  window.addEventListener("mousemove", () => {
    document.body.classList.add("show-controls");
    clearTimeout(idleTimer);
    idleTimer = setTimeout(
      () => document.body.classList.remove("show-controls"),
      CONTROLS_IDLE_MS
    );
  });

  let touchX = null;
  window.addEventListener("touchstart", (event) => {
    touchX = event.touches[0].clientX;
  }, { passive: true });
  window.addEventListener("touchend", (event) => {
    if (touchX === null) return;
    const dx = event.changedTouches[0].clientX - touchX;
    touchX = null;
    if (dx < -SWIPE_PX) actions.next();
    if (dx > SWIPE_PX) actions.previous();
  });
}

function bindChannel() {
  channel = openChannel((message) => {
    if (message.type === "goto" && Number.isInteger(message.index)) {
      goTo(message.index, { broadcast: false });
    } else if (message.type === "black") {
      setBlack(Boolean(message.on), { broadcast: false });
    } else if (message.type === "hello") {
      channel.send({ type: "state", index: current });
    }
  });
}

async function start() {
  initMermaid();
  slides = await loadSlides();
  sections = slides.map((slide) => buildSlide(slide, slides.length));
  stage.append(...sections);
  current = slideFromHash(slides.length);

  fitStage(stage);
  window.addEventListener("resize", () => fitStage(stage));
  await finishDeck(stage);

  if (!EMBED) {
    buildProgress();
    bindKeyboard();
    bindMouse();
    bindChannel();
  }
  window.addEventListener("hashchange", () => {
    goTo(slideFromHash(slides.length), { broadcast: !EMBED });
  });
  render();
  status.remove();
  document.body.classList.add("ready");
}

start().catch((error) => {
  status.textContent = `Erreur : ${error.message}. Lancez un serveur depuis la racine du dépôt : python -m http.server 8000`;
  status.classList.add("error");
  console.error(error);
});
