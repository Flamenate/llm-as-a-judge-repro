import { applyFrenchSpacing, loadSlides } from "./parser.js";
import { handleNavigationKey, openChannel, slideFromHash } from "./sync.js";

const SPEECH_MIN_PX = 16;
const SPEECH_MAX_PX = 44;

const miniature = document.getElementById("miniature");
const counter = document.getElementById("counter");
const title = document.getElementById("title");
const speech = document.getElementById("speech");
const notes = document.getElementById("notes");
const timerLabel = document.getElementById("timer");
const clock = document.getElementById("clock");

let slides = [];
let current = 0;
let black = false;
let channel = null;

const timer = {
  startedAt: Date.now(),
  elapsedBefore: 0,
  running: true,
  elapsed() {
    return this.elapsedBefore + (this.running ? Date.now() - this.startedAt : 0);
  },
  toggle() {
    if (this.running) {
      this.elapsedBefore = this.elapsed();
    } else {
      this.startedAt = Date.now();
    }
    this.running = !this.running;
  },
  reset() {
    this.elapsedBefore = 0;
    this.startedAt = Date.now();
  },
};

function pad(value) {
  return String(value).padStart(2, "0");
}

function tick() {
  const seconds = Math.floor(timer.elapsed() / 1000);
  timerLabel.textContent = `${pad(Math.floor(seconds / 60))}:${pad(seconds % 60)}`;
  timerLabel.classList.toggle("paused", !timer.running);
  const now = new Date();
  clock.textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}`;
}

function render() {
  const slide = slides[current];
  counter.innerHTML = `${current + 1} <small>/ ${slides.length}</small>`;
  title.innerHTML = `<span>${slide.number}.</span>${slide.title}`;
  speech.innerHTML = slide.speechHtml;
  notes.innerHTML = slide.notesHtml;
  applyFrenchSpacing(speech);
  applyFrenchSpacing(title);
  speech.parentElement.scrollTop = 0;
  miniature.contentWindow.location.hash = `#${current + 1}`;
  document.title = `Présentateur — ${current + 1}/${slides.length}`;
  history.replaceState(null, "", `#${current + 1}`);
}

function goTo(index, { broadcast = true } = {}) {
  const target = Math.min(Math.max(index, 0), slides.length - 1);
  const changed = target !== current;
  current = target;
  render();
  if (broadcast && changed) channel.send({ type: "goto", index: current });
}

function setSpeechSize(delta) {
  const size = parseFloat(getComputedStyle(speech).fontSize) + delta;
  speech.style.fontSize = `${Math.min(Math.max(size, SPEECH_MIN_PX), SPEECH_MAX_PX)}px`;
}

const actions = {
  next: () => goTo(current + 1),
  previous: () => goTo(current - 1),
  goTo: (index) => goTo(index),
  toggleBlack: () => {
    black = !black;
    channel.send({ type: "black", on: black });
  },
};

function bindControls() {
  const digits = { value: "" };
  window.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "t" || event.key === "T") {
      timer.toggle();
      tick();
    } else if (event.key === "+" || event.key === "=") {
      setSpeechSize(2);
    } else if (event.key === "-") {
      setSpeechSize(-2);
    } else if (!handleNavigationKey(event, actions, digits)) {
      return;
    }
    event.preventDefault();
  });
  document.getElementById("prev").addEventListener("click", actions.previous);
  document.getElementById("next").addEventListener("click", actions.next);
  document.getElementById("timer-toggle").addEventListener("click", () => {
    timer.toggle();
    tick();
  });
  document.getElementById("timer-reset").addEventListener("click", () => {
    timer.reset();
    tick();
  });
}

function bindChannel() {
  channel = openChannel((message) => {
    if ((message.type === "goto" || message.type === "state") && Number.isInteger(message.index)) {
      goTo(message.index, { broadcast: false });
    } else if (message.type === "black") {
      black = Boolean(message.on);
    }
  });
}

async function start() {
  slides = await loadSlides();
  current = slideFromHash(slides.length);
  miniature.src = `index.html?embed#${current + 1}`;
  await new Promise((resolve) => miniature.addEventListener("load", resolve, { once: true }));
  bindChannel();
  bindControls();
  render();
  channel.send({ type: "hello" });
  tick();
  setInterval(tick, 500);
}

start().catch((error) => {
  speech.textContent = `Erreur : ${error.message}`;
  console.error(error);
});
