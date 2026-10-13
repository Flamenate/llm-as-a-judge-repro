const CHANNEL_NAME = "calm-presentation";

/*
 * Messages:
 *   { type: "goto", index }   a window moved to slide index (0-based)
 *   { type: "hello" }         a new window asks for the current slide
 *   { type: "state", index }  reply to hello from the audience window
 *   { type: "black", on }     black screen toggled
 */
export function openChannel(onMessage) {
  const channel = new BroadcastChannel(CHANNEL_NAME);
  channel.addEventListener("message", (event) => onMessage(event.data || {}));
  return {
    send(message) {
      channel.postMessage(message);
    },
  };
}

export function slideFromHash(total) {
  const value = parseInt(window.location.hash.replace("#", ""), 10);
  if (Number.isNaN(value)) return 0;
  return Math.min(Math.max(value - 1, 0), total - 1);
}

/* Shared keyboard map. Returns true when the key was handled. */
export function handleNavigationKey(event, actions, digits) {
  const key = event.key;
  if (/^[0-9]$/.test(key)) {
    digits.value += key;
    return true;
  }
  if (key === "Enter" && digits.value) {
    actions.goTo(parseInt(digits.value, 10) - 1);
    digits.value = "";
    return true;
  }
  digits.value = "";
  switch (key) {
    case "ArrowRight":
    case "ArrowDown":
    case "PageDown":
    case " ":
    case "Enter":
      actions.next();
      return true;
    case "ArrowLeft":
    case "ArrowUp":
    case "PageUp":
    case "Backspace":
      actions.previous();
      return true;
    case "Home":
      actions.goTo(0);
      return true;
    case "End":
      actions.goTo(Infinity);
      return true;
    case "b":
    case "B":
      actions.toggleBlack();
      return true;
    default:
      return false;
  }
}
