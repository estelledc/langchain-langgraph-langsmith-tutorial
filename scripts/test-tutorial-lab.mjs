import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

class FakeElement {
  constructor(value = "") {
    this.value = value;
    this.checked = false;
    this.disabled = true;
    this.hidden = true;
    this.textContent = "";
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
  }

  addEventListener(type, listener) { this.listeners.set(type, listener); }
  dispatch(type) { this.listeners.get(type)?.(); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  removeAttribute(name) { this.attributes.delete(name); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
}

const choices = ["PromptTemplate", "ChatOpenAI", "LangSmith"].map((value) => new FakeElement(value));
const button = new FakeElement();
const codeSlot = new FakeElement();
const status = new FakeElement();
const journal = new FakeElement();
const stages = Array.from({ length: 4 }, () => new FakeElement());
const lab = {
  querySelectorAll: () => choices,
  querySelector: (selector) => ({
    "[data-check-answer]": button,
    "[data-code-slot]": codeSlot,
    "[data-task-status]": status,
    "[data-task-journal]": journal,
  })[selector],
};
const document = {
  querySelector: () => lab,
  querySelectorAll: () => stages,
};

const controller = readFileSync(new URL("../assets/js/tutorial-lab.js", import.meta.url), "utf8");
vm.runInNewContext(controller, { document });

choices[0].checked = true;
choices[0].dispatch("change");
assert.equal(codeSlot.textContent, "PromptTemplate");
assert.equal(button.disabled, false);
assert.equal(stages[2].getAttribute("aria-current"), "step");
button.dispatch("click");
assert.equal(status.dataset.state, "retry");
assert.equal(journal.hidden, true);

choices[0].checked = false;
choices[1].checked = true;
choices[1].dispatch("change");
button.dispatch("click");
assert.equal(status.dataset.state, "pass");
assert.match(status.textContent, /ChatOpenAI/);
assert.equal(journal.hidden, false);
assert.ok(stages.every((stage) => stage.dataset.state === "complete"));
assert.ok(stages.every((stage) => stage.getAttribute("aria-current") === null));

console.log("Task-lab behavior: PASS (wrong answer, correct answer, journal, ARIA progress)");
