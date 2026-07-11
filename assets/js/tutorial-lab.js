(() => {
  const lab = document.querySelector("[data-task-lab]");
  if (!lab) return;

  const choices = Array.from(lab.querySelectorAll('input[name="model-client"]'));
  const checkButton = lab.querySelector("[data-check-answer]");
  const codeSlot = lab.querySelector("[data-code-slot]");
  const status = lab.querySelector("[data-task-status]");
  const journal = lab.querySelector("[data-task-journal]");
  const stages = Array.from(document.querySelectorAll("[data-task-stage]"));

  const markProgress = (lastComplete) => {
    stages.forEach((stage, index) => {
      const state = index <= lastComplete ? "complete" : index === lastComplete + 1 ? "current" : "pending";
      stage.dataset.state = state;
      if (state === "current") {
        stage.setAttribute("aria-current", "step");
      } else {
        stage.removeAttribute("aria-current");
      }
    });
  };

  choices.forEach((choice) => {
    choice.addEventListener("change", () => {
      codeSlot.textContent = choice.value;
      checkButton.disabled = false;
      journal.hidden = true;
      status.dataset.state = "ready";
      status.textContent = "已补上代码。现在先说出它负责什么，再检查。";
      markProgress(1);
    });
  });

  checkButton.addEventListener("click", () => {
    const selected = choices.find((choice) => choice.checked);
    if (!selected) return;

    if (selected.value === "ChatOpenAI") {
      status.dataset.state = "pass";
      status.textContent = "正确。ChatOpenAI 创建模型客户端；PromptTemplate 组织输入，LangSmith 记录与评估运行。";
      journal.hidden = false;
      markProgress(3);
      return;
    }

    status.dataset.state = "retry";
    status.textContent = selected.value === "PromptTemplate"
      ? "再试一次：PromptTemplate 像填写目的地的表单，本身不负责把请求送进模型。"
      : "再试一次：LangSmith 像行程记录与质检台，不是发车的模型客户端。";
    journal.hidden = true;
    markProgress(1);
  });
})();
