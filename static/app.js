const authView = document.querySelector("#auth-view");
const dashboardView = document.querySelector("#dashboard-view");
const authForm = document.querySelector("#auth-form");
const authMessage = document.querySelector("#auth-message");
const predictionForm = document.querySelector("#prediction-form");
const predictionMessage = document.querySelector("#prediction-message");
let authMode = "signin";
let currentUser = null;

function showMessage(node, message = "", kind = "error") {
  node.textContent = message;
  node.classList.toggle("success", kind === "success");
}

function detailText(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg || "Please check the form values.").join(" ");
  return "Something went wrong. Please try again.";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(detailText(data.detail || data.message));
    error.status = response.status;
    throw error;
  }
  return data;
}

function setAuthMode(mode) {
  authMode = mode;
  const creating = mode === "register";
  document.querySelector("#signin-tab").classList.toggle("active", !creating);
  document.querySelector("#register-tab").classList.toggle("active", creating);
  document.querySelector("#signin-tab").setAttribute("aria-selected", String(!creating));
  document.querySelector("#register-tab").setAttribute("aria-selected", String(creating));
  document.querySelector("#auth-title").textContent = creating ? "Create your account" : "Enter dashboard";
  document.querySelector("#auth-description").textContent = creating
    ? "Create an account to unlock the prediction and model performance dashboard."
    : "Create a real account with your email, then sign in to unlock predictions and metrics.";
  document.querySelector("#confirm-field").hidden = !creating;
  document.querySelector("#auth-confirm").required = creating;
  document.querySelector("#auth-password").autocomplete = creating ? "new-password" : "current-password";
  document.querySelector("#auth-password").placeholder = creating ? "Minimum 8 characters" : "Enter your password";
  document.querySelector("#auth-submit").textContent = creating ? "Create Account" : "Log In";
  document.querySelector("#auth-helper").textContent = creating
    ? "Your password is stored as a one-way bcrypt hash on this server."
    : "Sign in with the email and password from your account.";
  showMessage(authMessage);
}

document.querySelector("#signin-tab").addEventListener("click", () => setAuthMode("signin"));
document.querySelector("#register-tab").addEventListener("click", () => setAuthMode("register"));

function showAuth() {
  authView.hidden = false;
  dashboardView.hidden = true;
  currentUser = null;
  setAuthMode("signin");
}

function showDashboard(user) {
  currentUser = user;
  document.querySelector("#signed-in-email").textContent = user.email;
  authView.hidden = true;
  dashboardView.hidden = false;
  showMessage(authMessage);
  loadMetadata();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  showMessage(authMessage);
  const email = document.querySelector("#auth-email").value.trim();
  const password = document.querySelector("#auth-password").value;
  const button = document.querySelector("#auth-submit");
  if (!email || !document.querySelector("#auth-email").checkValidity()) {
    showMessage(authMessage, "Enter a valid email address.");
    return;
  }
  if (authMode === "register" && password.length < 8) {
    showMessage(authMessage, "Use a password with at least 8 characters.");
    return;
  }
  if (authMode === "register" && password !== document.querySelector("#auth-confirm").value) {
    showMessage(authMessage, "The passwords do not match.");
    return;
  }
  button.disabled = true;
  button.textContent = authMode === "register" ? "Creating account…" : "Signing in…";
  try {
    if (authMode === "register") {
      await api("/auth/register", { method: "POST", body: JSON.stringify({ email, password, confirm_password: document.querySelector("#auth-confirm").value }) });
      setAuthMode("signin");
      document.querySelector("#auth-email").value = email;
      document.querySelector("#auth-password").value = "";
      showMessage(authMessage, "Account created. Sign in with your new credentials.", "success");
    } else {
      const result = await api("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
      showDashboard(result.user);
    }
  } catch (error) {
    showMessage(authMessage, error.message || "Unable to complete this request.");
  } finally {
    button.disabled = false;
    button.textContent = authMode === "register" ? "Create Account" : "Log In";
  }
});

function metricLabel(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function addMetricRow(parent, label, value) {
  const row = document.createElement("div");
  row.className = "metric-row";
  const name = document.createElement("span");
  name.textContent = label;
  const score = document.createElement("strong");
  score.textContent = metricLabel(value);
  row.append(name, score);
  parent.append(row);
}

function renderModels(metadata) {
  const list = document.querySelector("#model-cards");
  list.replaceChildren();
  for (const [name, metrics] of Object.entries(metadata.metrics || {})) {
    const selected = name === metadata.best_model;
    const card = document.createElement("article");
    card.className = `model-card${selected ? " selected" : ""}`;
    const head = document.createElement("div");
    head.className = "model-card-head";
    const title = document.createElement("h3");
    title.textContent = name;
    head.append(title);
    if (selected) {
      const badge = document.createElement("span");
      badge.className = "selected-tag";
      badge.textContent = "Selected";
      head.append(badge);
    }
    card.append(head);
    addMetricRow(card, "Test accuracy", metrics.accuracy);
    addMetricRow(card, "Precision", metrics.precision);
    addMetricRow(card, "Recall", metrics.recall);
    addMetricRow(card, "F1 score", metrics.f1);
    addMetricRow(card, "5-fold CV accuracy", metrics.cv_accuracy);
    list.append(card);
  }
}

function renderSignals(signals = []) {
  const container = document.querySelector("#feature-signals");
  container.replaceChildren();
  for (const signal of signals) {
    const chip = document.createElement("span");
    chip.className = "signal-chip";
    const label = document.createElement("span");
    label.textContent = signal.label || signal.name;
    const score = document.createElement("strong");
    score.textContent = Number(signal.score || 0).toFixed(2);
    chip.append(label, score);
    container.append(chip);
  }
}

async function loadMetadata() {
  try {
    const metadata = await api("/metadata");
    document.querySelector("#best-model-name").textContent = metadata.best_model || "—";
    document.querySelector("#dataset-rows").textContent = Number(metadata.dataset_rows || 0).toLocaleString();
    document.querySelector("#dataset-source").textContent = metadata.synthetic_fallback ? "Demo fallback" : "Public survey";
    document.querySelector("#demo-banner").hidden = !metadata.synthetic_fallback;
    document.querySelector("#selected-model-heading").textContent = metadata.best_model || "Selected model";
    document.querySelector("#selected-model-copy").textContent = metadata.synthetic_fallback
      ? "This demo run uses generated fallback records because the public survey could not be loaded. Its metrics do not describe patient data."
      : `${metadata.best_model} was selected by five-fold cross-validation on the ${Number(metadata.dataset_rows || 0).toLocaleString()}-row survey dataset.`;
    renderModels(metadata);
    renderSignals(metadata.feature_importances);
  } catch (error) {
    if (error.status === 401) showAuth();
    else {
      document.querySelector("#model-cards").textContent = error.message || "Model information is temporarily unavailable.";
      document.querySelector("#selected-model-copy").textContent = "Model information is temporarily unavailable.";
    }
  }
}

function setResult(result) {
  const card = document.querySelector("#result-card");
  const high = result.label === 1;
  const percent = Number(result.probability_percent || 0);
  card.classList.toggle("is-low", !high);
  const badge = document.querySelector("#result-badge");
  badge.className = `result-badge ${high ? "high" : "low"}`;
  badge.textContent = high ? "Higher survey match" : "Lower survey match";
  document.querySelector("#result-percent").textContent = `${percent.toFixed(1)}%`;
  document.querySelector("#result-title").textContent = high ? "Cancer-associated pattern detected" : "No-cancer pattern detected";
  document.querySelector("#result-progress").style.width = `${Math.max(0, Math.min(100, percent))}%`;
  document.querySelector("#result-description").textContent = high
    ? "The selected model matched this record to the cancer-associated class in its training survey. This is not a diagnosis."
    : "The selected model matched this record to the no-cancer class in its training survey. A lower score cannot rule out cancer.";
  document.querySelector("#result-model").textContent = `Selected classifier: ${result.model}${result.synthetic_fallback ? " · demo fallback data" : ""}`;
}

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  showMessage(predictionMessage);
  const ageInput = document.querySelector("#age");
  if (!ageInput.value || Number(ageInput.value) < 18 || Number(ageInput.value) > 120) {
    showMessage(predictionMessage, "Enter an age between 18 and 120.");
    ageInput.focus();
    return;
  }
  const fields = [...predictionForm.querySelectorAll("select")];
  const incomplete = fields.find((field) => !field.value);
  if (incomplete) {
    showMessage(predictionMessage, "Please complete all fields before submitting.");
    incomplete.focus();
    return;
  }
  const payload = Object.fromEntries(new FormData(predictionForm).entries());
  payload.age = Number(payload.age);
  const button = document.querySelector("#predict-button");
  button.disabled = true;
  button.firstChild.textContent = "Calculating… ";
  try {
    const result = await api("/predict", { method: "POST", body: JSON.stringify(payload) });
    setResult(result);
    document.querySelector("#result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    if (error.status === 401) showAuth();
    else showMessage(predictionMessage, error.message || "Unable to create a screening estimate.");
  } finally {
    button.disabled = false;
    button.firstChild.textContent = "Run screening estimate ";
  }
});

predictionForm.addEventListener("reset", () => {
  window.setTimeout(() => {
    showMessage(predictionMessage);
    document.querySelector("#result-card").classList.remove("is-low");
    document.querySelector("#result-badge").className = "result-badge neutral";
    document.querySelector("#result-badge").textContent = "Awaiting inputs";
    document.querySelector("#result-percent").textContent = "—";
    document.querySelector("#result-title").textContent = "Your result will appear here";
    document.querySelector("#result-progress").style.width = "0%";
    document.querySelector("#result-description").textContent = "Complete the questionnaire to see the selected model’s survey classification.";
    document.querySelector("#result-model").textContent = "";
  }, 0);
});

document.querySelector("#logout-button").addEventListener("click", async () => {
  const button = document.querySelector("#logout-button");
  button.disabled = true;
  try {
    await api("/auth/logout", { method: "POST", body: "{}" });
  } catch (error) {
    if (error.status !== 401) {
      button.disabled = false;
      return;
    }
  }
  showAuth();
  button.disabled = false;
});

(async function restoreSession() {
  try {
    const result = await api("/auth/me");
    showDashboard(result.user);
  } catch (_) {
    showAuth();
  }
})();
