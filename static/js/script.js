// ---------- Dark / Light mode ----------
(function initTheme() {
  const saved = localStorage.getItem("theme");
  const theme = saved || "light";
  document.body.setAttribute("data-theme", theme);
  updateToggleLabel(theme);
})();

function updateToggleLabel(theme) {
  const btn = document.getElementById("themeToggle");
  if (!btn) return;
  btn.textContent = theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
}

document.getElementById("themeToggle")?.addEventListener("click", () => {
  const current = document.body.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.body.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  updateToggleLabel(next);
});

// ---------- Client-side form validation + loading animation ----------
const form = document.getElementById("creditForm");
if (form) {
  form.addEventListener("submit", (e) => {
    let valid = true;

    // Clear previous errors
    document.querySelectorAll(".error-msg").forEach((el) => (el.textContent = ""));

    form.querySelectorAll("input[required], select[required]").forEach((field) => {
      const errorEl = document.querySelector(`.error-msg[data-for="${field.id}"]`);
      if (!field.value || field.value.trim() === "") {
        valid = false;
        if (errorEl) errorEl.textContent = "This field is required.";
        return;
      }
      if (field.type === "number") {
        const val = parseFloat(field.value);
        const min = field.min !== "" ? parseFloat(field.min) : -Infinity;
        const max = field.max !== "" ? parseFloat(field.max) : Infinity;
        if (isNaN(val) || val < min || val > max) {
          valid = false;
          if (errorEl) errorEl.textContent = `Enter a value between ${field.min} and ${field.max}.`;
        }
      }
    });

    if (!valid) {
      e.preventDefault();
      return;
    }

    // Show loading overlay + disable button while the server processes the request
    const overlay = document.getElementById("loadingOverlay");
    const submitBtn = document.getElementById("submitBtn");
    if (overlay) overlay.classList.add("active");
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Predicting...";
    }
  });
}

// ---------- Model metrics dashboard (index.html) ----------
const metricsSummaryEl = document.getElementById("metricsSummary");
const metricsChartEl = document.getElementById("metricsChart");

if (metricsSummaryEl && metricsChartEl) {
  fetch("/api/metrics")
    .then((res) => res.json())
    .then((data) => {
      if (data.error) {
        metricsSummaryEl.innerHTML = `<p>${data.error}</p>`;
        return;
      }

      const modelNames = Object.keys(data);

      // Find the best model by ROC-AUC to highlight in summary cards
      let bestModel = modelNames[0];
      modelNames.forEach((name) => {
        if (data[name].roc_auc > data[bestModel].roc_auc) bestModel = name;
      });

      const best = data[bestModel];
      metricsSummaryEl.innerHTML = `
        <div class="metric-card"><div class="metric-value">${bestModel}</div><div class="metric-label">Best Model</div></div>
        <div class="metric-card"><div class="metric-value">${(best.accuracy * 100).toFixed(1)}%</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-card"><div class="metric-value">${(best.f1_score * 100).toFixed(1)}%</div><div class="metric-label">F1 Score</div></div>
        <div class="metric-card"><div class="metric-value">${(best.roc_auc * 100).toFixed(1)}%</div><div class="metric-label">ROC-AUC</div></div>
      `;

      const metricKeys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"];
      const colors = ["#3F51B5", "#7C4DFF", "#2196F3", "#FF9800", "#4CAF50"];

      new Chart(metricsChartEl, {
        type: "bar",
        data: {
          labels: modelNames,
          datasets: metricKeys.map((key, i) => ({
            label: key.replace("_", " ").toUpperCase(),
            data: modelNames.map((m) => data[m][key]),
            backgroundColor: colors[i],
          })),
        },
        options: {
          responsive: true,
          scales: { y: { beginAtZero: true, max: 1 } },
          plugins: { legend: { position: "bottom" } },
        },
      });
    })
    .catch(() => {
      metricsSummaryEl.innerHTML = "<p>Could not load model metrics. Make sure app.py is running and train.py has been executed.</p>";
    });
}

// ---------- Confidence ring (result.html) ----------
const ringCanvas = document.getElementById("confidenceRing");
if (ringCanvas && typeof window.RESULT_CONFIDENCE !== "undefined") {
  const confidence = window.RESULT_CONFIDENCE;
  const isHighRisk = window.RESULT_IS_HIGH_RISK;
  const color = isHighRisk ? "#c62828" : "#2e7d32";

  new Chart(ringCanvas, {
    type: "doughnut",
    data: {
      labels: ["Confidence", "Remaining"],
      datasets: [
        {
          data: [confidence, 100 - confidence],
          backgroundColor: [color, "rgba(150,150,150,0.15)"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      cutout: "75%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    },
  });
}
