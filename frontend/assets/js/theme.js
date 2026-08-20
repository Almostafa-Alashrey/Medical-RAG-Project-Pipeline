(() => {
  const storageKey = "pressiq-theme";
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function getTheme() {
    return localStorage.getItem(storageKey) || (darkQuery.matches ? "dark" : "light");
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const button = document.querySelector(".theme-toggle");
    if (button) {
      const isDark = theme === "dark";
      button.setAttribute("aria-pressed", String(isDark));
      button.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
      button.title = isDark ? "Switch to light mode" : "Switch to dark mode";
      button.textContent = isDark ? "☀" : "☾";
    }
  }

  // Run immediately to prevent a light-theme flash on page load.
  applyTheme(getTheme());

  document.addEventListener("DOMContentLoaded", () => {
    const navRight = document.querySelector(".nav-right");
    if (!navRight) return;

    const button = document.createElement("button");
    button.className = "theme-toggle";
    button.type = "button";
    button.addEventListener("click", () => {
      const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      localStorage.setItem(storageKey, nextTheme);
      applyTheme(nextTheme);
    });
    navRight.prepend(button);
    applyTheme(getTheme());
  });
})();
