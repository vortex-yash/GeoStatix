// Replace these two URLs before publishing.
const CONFIG = {
  appUrl: "https://geostatix.streamlit.app",
  githubUrl: "https://github.com/vortex-yash/GeoStatix"
};

document.querySelectorAll('[data-app-link]').forEach(link => {
  link.href = CONFIG.appUrl;
  link.addEventListener('click', event => {
    if (CONFIG.appUrl === '#') {
      event.preventDefault();
      alert('Add your deployed GeoStatix Streamlit URL in website/script.js first.');
    }
  });
});

const githubLinks = document.querySelectorAll('a[href="https://github.com/"]');
githubLinks.forEach(link => link.href = CONFIG.githubUrl);

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add('in-view');
  });
}, { threshold: 0.08 });

document.querySelectorAll('.cap-card, .step, .stack-list div').forEach(el => observer.observe(el));
