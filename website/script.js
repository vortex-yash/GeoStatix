const CONFIG = {
  appUrl: "https://geostatix.streamlit.app",
  githubUrl: "https://github.com/vortex-yash/GeoStatix"
};

document.querySelectorAll('[data-app-link]').forEach(link => link.href = CONFIG.appUrl);

document.querySelectorAll('.cap-card, .step, .stack-list div, .audience-card, .faq-list details').forEach(el => {
  el.classList.add('observe-in');
});

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('in-view');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.08 });

document.querySelectorAll('.observe-in').forEach(el => observer.observe(el));

const backdrop = document.getElementById('modalBackdrop');
const title = document.getElementById('modalTitle');
const kicker = document.getElementById('modalKicker');
const description = document.getElementById('modalDescription');
const form = document.getElementById('profileForm');
const status = document.getElementById('formStatus');
const tabs = document.querySelectorAll('[data-modal-switch]');

function renderForm(mode) {
  const register = mode === 'register';
  tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.modalSwitch === mode));
  kicker.textContent = register ? 'NEW TO GEOSTATIX?' : 'ALREADY A USER?';
  title.textContent = register ? 'Tell us about yourself' : 'Sign in to GeoStatix';
  description.textContent = register
    ? 'Create a lightweight profile so the future GeoStatix experience can be tailored to you.'
    : 'Account sign-in is planned for the next release. You can still explore the live platform without an account.';
  form.innerHTML = register
    ? '<label>Name<input name="name" type="text" placeholder="Your name" required></label><label>Email<input name="email" type="email" placeholder="you@example.com" required></label><label>I'm a<select name="role"><option>Student</option><option>Researcher</option><option>Geologist</option><option>Mining / Exploration Professional</option><option>Petroleum Professional</option><option>Data / Analytics</option><option>Other</option></select></label><label>Areas of interest<select name="interest"><option>Geostatistics</option><option>Exploration Geology</option><option>Mining</option><option>Petroleum Geoscience</option><option>Environmental Geoscience</option><option>Statistics & Data Analytics</option><option>Academic Research</option></select></label><label>Tell us about yourself <span class="optional">Optional</span><textarea name="about" rows="3" placeholder="What are you hoping to explore with GeoStatix?"></textarea></label><button class="button" type="submit">Continue <span>→</span></button>'
    : '<label>Email<input name="email" type="email" placeholder="you@example.com" required></label><label>Password<input name="password" type="password" placeholder="Password" required></label><button class="button" type="submit">Sign in <span>→</span></button>';
  status.textContent = '';
  setTimeout(() => backdrop.querySelector('input')?.focus(), 50);
}

function openModal(mode) {
  renderForm(mode);
  backdrop.classList.add('open');
  backdrop.setAttribute('aria-hidden', 'false');
}

function closeModal() {
  backdrop.classList.remove('open');
  backdrop.setAttribute('aria-hidden', 'true');
}

document.querySelectorAll('[data-modal]').forEach(button => {
  button.addEventListener('click', () => openModal(button.dataset.modal));
});

tabs.forEach(tab => tab.addEventListener('click', () => renderForm(tab.dataset.modalSwitch)));
document.querySelector('.modal-close').addEventListener('click', closeModal);
backdrop.addEventListener('click', e => { if (e.target === backdrop) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

form.addEventListener('submit', e => {
  e.preventDefault();
  const register = form.querySelector('[name="name"]');
  status.textContent = register
    ? 'Thanks — the profile flow is ready. Secure account storage will be connected in a future release.'
    : 'Sign-in is not connected yet. You can launch GeoStatix without an account.';
});
