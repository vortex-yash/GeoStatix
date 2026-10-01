const CONFIG = {
  appUrl: "https://geostatix.streamlit.app",
  githubUrl: "https://github.com/vortex-yash/GeoStatix"
};

// ============================================================
// SUPABASE
// ============================================================

const SUPABASE_URL = "https://shkdzzrseujkzjrcxeyk.supabase.co";
const SUPABASE_PUBLISHABLE_KEY = "sb_publishable_LgBk4WD11WfYEUoc9NsEuQ_G-_x-oi0";

const supabaseClient = window.supabase.createClient(
  SUPABASE_URL,
  SUPABASE_PUBLISHABLE_KEY
);


// ============================================================
// EXISTING WEBSITE CONFIGURATION
// ============================================================

document.querySelectorAll("[data-app-link]").forEach(link => {
  link.href = CONFIG.appUrl;
});

document.querySelectorAll(
  ".cap-card, .step, .stack-list div, .audience-card, .faq-list details"
).forEach(el => {
  el.classList.add("observe-in");
});


// ============================================================
// REVEAL-ON-SCROLL
// ============================================================

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.08 }
  );

  document.querySelectorAll(".observe-in").forEach(el => {
    observer.observe(el);
  });
} else {
  document.querySelectorAll(".observe-in").forEach(el => {
    el.classList.add("in-view");
  });
}


// ============================================================
// AUTH MODAL
// ============================================================

const backdrop = document.getElementById("modalBackdrop");
const title = document.getElementById("modalTitle");
const kicker = document.getElementById("modalKicker");
const description = document.getElementById("modalDescription");
const form = document.getElementById("profileForm");
const status = document.getElementById("formStatus");
const tabs = document.querySelectorAll("[data-modal-switch]");

let currentMode = "signin";
let pendingEmail = "";
let pendingProfile = {};

// ============================================================
// AUTH SESSION
// ============================================================

let currentUser = null;

async function loadAuthSession() {
  try {
    const { data, error } = await supabaseClient.auth.getSession();

    if (error) {
      console.error("Session error:", error);
      return;
    }

    currentUser = data.session?.user || null;

    updateAuthUI();
  } catch (error) {
    console.error("Unable to load auth session:", error);
  }
}

function updateAuthUI() {
  const loginButton = document.querySelector(".nav-login");
  const registerButton = document.querySelector(".nav-register");

  if (!loginButton || !registerButton) {
    return;
  }

  if (currentUser) {
    const name =
      currentUser.user_metadata?.full_name ||
      currentUser.email?.split("@")[0] ||
      "there";

    loginButton.textContent = `Hi, ${name}`;
    registerButton.textContent = "Sign out";
  } else {
    loginButton.textContent = "Sign in";
    registerButton.textContent = "Create profile";
  }
}

supabaseClient.auth.onAuthStateChange((event, session) => {
  currentUser = session?.user || null;
  updateAuthUI();
});



// ============================================================
// RENDER AUTH FORM
// ============================================================

function renderForm(mode) {
  currentMode = mode;

  const register = mode === "register";

  tabs.forEach(tab => {
    tab.style.display = "";
    tab.classList.toggle(
      "active",
      tab.dataset.modalSwitch === mode
    );
  });

  kicker.textContent = register
    ? "NEW TO GEOSTATIX?"
    : "ALREADY A USER?";

  title.textContent = register
    ? "Create your GeoStatix profile"
    : "Sign in to GeoStatix";

  description.textContent = register
    ? "Create your profile using your email. We'll send you a one-time verification code."
    : "Enter your email and we'll send you a one-time verification code.";

  form.innerHTML = register
    ? `
      <label>
        Name
        <input
          name="name"
          type="text"
          placeholder="Your name"
          required
        >
      </label>

      <label>
        Email
        <input
          name="email"
          type="email"
          placeholder="you@example.com"
          required
        >
      </label>

      <label>
        I'm a
        <select name="role">
          <option>Student</option>
          <option>Researcher</option>
          <option>Geologist</option>
          <option>Mining / Exploration Professional</option>
          <option>Petroleum Professional</option>
          <option>Data / Analytics</option>
          <option>Other</option>
        </select>
      </label>

      <label>
        Areas of interest
        <select name="interest">
          <option>Geostatistics</option>
          <option>Exploration Geology</option>
          <option>Mining</option>
          <option>Petroleum Geoscience</option>
          <option>Environmental Geoscience</option>
          <option>Statistics & Data Analytics</option>
          <option>Academic Research</option>
        </select>
      </label>

      <label>
        Tell us about yourself
        <span class="optional">Optional</span>
        <textarea
          name="about"
          rows="3"
          placeholder="What are you hoping to explore with GeoStatix?"
        ></textarea>
      </label>

      <button class="button" type="submit">
        Send verification code <span>→</span>
      </button>
    `
    : `
      <label>
        Email
        <input
          name="email"
          type="email"
          placeholder="you@example.com"
          required
        >
      </label>

      <button class="button" type="submit">
        Send verification code <span>→</span>
      </button>
    `;

  status.textContent = "";

  setTimeout(() => {
    backdrop.querySelector("input")?.focus();
  }, 50);
}

// ============================================================
// PROFILE VIEW
// ============================================================

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderProfileView(message = "") {
  const metadata = currentUser?.user_metadata || {};

  const name =
    metadata.full_name ||
    currentUser?.email?.split("@")[0] ||
    "User";

  const email = currentUser?.email || "—";
  const role = metadata.role || "—";
  const interest = metadata.interest || "—";
  const about = metadata.about || "No description added.";

  // Hide Sign in / Create profile tabs in profile view
  tabs.forEach(tab => {
    tab.style.display = "none";
  });

  kicker.textContent = "YOUR GEOSTATIX PROFILE";

  title.textContent = `Welcome, ${name}`;

  description.textContent =
    "Your GeoStatix account information.";
  if (message) {
  form.innerHTML = `
    <div
      style="
        padding:12px 14px;
        margin-bottom:16px;
        border:1px solid rgba(180,255,120,.25);
        border-radius:12px;
        background:rgba(180,255,120,.06);
        font-size:.9rem;
      "
    >
      ✓ ${escapeHtml(message)}
    </div>
  `;
}
form.innerHTML = `
  ${
    message
      ? `
        <div
          style="
            padding:12px 14px;
            margin-bottom:16px;
            border:1px solid rgba(180,255,120,.25);
            border-radius:12px;
            background:rgba(180,255,120,.06);
            font-size:.9rem;
          "
        >
          ✓ ${escapeHtml(message)}
        </div>
      `
      : ""
  }

  <!-- Profile identity -->
  <div
    style="
      display:flex;
      align-items:center;
      gap:14px;
      padding:18px;
      margin-bottom:14px;
      border:1px solid rgba(255,255,255,.08);
      border-radius:16px;
      background:rgba(255,255,255,.03);
    "
  >
    <div
      style="
        width:52px;
        height:52px;
        min-width:52px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:1.15rem;
        font-weight:700;
        background:rgba(180,255,120,.12);
        border:1px solid rgba(180,255,120,.25);
      "
    >
      ${escapeHtml(name.charAt(0).toUpperCase())}
    </div>

    <div>
      <div style="font-size:.75rem; opacity:.55; margin-bottom:4px;">
        GEOSTATIX MEMBER
      </div>

      <div style="font-size:1.05rem; font-weight:650;">
        ${escapeHtml(name)}
      </div>
    </div>
  </div>

  <!-- Name -->
  <div
    class="glass-strip"
    style="
      width:100%;
      box-sizing:border-box;
      padding:18px;
      margin-bottom:14px;
    "
  >
    <div style="font-size:.78rem; opacity:.55; margin-bottom:6px;">
      NAME
    </div>

    <div style="font-weight:600;">
      ${escapeHtml(name)}
    </div>
  </div>

  <!-- Email -->
  <div
    class="glass-strip"
    style="
      width:100%;
      box-sizing:border-box;
      padding:18px;
      margin-bottom:14px;
    "
  >
    <div style="font-size:.78rem; opacity:.55; margin-bottom:6px;">
      EMAIL
    </div>

    <div
      style="
        font-weight:600;
        word-break:break-word;
      "
    >
      ${escapeHtml(email)}
    </div>

    <div style="margin-top:8px; font-size:.8rem; opacity:.65;">
      ✓ Email verified
    </div>
  </div>

  <!-- Role + Interest -->
  <div
    style="
      display:grid;
      grid-template-columns:minmax(0,1fr) minmax(0,1fr);
      gap:12px;
      width:100%;
      box-sizing:border-box;
      margin-bottom:14px;
    "
  >

    <div
      class="glass-strip"
      style="
        min-width:0;
        padding:16px;
        box-sizing:border-box;
      "
    >
      <div style="font-size:.75rem; opacity:.55; margin-bottom:6px;">
        ROLE
      </div>

      <div
        style="
          font-weight:600;
          word-break:break-word;
        "
      >
        ${escapeHtml(role)}
      </div>
    </div>

    <div
      class="glass-strip"
      style="
        min-width:0;
        padding:16px;
        box-sizing:border-box;
      "
    >
      <div style="font-size:.75rem; opacity:.55; margin-bottom:6px;">
        INTEREST
      </div>

      <div
        style="
          font-weight:600;
          word-break:break-word;
        "
      >
        ${escapeHtml(interest)}
      </div>
    </div>

  </div>

  <!-- About -->
  <div
    class="glass-strip"
    style="
      width:100%;
      box-sizing:border-box;
      padding:18px;
      margin-bottom:16px;
    "
  >
    <div style="font-size:.75rem; opacity:.55; margin-bottom:7px;">
      ABOUT
    </div>

    <div
      style="
        line-height:1.5;
        opacity:.85;
        word-break:break-word;
      "
    >
      ${escapeHtml(about)}
    </div>
  </div>

  <!-- Actions -->
  <button
    type="button"
    class="button"
    id="editProfile"
    style="width:100%;"
  >
    Edit profile
  </button>

  <button
    type="button"
    class="button button-ghost glass-button"
    id="profileClose"
    style="
      width:100%;
      margin-top:10px;
    "
  >
    Close
  </button>
`;

document
  .getElementById("editProfile")
  .addEventListener("click", renderEditProfile);

document
  .getElementById("profileClose")
  .addEventListener("click", closeModal);

}

// ============================================================
// EDIT PROFILE
// ============================================================

function renderEditProfile() {
  const metadata = currentUser?.user_metadata || {};

  const name = metadata.full_name || "";
  const role = metadata.role || "Student";
  const interest = metadata.interest || "Geostatistics";
  const about = metadata.about || "";

  tabs.forEach(tab => {
    tab.style.display = "none";
  });

  kicker.textContent = "EDIT YOUR PROFILE";

  title.textContent = "Update your GeoStatix profile";

  description.textContent =
    "Changes will be saved to your account.";

  form.innerHTML = `
    <label>
      Name
      <input
        id="editName"
        type="text"
        value="${escapeHtml(name)}"
        required
      >
    </label>

    <label>
      I'm a
      <select id="editRole">
        <option ${role === "Student" ? "selected" : ""}>Student</option>
        <option ${role === "Researcher" ? "selected" : ""}>Researcher</option>
        <option ${role === "Geologist" ? "selected" : ""}>Geologist</option>
        <option ${role === "Mining / Exploration Professional" ? "selected" : ""}>Mining / Exploration Professional</option>
        <option ${role === "Petroleum Professional" ? "selected" : ""}>Petroleum Professional</option>
        <option ${role === "Data / Analytics" ? "selected" : ""}>Data / Analytics</option>
        <option ${role === "Other" ? "selected" : ""}>Other</option>
      </select>
    </label>

    <label>
      Areas of interest
      <select id="editInterest">
        <option ${interest === "Geostatistics" ? "selected" : ""}>Geostatistics</option>
        <option ${interest === "Exploration Geology" ? "selected" : ""}>Exploration Geology</option>
        <option ${interest === "Mining" ? "selected" : ""}>Mining</option>
        <option ${interest === "Petroleum Geoscience" ? "selected" : ""}>Petroleum Geoscience</option>
        <option ${interest === "Environmental Geoscience" ? "selected" : ""}>Environmental Geoscience</option>
        <option ${interest === "Statistics & Data Analytics" ? "selected" : ""}>Statistics & Data Analytics</option>
        <option ${interest === "Academic Research" ? "selected" : ""}>Academic Research</option>
      </select>
    </label>

    <label>
      Tell us about yourself
      <span class="optional">Optional</span>

      <textarea
        id="editAbout"
        rows="4"
        placeholder="What are you hoping to explore with GeoStatix?"
      >${escapeHtml(about)}</textarea>
    </label>

    <button
      type="button"
      class="button"
      id="saveProfile"
      style="width:100%;"
    >
      Save changes
    </button>

    <button
      type="button"
      class="button button-ghost glass-button"
      id="cancelEdit"
      style="width:100%; margin-top:10px;"
    >
      Cancel
    </button>

    <p
      class="form-status"
      id="profileEditStatus"
      style="margin-top:10px;"
    ></p>
  `;

  document
    .getElementById("saveProfile")
    .addEventListener("click", saveProfile);

  document
    .getElementById("cancelEdit")
    .addEventListener("click", renderProfileView);
}

// ============================================================
// SAVE PROFILE
// ============================================================

async function saveProfile() {
  if (!currentUser) {
    return;
  }

  const name = document
    .getElementById("editName")
    .value
    .trim();

  const role = document
    .getElementById("editRole")
    .value;

  const interest = document
    .getElementById("editInterest")
    .value;

  const about = document
    .getElementById("editAbout")
    .value
    .trim();

  const saveButton = document.getElementById("saveProfile");
  const editStatus = document.getElementById("profileEditStatus");

  if (!name) {
    editStatus.textContent = "Please enter your name.";
    return;
  }

  saveButton.disabled = true;
  saveButton.textContent = "Saving…";
  editStatus.textContent = "";

  try {
    const { data, error } =
      await supabaseClient.auth.updateUser({
        data: {
          full_name: name,
          role,
          interest,
          about
        }
      });

    if (error) {
      throw error;
    }

    currentUser = data.user;

    updateAuthUI();
    
    renderProfileView("Profile updated successfully.");
  } catch (error) {
    console.error("Profile update error:", error);

    editStatus.textContent =
      error.message ||
      "Unable to save your profile. Please try again.";

    saveButton.disabled = false;
    saveButton.textContent = "Save changes";
  }
}

// ============================================================
// OPEN / CLOSE MODAL
// ============================================================

function openModal(mode) {

  if (mode === "profile") {
    renderProfileView();
  } else {
    renderForm(mode);
  }

  backdrop.classList.add("open");
  backdrop.setAttribute("aria-hidden", "false");
}

function closeModal() {
  backdrop.classList.remove("open");
  backdrop.setAttribute("aria-hidden", "true");
}

document.querySelectorAll("[data-modal]").forEach(button => {
  button.addEventListener("click", async () => {

    // Logged-in user
    if (currentUser) {

      // Sign out button
      if (button.classList.contains("nav-register")) {
        const { error } = await supabaseClient.auth.signOut();

        if (error) {
          console.error("Sign out error:", error);
          return;
        }

        currentUser = null;
        updateAuthUI();
        return;
      }

      // User/profile button
      if (button.classList.contains("nav-login")) {
        openModal("profile");
        return;
      }
    }

    // Logged-out user
    openModal(button.dataset.modal);
  });
});

tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    renderForm(tab.dataset.modalSwitch);
  });
});

document
  .querySelector(".modal-close")
  .addEventListener("click", closeModal);

backdrop.addEventListener("click", event => {
  if (event.target === backdrop) {
    closeModal();
  }
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeModal();
  }
});


// ============================================================
// SEND OTP
// ============================================================

async function sendOtp(event) {
  event.preventDefault();

  const formData = new FormData(form);

  const email = String(formData.get("email") || "")
    .trim()
    .toLowerCase();

  if (!email) {
    status.textContent = "Please enter your email address.";
    return;
  }

  pendingEmail = email;

  if (currentMode === "register") {
    pendingProfile = {
      name: String(formData.get("name") || "").trim(),
      role: String(formData.get("role") || ""),
      interest: String(formData.get("interest") || ""),
      about: String(formData.get("about") || "").trim()
    };
  } else {
    pendingProfile = {};
  }

  const button = form.querySelector("button");

  if (button) {
    button.disabled = true;
    button.innerHTML = "Sending code…";
  }

  status.textContent = "";

  try {
    const { error } = await supabaseClient.auth.signInWithOtp({
      email,
      options: {
        shouldCreateUser: currentMode === "register",

        data:
          currentMode === "register"
            ? {
                full_name: pendingProfile.name,
                role: pendingProfile.role,
                interest: pendingProfile.interest,
                about: pendingProfile.about
              }
            : undefined
      }
    });

    if (error) {
      throw error;
    }

    renderOtpForm();

  } catch (error) {
    console.error("OTP error:", error);

    status.textContent =
      error.message ||
      "Unable to send the verification code. Please try again.";

    if (button) {
      button.disabled = false;
      button.innerHTML =
        'Send verification code <span>→</span>';
    }
  }
}


// ============================================================
// OTP FORM
// ============================================================

function renderOtpForm() {
  kicker.textContent = "CHECK YOUR EMAIL";

  title.textContent = "Enter your verification code";

  description.textContent =
    `We sent a 6-digit verification code to ${pendingEmail}.`;

  form.innerHTML = `
    <label>
      Verification code
      <input
        name="otp"
        type="text"
        inputmode="numeric"
        autocomplete="one-time-code"
        maxlength="6"
        pattern="[0-9]{6}"
        placeholder="123456"
        required
      >
    </label>

    <button class="button" type="submit">
      Verify and continue <span>→</span>
    </button>

    <button
      type="button"
      class="button button-ghost glass-button"
      id="resendOtp"
    >
      Resend code
    </button>
  `;

  status.textContent = "";

  setTimeout(() => {
    backdrop.querySelector('[name="otp"]')?.focus();
  }, 50);

  document
    .getElementById("resendOtp")
    .addEventListener("click", () => {
      renderForm(currentMode);

      const emailInput = form.querySelector('[name="email"]');

      if (emailInput) {
        emailInput.value = pendingEmail;
      }
    });
}


// ============================================================
// VERIFY OTP
// ============================================================

async function verifyOtp(event) {
  event.preventDefault();

  const otpInput = form.querySelector('[name="otp"]');
  const token = otpInput?.value.trim();

  if (!token || token.length !== 6) {
    status.textContent =
      "Please enter the 6-digit verification code.";
    return;
  }

  const button = form.querySelector(
    'button[type="submit"]'
  );

  if (button) {
    button.disabled = true;
    button.innerHTML = "Verifying…";
  }

  try {
    const { data, error } = await supabaseClient.auth.verifyOtp({
      email: pendingEmail,
      token,
      type: "email"
    });

    if (error) {
      throw error;
    }

    if (!data.session) {
      throw new Error(
        "Verification succeeded, but no session was created."
      );
    }

    status.textContent = "";

    const displayName =
      pendingProfile.name ||
      data.user?.user_metadata?.full_name ||
      "there";

    kicker.textContent = "EMAIL VERIFIED";

    title.textContent =
      currentMode === "register"
        ? `Welcome to GeoStatix, ${displayName}`
        : "Welcome back to GeoStatix";

    description.textContent =
      currentMode === "register"
        ? "Your profile is ready. You can now launch the GeoStatix analytics platform."
        : "You're signed in successfully. Continue to the GeoStatix analytics platform.";

    form.innerHTML = `
      <div class="glass-strip" style="padding: 16px 18px; margin-bottom: 16px;">
        <div style="display:flex; align-items:center; gap:10px; font-weight:600;">
          <span style="display:grid; place-items:center; width:28px; height:28px; border-radius:50%; border:1px solid rgba(170,255,110,.35); background:rgba(170,255,110,.10);">✓</span>
          Your email has been verified
        </div>

        <div style="margin-top:8px; opacity:.72; font-size:.9rem;">
          ${pendingEmail}
        </div>
      </div>

      <a
        class="button"
        id="continueToGeoStatix"
        href="${CONFIG.appUrl}"
        target="_blank"
        rel="noreferrer"
        style="display:flex; justify-content:center; text-decoration:none;"
      >
        Launch GeoStatix <span>↗</span>
      </a>

      <button
        type="button"
        class="button button-ghost glass-button"
        id="closeWelcome"
        style="width:100%; margin-top:10px;"
      >
        Stay on website
      </button>
    `;

    document
      .getElementById("closeWelcome")
      .addEventListener("click", closeModal);

  } catch (error) {
    console.error("Verification error:", error);

    status.textContent =
      error.message ||
      "Invalid or expired verification code.";

    if (button) {
      button.disabled = false;
      button.innerHTML =
        'Verify and continue <span>→</span>';
    }
  }
}

// ============================================================
// FORM SUBMISSION ROUTER
// ============================================================

form.addEventListener("submit", event => {
  if (form.querySelector('[name="otp"]')) {
    verifyOtp(event);
  } else {
    sendOtp(event);
  }
});
loadAuthSession();