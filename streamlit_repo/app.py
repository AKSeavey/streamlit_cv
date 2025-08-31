# app.py — Home (single scroll) + Portfolio subpages (no map)
# ----------------------------------------------------------------
# Changes in this refactor:
#  • Removed interactive map section + sidebar button
#  • Dropped pydeck/pandas + map helpers, LOCATIONS/CATEGORY_COLORS data
#  • Kept SHOW_MAP scaffold (off) so you can re-enable in the future
# ----------------------------------------------------------------

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from subpages.pymoo_discgolf import pymoo_discgolf_page

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit.components.v1 import html as st_html
from html import escape

# -----------------------------
# Page config + CSS
# -----------------------------
st.set_page_config(
    page_title="Andrew Knight Seavey — Interactive CV",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* --- Portfolio grid: scoped only to the six-card grid --- */
.portfolio-grid .portfolio-card {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  display: flex;
  flex-direction: column;
  height: 360px;                 /* total card height */
  background: rgba(255,255,255,0.04);
}

/* cover image with fixed height + crop */
.portfolio-grid .portfolio-cover img {
  width: 100% !important;
  height: 200px !important;      /* fixed image height */
  object-fit: cover;
  border-radius: 8px;
  display: block;
}


/* Hero heading + subheading on the home intro */
.hero {
  font-weight: 800;
  line-height: 1.1;
  margin: 0 0 6px;
  letter-spacing: .2px;
  font-size: clamp(28px, 3.6vw, 48px);
}

.hero-sub {
  font-size: clamp(15px, 1.2vw, 18px);
  line-height: 1.6;
  opacity: .9;
  max-width: 62ch;
}

/* Optional: tighten on small screens */
@media (max-width: 700px) {
  .hero { font-size: 26px; }
  .hero-sub { font-size: 16px; }
}
                        
/* Hard reset so the blurb ignores any clamp/ellipsis from ancestors */
.blurb {
  all: unset;                         /* wipe inherited text rules */
  display: block !important;
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  -webkit-line-clamp: initial !important;
  -webkit-box-orient: initial !important;
  max-height: none !important;
  height: auto !important;

  /* re-apply readable body text */
  font-family: inherit !important;
  font-size: 0.98rem !important;
  line-height: 1.55 !important;
  color: inherit !important;
  opacity: .95;
}

/* Make sure children can wrap too */
.blurb * {
  all: revert;                        /* allow normal HTML defaults */
  overflow: visible !important;
  white-space: normal !important;
  text-overflow: clip !important;
}

/* title block gets a fixed minimum height to normalize card layout */
.portfolio-grid .portfolio-title {
  margin: 10px 2px 0;
  font-weight: 700;
  font-size: 1.05rem;
  line-height: 1.2;
  min-height: 2.6em;             /* ~2 lines -> keeps button aligned */
  display: -webkit-box;
  -webkit-line-clamp: 2;         /* clamp long titles to 2 lines */
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* optional: center the button area */
.portfolio-grid .portfolio-button {
  margin-top: auto;              /* push to bottom of the card */
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paths & Data  (only change made: lock assets to app.py folder)
# -----------------------------
BASE = Path(__file__).resolve().parent
ASSETS = BASE / "assets"

# --- Feature flags ---
SHOW_GALLERY = False           # sitewide: remove galleries everywhere
DEFAULT_SHOW_DETAILS = False   # per-item can override via item['show_details']=False
SHOW_MAP = False               # scaffold retained; map code removed in this refactor

DS_SKILLS: Dict[str, int] = {
    "Computational Design": 80, "Python": 75, "C#": 30, "ML": 60,
    "Visualization": 90, "Automation": 60, "Geospatial": 90, "NLP": 30,
}
AEC_SKILLS: Dict[str, int] = {
    "Revit": 75, "AutoCAD": 90, "Adobe Suite": 85, "Construction \n Management": 75,
    "GIS": 90, "SITES": 75, "3D Rendering": 80, "Point Cloud": 75,
}
LANG_LEVELS: Dict[str, int] = {"English": 98, "Spanish": 85, "Norwegian": 20, "Japanese": 60, "Mandarin": 75, "K'ichee' Maya": 20}

# ---- Portfolio (slim, single-artifact per case) ----
PORTFOLIO: List[Dict] = [
    {
        "slug": "generative-design",
        "title": "Generative Design (Revit Dynamo)",
        "blurb": "Diagram of a recent Revit Dynamo generative design project I worked on. Due to controlled unclassified information (CUI), I'm not able to show actual graphics. The big idea was optimizing a warehouse to fit a specific list of equipment for a client. The end result gave them confidence to plan for additional equipment while optimizing adjacencies and maximizing the usability of the space.",
        "thumbnail": ASSETS/"portfolio"/"gd_pruned_thumb.png",
        "kind": "diagram",
        "diagram_image": ASSETS/"portfolio"/"gd.jpg",
        "links": [],
    },
    {
        "slug": "keystone-species",
        "title": "Keystone Species · ML Capstone",
        "blurb": "Capstone project for my 2023 data science bootcamp. I collaborated with two other classmates and built a tree-based ML model predicting urban keystone species distribution based on land cover, soil type and iNaturalist data. This is the seed for a larger project that infers the appropriate placement of native plant species in urban environments to maximize ecosystem services and biodiversity.",
        "thumbnail": ASSETS/"portfolio"/"keystone_img_1.jpg",
        "kind": "pdf",
        "pdf": ASSETS/"portfolio"/"keystone_capstone.pdf",
    },
    {
        "slug": "AEC-work-samples",
        "title": "AEC Work Samples (Selected)",
        "blurb": "A collection of my work from my years of landscape architecture practice. Spanning between technical and artistic, that's what I love about AEC.",
        "thumbnail": ASSETS/"portfolio"/"aec_img.png",
        "kind": "pdf",
        "pdf": ASSETS/"portfolio"/"aec_selected.pdf",
    },
    {
        "slug": "mla-thesis",
        "title": "MLA Thesis: Biofuel in Liminal Spaces",
        "blurb": "Imagine if we cultivated our underutilized urban spaces to benefits all species (including us). My thesis explored the potential of cultivating biofuel crops in neglected urban areas to produce renewable energy while enhancing urban ecology and community well-being.",
        "thumbnail": ASSETS/"portfolio"/"mla_thesis_img.jpg",
        "kind": "pdf",
        "pdf": ASSETS/"portfolio"/"mla_thesis.pdf",
    },
    {
        "slug": "pymoo-sample",
        "title": "Genetic Algorithm · Disc Golf Course Optimization",
        "blurb": "Lightweight multi-objective GA exploring disc golf course layouts. This app is a playful demonstration of how a portable genetic algorithm can be embedded in a shareable web app. It’s meant as a conversation starter to spark ideas about multi-objective optimization, design trade-offs, and interactive visualization. This MVP is not a fully vetted research tool or professional grade design engine, but hopefully points to what is possible . Parameters are simplified, assumptions are flat-ground, and constraints are intentionally loose so you can explore and learn quickly.",
        "thumbnail": ASSETS/"portfolio"/"discgolf_img.png",
        "kind": "demo",
        "demo": "pymoo-discgolf",
    },
    {
        "slug": "micro-farm",
        "title": "Growing Blue Corn (life away from the screen)",
        "blurb":"A photo story of cultivating blue corn in New Mexico, exploring traditional agricultural practices and the connection between land, culture, and food.",
        "thumbnail": ASSETS/"portfolio"/"corn_thumb.png",
        "kind": "story",
        "story_image": ASSETS/"portfolio"/"seavey - blue corn.jpg",
        "links": [],
    },
]

# -----------------------------
# Helpers (routing, rerun, scroll)
# -----------------------------
def rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def get_view() -> str:
    v = st.session_state.get("view", "home")
    return v if v in {"home", "case"} else "home"

def set_view(v: str):
    st.session_state["view"] = v

def get_case_slug() -> Optional[str]:
    return st.session_state.get("case")

def set_case(slug: Optional[str]):
    if slug is None:
        st.session_state.pop("case", None)
    else:
        st.session_state["case"] = slug

def set_pending_jump(anchor_id: str):
    st.session_state["pending_jump"] = anchor_id

def consume_pending_jump() -> Optional[str]:
    return st.session_state.pop("pending_jump", None)

SECTION_IDS = {"intro": "sec-intro", "work": "sec-work", "dlc": "sec-dlc"}
# Map anchor is deliberately not registered in this refactor; SHOW_MAP scaffold kept for future.

def js_scroll_to_anchor(anchor_id: str):
    st_html(
        f"""
<script>
(function(){{
  const root = window.parent.document;
  const targetId = "{anchor_id}";
  function scrollNow() {{
    const el = root.getElementById(targetId);
    if (el) {{
      try {{
        el.scrollIntoView({{behavior:'smooth', block:'start'}});
      }} catch(e) {{
        const scroller = root.querySelector('section.main div.block-container');
        if (scroller) scroller.scrollTo({{top: el.getBoundingClientRect().top + scroller.scrollTop - 80, behavior:'smooth'}});
      }}
      return true;
    }}
    return false;
  }}
  if (!scrollNow()) {{
    const obs = new MutationObserver(() => {{ if (scrollNow()) obs.disconnect(); }});
    obs.observe(root, {{childList:true, subtree:true}});
    setTimeout(() => {{ scrollNow(); }}, 400);
  }}
}})();
</script>
""",
        height=0,
    )

def disable_scroll_restoration():
    st_html("""<script>try { window.parent.history.scrollRestoration = 'manual'; } catch(e) {}</script>""", height=0)

def force_case_top_strong():
    st_html(
        """
<script>
(function(){
  const win = window.parent;
  const doc = win.document;
  function findScroller(){
    const sels = [
      'section[data-testid="stAppViewContainer"] .main .block-container',
      'section.main .block-container',
      'div.block-container',
      'main .block-container',
      'section[data-testid="stAppViewContainer"]',
      'section.main'
    ];
    for (const s of sels){
      const el = doc.querySelector(s);
      if (el) return el;
    }
    return doc.scrollingElement || doc.documentElement || doc.body;
  }
  const scroller = findScroller();
  function snap(){
    try { win.scrollTo(0, 0); } catch(e){}
    try { (doc.scrollingElement || doc.documentElement || doc.body).scrollTop = 0; } catch(e){}
    try { if (scroller && typeof scroller.scrollTop === 'number') scroller.scrollTop = 0; } catch(e){}
  }
  let rafCount = 0;
  (function rafBurst(){ if (rafCount++ < 20) { snap(); win.requestAnimationFrame(rafBurst); } })();
  const iv = setInterval(snap, 100);
  setTimeout(() => clearInterval(iv), 2500);
  try {
    const obs = new MutationObserver(() => snap());
    obs.observe(scroller || doc.body, {childList:true, subtree:true});
    setTimeout(() => obs.disconnect(), 4000);
  } catch(e){}
})();
</script>
""",
        height=0,
    )

def enforce_case_top():
    st_html(
        """
<script>
(function(){
  const rootWin = window.parent;
  function toTop(){
    try { rootWin.scrollTo({top:0, left:0, behavior:'auto'}); } catch(e){}
    try {
      const scroller = rootWin.document.querySelector('section.main div.block-container');
      if (scroller) scroller.scrollTo({top:0, left:0, behavior:'auto'});
    } catch(e){}
  }
  toTop();
  [60, 200, 500, 900].forEach(t => setTimeout(toTop, t));
})();
</script>
""",
        height=0,
    )

# -----------------------------
# Viz helpers (radar)
# -----------------------------
def is_dark_theme() -> bool:
    try:
        base = (st.get_option("theme.base") or "light").lower()
    except Exception:
        base = "light"
    return base == "dark"

def radar_chart(title: str, scores: Dict[str, float], size_px: int = 520, dark: bool | None = None, title_y: float = 1.25):
    if dark is None: dark = is_dark_theme()
    labels = list(scores.keys()); values = list(scores.values())
    angles = np.linspace(0, 2*math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]; v = values + values[:1]
    dpi = 200
    fig = plt.figure(figsize=(size_px/dpi, size_px/dpi), dpi=dpi); ax = plt.subplot(111, polar=True)
    fig.patch.set_alpha(0.0); ax.set_facecolor("none"); ax.set_theta_offset(math.pi/2); ax.set_theta_direction(-1)
    label_color = "#ffffff"; tick_color = "#e5e7eb"; grid_color = "#9ca3af"
    plt.xticks(angles[:-1], labels, fontsize=10, color=label_color)
    ax.tick_params(pad=8, colors=tick_color); ax.set_ylim(0, 100); ax.set_yticks([20,40,60,80,100])
    ax.set_yticklabels([20,40,60,80,100], fontsize=8, color=tick_color)
    ax.yaxis.grid(True, linestyle="dotted", alpha=0.25, color=grid_color)
    ax.xaxis.grid(True, linestyle="dotted", alpha=0.25, color=grid_color)
    line_color = (0.3,0.9,1.0,0.95) if dark else (0.1,0.4,0.7,0.95)
    for lw, a in [(10,0.06),(8,0.08),(6,0.10),(4,0.12)]: ax.plot(angles, v, linewidth=lw, color=(line_color[0], line_color[1], line_color[2], a))
    ax.plot(angles, v, linewidth=2, color=line_color); ax.fill(angles, v, alpha=0.10, color=line_color)
    ax.set_title(title, y=title_y, fontsize=12, color=line_color)
    st.pyplot(fig, use_container_width=False, transparent=True, bbox_inches="tight")

# -----------------------------
# Sidebar
# -----------------------------
view = get_view()

if view == "home":
    st.sidebar.title("Navigate")
    if st.sidebar.button("Intro", use_container_width=True):
        set_pending_jump(SECTION_IDS["intro"]); rerun()
    if st.sidebar.button("Work Samples", use_container_width=True):
        set_pending_jump(SECTION_IDS["work"]); rerun()
    # Map button removed (SHOW_MAP scaffold kept, but section is gone)
    if st.sidebar.button("Download / Contact", use_container_width=True):
        set_pending_jump(SECTION_IDS["dlc"]); rerun()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Portfolio")
    for p in PORTFOLIO:
        if st.sidebar.button(f"• {p['title']}", key=f"nav_{p['slug']}"):
            set_view("case"); set_case(p["slug"]); rerun()
else:
    st.sidebar.title("Portfolio")
    if st.sidebar.button("← Back to Home", use_container_width=True):
        set_view("home"); set_case(None); set_pending_jump(SECTION_IDS["work"]); rerun()
    st.sidebar.markdown("---")
    for p in PORTFOLIO:
        if st.sidebar.button(p["title"], key=f"goto_{p['slug']}"):
            set_view("case"); set_case(p["slug"]); rerun()

# -----------------------------
# HOME (single page)
# -----------------------------
def render_home():
    disable_scroll_restoration()
    jump_id = consume_pending_jump()
    if jump_id: js_scroll_to_anchor(jump_id)

    # Anchors
    st.markdown(f"<div id='{SECTION_IDS['intro']}' class='section'></div>", unsafe_allow_html=True)

    # Intro (3 cols)
    col_left, col_mid, col_right = st.columns([1.6, 1.0, 1.1], vertical_alignment="center")
    with col_left:
        st.markdown('<div class="hero">Data Scientist · SITES AP Landscape Architect · Computational Design Explorer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">'
            'I build bridges between the design world and emerging technology. '
            'With over a decade in AEC and hands-on depth in Python, SQL, and machine learning, '
            'I integrate geospatial intelligence, predictive analytics, and computational/BIM workflows '
            'to generate insights into resilience, efficiency, and ecological performance.'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_mid:
        portrait_path = ASSETS / "ds_portrait.jpg"
        st.markdown('<div class="portrait">', unsafe_allow_html=True)
        if portrait_path.exists(): st.image(portrait_path, width=240)
        else: st.error(f"image not found at {portrait_path}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.subheader("Introduksjon på Norsk")
        YT_URL = "https://youtu.be/nBzj2h8HmjI"
        # video_path = ASSETS / "norsk_intro.mp4"
        if st.toggle("Spill av video", value=False, key="toggle_norsk"):
            st.video(YT_URL); st.caption(" ")

    st.markdown("---")
    st.markdown('<div class="section-title">Snapshot</div>', unsafe_allow_html=True)
    c1m, c2m, c3m, c4m, c5m = st.columns(5)
    c1m.metric("Years in AEC", "12+"); c2m.metric("Data Science Bootcamp", "2023")
    c3m.metric("Master’s in Landscape Architecture", "2012")
    c4m.metric("Registered Landscape Architect", "NM #572")
    c5m.metric("SITES Accredited Professional", "#4082")

    st.markdown("---")
    st.markdown('<div class="section-title">Skill & Language Radars</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: radar_chart("Data Science Skills", DS_SKILLS)
    with c2: radar_chart("AEC Toolkit", AEC_SKILLS)
    with c3: radar_chart("Languages", LANG_LEVELS)

    # Work Samples
    st.markdown(f"<div id='{SECTION_IDS['work']}' class='section'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Work Samples</div>', unsafe_allow_html=True)
    st.markdown('<div class="portfolio-grid">', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, p in enumerate(PORTFOLIO[:6]):
        with cols[i % 3]:
            st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)

            # cover
            st.markdown('<div class="portfolio-cover">', unsafe_allow_html=True)
            thumb = p["thumbnail"]
            if Path(thumb).exists():
                st.image(thumb, use_container_width=True)
            else:
                st.image(f"https://picsum.photos/seed/{p['slug']}/600/400", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # title
            st.markdown(f'<div class="portfolio-title">{p["title"]}</div>', unsafe_allow_html=True)

            # button
            st.markdown('<div class="portfolio-button">', unsafe_allow_html=True)
            if st.button("View project", key=f"open_{p['slug']}", use_container_width=True):
                set_view("case"); set_case(p["slug"]); rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Download / Contact
    st.markdown(f"<div id='{SECTION_IDS['dlc']}' class='section'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Download / Contact</div>', unsafe_allow_html=True)
    cL, cR = st.columns([1, 1])
    with cL:
        st.subheader("Downloads")
        onepage = ASSETS / "Drew Seavey Resume Aug 2025.pdf"
        portfolio = ASSETS / "keystone_capstone.pdf"
        if onepage.exists():
            st.download_button("⬇️  Resume", data=onepage.read_bytes(), file_name="Drew Seavey Resume Aug 2025.pdf", use_container_width=True)
        else: st.error("Missing assets/resume_1page.pdf")
        if portfolio.exists():
            st.download_button("⬇️  Data Science Capstone Project", data=portfolio.read_bytes(), file_name="keystone_capstone.pdf", use_container_width=True)
        else: st.error("Missing assets/resume_portfolio.pdf")
        st.caption(" ")
    with cR:
        st.subheader("Contact")
        st.write("I am a data scientist and SITES AP accredited landscape architect working at the intersection of the design world and emerging technology. I integrate geospatial intelligence, predictive analytics, and computational workflows to advance sustainable, technology-driven design.")
        st.write("**Email:** drew.seavey@gmail.com")
        st.write("**LinkedIn:** [linkedin.com/in/drewseavey](https://linkedin.com/in/drewseavey)")
        st.write("**GitHub:** [github.com/AKSeavey](https://github.com/AKSeavey)")
        # st.caption("(Replace with your actual links.)")

# -----------------------------
# CASE (subpage)
# -----------------------------
def get_portfolio_item(slug: str) -> Optional[Dict]:
    return next((p for p in PORTFOLIO if p["slug"] == slug), None)

@st.cache_data(show_spinner=False)
def _page_jpeg(pdf_path_str: str, page_index: int, dpi: int, grayscale: bool, quality: int) -> bytes:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path_str)
    try:
        mat = fitz.Matrix(dpi/72, dpi/72)
        cs = fitz.csGRAY if grayscale else fitz.csRGB
        page = doc.load_page(page_index)
        pix  = page.get_pixmap(matrix=mat, colorspace=cs, alpha=False)
        try:
            return pix.tobytes("jpeg", quality=quality)
        except TypeError:
            return pix.tobytes("jpeg")
    finally:
        doc.close()

def show_pdf_scrollable_fixed(
    pdf_path: Path,
    dpi: int = 105,
    quality: int = 70,
    grayscale: bool = False,
    chunk_size: int = 6,
    key_prefix: str = "pdfscroll",
    show_jump: bool = False,
):
    import fitz
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        st.error(f"Missing PDF at: {pdf_path}")
        return
    with fitz.open(str(pdf_path)) as d:
        n_pages = d.page_count
    st.caption(f"{pdf_path.name} · {n_pages} pages · {dpi} DPI{' · grayscale' if grayscale else ''}")

    shown_key = f"{key_prefix}_shown"
    if shown_key not in st.session_state:
        st.session_state[shown_key] = min(chunk_size, n_pages)
    shown = st.session_state[shown_key]

    for i in range(shown):
        img = _page_jpeg(str(pdf_path), i, dpi, grayscale, quality)
        st.image(img, caption=f"Page {i+1}/{n_pages}", use_container_width=True)

    if shown < n_pages:
        if st.button(f"Show next {min(chunk_size, n_pages-shown)} pages", key=f"{key_prefix}_more", use_container_width=True):
            st.session_state[shown_key] = min(shown + chunk_size, n_pages)
            st.rerun()

    if show_jump:
        with st.expander("Jump to page", expanded=False):
            j = st.slider("Go to page", 1, n_pages, 1, key=f"{key_prefix}_jump")
            img = _page_jpeg(str(pdf_path), j-1, dpi, grayscale, quality)
            st.image(img, caption=f"Page {j}/{n_pages}", use_container_width=True)

def tiny_ga_demo():
    st.caption("Toy multi-objective search: exploring tradeoffs between cost and water (lower is better).")
    n = st.slider("Population size", 40, 200, 80, 10)
    steps = st.slider("Iterations", 10, 400, 120, 10)
    seed = st.slider("Seed", 0, 9999, 42, 1)
    np.random.seed(seed)
    pop = np.random.rand(n, 2)
    def evaluate(xy):
        x, y = xy[...,0], xy[...,1]
        f1 = (x - 0.2)**2 + 0.1*y
        f2 = 0.15*x + (y - 0.8)**2
        return np.c_[f1, f2]
    objs = evaluate(pop)
    for _ in range(steps):
        trial = pop + np.random.normal(0, 0.05, pop.shape)
        trial = np.clip(trial, 0, 1)
        trial_objs = evaluate(trial)
        keep = (trial_objs[:,0] + trial_objs[:,1]) < (objs[:,0] + objs[:,1])
        pop[keep] = trial[keep]; objs[keep] = trial_objs[keep]
    def non_dominated(points):
        m = len(points); nd = np.ones(m, dtype=bool)
        for i in range(m):
            if not nd[i]: continue
            nd &= np.any(points < points[i], axis=1) | np.all(points == points[i], axis=1)
            nd[i] = True
        return nd
    nd_mask = non_dominated(objs)
    dom = objs[~nd_mask]; nd = objs[nd_mask]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=200)
    ax.scatter(dom[:,0], dom[:,1], s=16, alpha=0.4, label="Dominated")
    ax.scatter(nd[:,0], nd[:,1], s=28, alpha=0.9, label="Approx. Pareto")
    ax.set_xlabel("Objective 1 (lower is better)"); ax.set_ylabel("Objective 2 (lower is better)")
    ax.set_title("Toy Multi-Objective Tradeoffs"); ax.grid(True, linestyle="dotted", alpha=0.25); ax.legend()
    st.pyplot(fig, transparent=True)

def render_case(slug: str):
    disable_scroll_restoration()
    enforce_case_top()
    item = get_portfolio_item(slug)
    if not item:
        st.error("Project not found.")
        if st.button("← Back to Home"):
            set_view("home"); set_case(None); rerun()
        return

    cols = st.columns([1.2, 1])
    with cols[0]:
        st.title(item.get("title", ""))
        USE_HTML_BLURB = True   # set False if you still see truncation

        if item.get("blurb"):
            if USE_HTML_BLURB:
                safe_blurb = escape(item["blurb"]).replace("\n", "<br/>")
                st.markdown(f"<div class='blurb'>{safe_blurb}</div>", unsafe_allow_html=True)
            else:
                st.write(item["blurb"])   # Streamlit handles wrap; no ellipsis
        meta_bits = []
        if item.get("role"):    meta_bits.append(f"<div class='kv-item'><b>My Role</b><br/>{item['role']}</div>")
        if item.get("outcome"): meta_bits.append(f"<div class='kv-item'><b>Outcome</b><br/>{item['outcome']}</div>")
        if meta_bits: st.markdown(f"<div class='kv-grid'>{''.join(meta_bits)}</div>", unsafe_allow_html=True)
        if item.get("stack"):
            st.markdown("**Stack / Methods**")
            st.markdown("".join([f"<span class='badge'>{t}</span>" for t in item["stack"]]), unsafe_allow_html=True)
    with cols[1]:
        thumb = item.get("thumbnail")
        if thumb and Path(thumb).exists():
            st.image(thumb, use_container_width=True)
        else:
            st.image(f"https://picsum.photos/seed/{item['slug']}/900/540", use_container_width=True)
    st.markdown("---")

    kind = item.get("kind")
    if kind == "diagram":
        st.subheader("Workflow Diagram (abstracted)")
        st.caption("IP-safe: shows the shape of the approach, not project specifics.")
        img_path = item.get("diagram_image")
        if img_path and Path(img_path).exists():
            st.image(img_path, use_container_width=True)
        else:
            st.info("Add a diagram image via item['diagram_image'] (PNG/JPG/SVG).")

    elif kind == "pdf":
        st.subheader("Document")
        pdf_path = item.get("pdf")
        if pdf_path:
            show_pdf_scrollable_fixed(
                pdf_path, dpi=105, quality=70, grayscale=False,
                chunk_size=6, key_prefix=f"pdf_{item['slug']}", show_jump=False
            )
        else:
            st.warning("No PDF path set for this item yet.")

    elif kind == "demo":
        demo_key = item.get("demo")
        if demo_key == "pymoo-discgolf":
            st.subheader("Genetic Design — Disc Golf Course Optimizer (NSGA-II)")
            pymoo_discgolf_page()
        elif demo_key == "pymoo-lite":
            st.subheader("Mini Genetic Algorithm Demo")
            tiny_ga_demo()
        else:
            st.info("Demo placeholder. Configure a supported `demo` key (e.g., 'pymoo-discgolf' or 'pymoo-lite').")

    elif kind == "story":
        st.subheader(" ")
        img_path = item.get("story_image")
        if img_path and Path(img_path).exists():
            st.image(img_path, use_container_width=True)
        # st.markdown(
        #     "- **Soil:** compost + mulch, periodic pH checks\n"
        #     "- **Water:** drip-line + manual deep-watering during heat waves\n"
        #     "- **Yields:** tomatoes, peppers, herbs; experimenting with drought-tolerant varietals"
        # )
    else:
        st.info("This case hasn’t been configured yet. Set `kind` to 'diagram' | 'pdf' | 'demo' | 'story'.")

    show_details = item.get("show_details", DEFAULT_SHOW_DETAILS)
    if show_details:
        with st.expander("Details (Problem → Approach → Artifacts → Impact)", expanded=False):
            st.subheader("Problem / Context"); st.write("_Briefly describe the problem, constraints, and stakeholders._")
            st.subheader("Approach"); st.write("_Data sources, models/algorithms, geospatial steps, UX decisions, iterations._")
            st.subheader("Key Artifacts"); st.write("_Dashboards, code snippets, parameter panels, maps/figures._")
            st.subheader("Results & Impact"); st.write("_Quantified outcomes, speedups, clarity for stakeholders, lessons learned._")

    links = item.get("links", [])
    if links:
        st.subheader("Links")
        for link in links:
            label = link.get("label", "Resource")
            url = link.get("url"); path = link.get("path")
            if url:
                st.markdown(f"- [{label}]({url})")
            elif path and Path(path).exists():
                st.download_button(f"⬇️ {label}", data=Path(path).read_bytes(),
                                   file_name=Path(path).name, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Back to Home", use_container_width=True):
            set_view("home"); set_case(None); set_pending_jump(SECTION_IDS["work"]); rerun()
    with c2:
        slugs = [p["slug"] for p in PORTFOLIO]
        idx = slugs.index(slug)
        next_slug = slugs[(idx + 1) % len(slugs)]
        if st.button(f"Next: {get_portfolio_item(next_slug)['title']} →", use_container_width=True):
            st.session_state["_force_top"] = True
            set_view("case"); set_case(next_slug); rerun()

    force_case_top_strong()

# -----------------------------
# Dispatch
# -----------------------------
if get_view() == "home":
    render_home()
else:
    render_case(get_case_slug() or PORTFOLIO[0]["slug"])
