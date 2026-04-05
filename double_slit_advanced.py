"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ADVANCED DOUBLE SLIT INTERFERENCE SIMULATION                       ║
║          Research-Level Wave Optics & Photonics Project                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author      : Advanced Engineering Physics / Photonics Research            ║
║  Version     : 2.0  (research-grade upgrade)                                ║
║  Libraries   : NumPy · Matplotlib · SciPy (optional)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  THEORETICAL FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ASSUMPTIONS IN THIS MODEL
  ──────────────────────────
  1. Small-angle approximation  (sinθ ≈ tanθ ≈ θ)  valid when y << L.
     This allows path difference Δ = d·sinθ  to be written as  d·y/L.

  2. Scalar diffraction theory:  polarisation effects are neglected.
     The two waves are treated as scalar fields superposing algebraically.

  3. Coherent, monochromatic illumination for single-λ runs.
     White-light mode relaxes this by superposing independent wavelengths
     (spatially coherent but mutually incoherent across wavelengths).

  4. Far-field (Fraunhofer) condition:  L >> d²/λ
     For d = 0.25 mm, λ = 550 nm  →  d²/λ ≈ 0.11 m.
     With L = 1 m the condition is satisfied.

  CORE EQUATIONS
  ──────────────
  Intensity at screen position y:
      I(y) = I_env(y) · [I₁ + I₂ + 2√(I₁·I₂)·cos(δ)]

  Phase difference:
      δ = 2π·d·y / (λ·L)          [small-angle approx.]

  Single-slit diffraction envelope (Fraunhofer):
      I_env = sinc²(β),  β = π·a·y / (λ·L)

  Equal slits (I₁ = I₂ = I₀/4):
      I(y) = I₀ · sinc²(β) · cos²(δ/2)

  Fringe visibility (Michelson contrast):
      V = (I_max - I_min) / (I_max + I_min)  ∈ [0, 1]
      Perfect coherence → V = 1; incoherent → V = 0.

  Fringe spacing (analytic):
      Δy = λ·L / d

  GAUSSIAN BEAM ENVELOPE (laser realism):
      A Gaussian laser beam has transverse profile  exp(-r²/w²),
      where w is the 1/e² beam radius.  This modulates I(y) as:
          I_gauss(y) = exp(-2·y² / w_screen²)
      where w_screen = w₀·√(1 + (λL/πw₀²)²) via Gaussian beam propagation.

  FOURIER OPTICS (research extension):
      The far-field diffraction pattern equals the squared magnitude of the
      Fourier transform of the aperture transmission function t(x):
          U(y) ∝ ∫ t(x) · exp(-i·2π·x·y / (λ·L)) dx
          I(y) = |U(y)|²
      For two slits of width a centred at ±d/2:
          T(f) = a·[sinc(a·f)·(e^{iπdf} + e^{-iπdf})]
               = 2a·sinc(a·f)·cos(π·d·f)
      where f = y/(λ·L) is the spatial frequency coordinate.
      This approach is fundamental to Fourier optics and optical signal
      processing — the basis of holography, spatial filtering, and
      coherent imaging systems.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODULE MAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODULE 1  ── Constants & colour utilities
  MODULE 2  ── Physics engine (all computation functions)
  MODULE 3  ── Data analysis (fringe metrics)
  MODULE 4  ── Static plotting suite (7 plot functions)
  MODULE 5  ── Animation
  MODULE 6  ── Enhanced interactive widget
  MODULE 7  ── Main orchestrator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Standard library ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from   matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from   matplotlib.colors  import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches
from   matplotlib.ticker  import AutoMinorLocator

# ══════════════════════════════════════════════════════════════════════════
#  MODULE 1 ── CONSTANTS & COLOUR UTILITIES
# ══════════════════════════════════════════════════════════════════════════

# Physical constant
C_LIGHT: float = 2.998e8          # Speed of light (m/s)  [not used in static
                                   # optics but useful for temporal coherence]

# ── Wavelength → perceptual RGB lookup table ───────────────────────────────
#    Hand-tuned CIE-approximate mapping for visible spectrum 380–750 nm.
#    Values in [0, 1] for each of R, G, B.
_WL_RGB_TABLE: dict = {
    380: (0.18, 0.00, 0.22),
    420: (0.40, 0.00, 0.80),
    450: (0.00, 0.00, 1.00),
    490: (0.00, 0.60, 1.00),
    510: (0.00, 1.00, 0.40),
    550: (0.40, 1.00, 0.00),
    580: (1.00, 0.90, 0.00),
    610: (1.00, 0.50, 0.00),
    650: (1.00, 0.00, 0.00),
    700: (0.70, 0.00, 0.00),
    750: (0.35, 0.00, 0.00),
}


def wavelength_to_rgb(wavelength_nm: float) -> tuple:
    """
    Convert a visible-light wavelength to an approximate perceptual RGB colour.

    Uses piecewise linear interpolation over a hand-tuned table that
    approximates the CIE colour matching functions for 380–750 nm.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometres.  Values outside [380, 750] are clamped.

    Returns
    -------
    tuple of (r, g, b), each a float in [0, 1].
    """
    keys = sorted(_WL_RGB_TABLE.keys())
    wl   = float(np.clip(wavelength_nm, keys[0], keys[-1]))

    for i in range(len(keys) - 1):
        if keys[i] <= wl <= keys[i + 1]:
            t   = (wl - keys[i]) / (keys[i + 1] - keys[i])
            c0  = np.array(_WL_RGB_TABLE[keys[i]])
            c1  = np.array(_WL_RGB_TABLE[keys[i + 1]])
            return tuple(np.clip(c0 + t * (c1 - c0), 0.0, 1.0))
    return (1.0, 1.0, 1.0)


def make_wavelength_cmap(wavelength_nm: float) -> LinearSegmentedColormap:
    """
    Build a black → dim → bright-wavelength-colour colormap for 2-D images.

    Having the low end black means dark fringes look like the absence of
    light, which is physically correct.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nm — sets the saturated endpoint colour.

    Returns
    -------
    matplotlib LinearSegmentedColormap
    """
    rgb = wavelength_to_rgb(wavelength_nm)
    mid = tuple(c * 0.40 for c in rgb)   # 40 % brightness midpoint
    cmap = LinearSegmentedColormap.from_list(
        f"wl_{int(wavelength_nm)}",
        [(0.00, (0.02, 0.02, 0.04)),
         (0.30, mid),
         (1.00, rgb)]
    )
    return cmap


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 2 ── PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════════════════

# ── 2.1  Fringe spacing (analytic) ────────────────────────────────────────

def fringe_spacing(
    wavelength_m:   float,
    slit_sep_m:     float,
    screen_dist_m:  float
) -> float:
    """
    Return the analytic fringe spacing  Δy = λL/d  (metres).

    This result follows directly from requiring the path difference
    d·sinθ = mλ  and applying the small-angle approximation sinθ ≈ y/L.

    Parameters
    ----------
    wavelength_m   : Wavelength λ in metres.
    slit_sep_m     : Centre-to-centre slit separation d in metres.
    screen_dist_m  : Screen distance L in metres.

    Returns
    -------
    float : Fringe spacing Δy in metres.
    """
    return (wavelength_m * screen_dist_m) / slit_sep_m


# ── 2.2  Gaussian beam parameters ─────────────────────────────────────────

def gaussian_beam_radius(
    waist_m:       float,
    wavelength_m:  float,
    propagation_m: float
) -> float:
    """
    Compute the 1/e² beam radius after propagating distance z from the waist.

    Gaussian beam propagation (paraxial approximation):
        w(z) = w₀ · √( 1 + (z/z_R)² )
    where z_R = π·w₀²/λ  is the Rayleigh range.

    This is fundamental to laser optics: it tells us how much a focused
    laser beam spreads as it travels — relevant to any real experiment
    that uses a laser source rather than a plane wave.

    Parameters
    ----------
    waist_m       : Beam waist w₀ (metres) — minimum beam radius at focus.
    wavelength_m  : Wavelength λ (metres).
    propagation_m : Propagation distance z (metres) from waist to slits.

    Returns
    -------
    float : Beam radius w(z) in metres at position z.
    """
    z_R = np.pi * waist_m**2 / wavelength_m   # Rayleigh range
    return waist_m * np.sqrt(1.0 + (propagation_m / z_R) ** 2)


def gaussian_beam_envelope_1d(
    y:             np.ndarray,
    beam_waist_m:  float,
    wavelength_m:  float,
    screen_dist_m: float,
    z_source_m:    float = 0.10
) -> np.ndarray:
    """
    Compute the Gaussian beam transverse intensity profile on the screen.

    A real laser source produces a beam with a Gaussian transverse profile.
    After propagating through the slits and diffracting to the screen, the
    overall envelope of the fringe pattern is shaped by this beam profile,
    not by a flat-top plane wave.

    Model:
        The beam radius at the slit plane (z = z_source from waist) is w_slit.
        On the screen (z = z_source + L), the beam further expands to w_screen.
        The intensity envelope is  I_env(y) = exp(-2y²/w_screen²).

    Parameters
    ----------
    y             : Screen position array (m).
    beam_waist_m  : Laser beam waist w₀ (m).  Typical HeNe: 0.5–1 mm.
    wavelength_m  : Wavelength (m).
    screen_dist_m : Slit-to-screen distance L (m).
    z_source_m    : Waist-to-slit distance (m).

    Returns
    -------
    np.ndarray : Normalised Gaussian envelope values in [0, 1].
    """
    # Beam radius at the screen
    w_screen = gaussian_beam_radius(
        beam_waist_m, wavelength_m, z_source_m + screen_dist_m
    )
    # Gaussian intensity profile:  I ∝ exp(-2r²/w²)
    return np.exp(-2.0 * y**2 / w_screen**2)


# ── 2.3  Core 1-D intensity (full physics model) ──────────────────────────

def compute_intensity_1d(
    y:              np.ndarray,
    wavelength_m:   float,
    slit_sep_m:     float,
    screen_dist_m:  float,
    slit_width_m:   float  = 4.0e-5,
    I1:             float  = 1.0,
    I2:             float  = 1.0,
    misalign_m:     float  = 0.0,
    noise_sigma:    float  = 0.0,
    use_gaussian:   bool   = False,
    beam_waist_m:   float  = 5.0e-4,
    rng_seed:       int    = 42
) -> np.ndarray:
    """
    Compute the 1-D double-slit interference intensity pattern.

    This is the core physics function.  It supports:
      • Equal or unequal slit intensities  (I1, I2)
      • Slit misalignment (vertical shift of one slit)
      • Gaussian beam illumination envelope
      • Additive Gaussian detector noise
      • Single-slit diffraction envelope

    FULL FORMULA (unequal slits + misalignment):
        I(y) = sinc²(β₁)·I₁ + sinc²(β₂)·I₂
               + 2·√(I₁·I₂)·sinc(β₁)·sinc(β₂)·cos(δ + Δφ_misalign)

    where:
        δ        = π·d·y / (λ·L)          [half the phase difference]
        β₁       = π·a·(y + d/2) / (λ·L)  [diffraction arg, slit 1]
        β₂       = π·a·(y - d/2) / (λ·L)  [diffraction arg, slit 2]
        Δφ_misalign = 2π·Δy_slit·y / (λ·L) [extra phase from misalignment]

    For equal, aligned slits this simplifies to I₀·sinc²(β)·cos²(δ/2).

    Parameters
    ----------
    y             : Screen position array (m).
    wavelength_m  : Wavelength λ (m).
    slit_sep_m    : Centre-to-centre slit separation d (m).
    screen_dist_m : Screen distance L (m).
    slit_width_m  : Individual slit width a (m).  Controls diffraction envelope.
    I1, I2        : Relative intensities of each slit.  Default 1.0 each.
                    Setting I2 = 0.8 simulates partial blockage of slit 2.
    misalign_m    : Lateral (y) misalignment of slit 2 (m).  0 = perfect.
    noise_sigma   : Std. dev. of additive Gaussian noise as fraction of peak.
    use_gaussian  : If True, multiply by Gaussian beam envelope.
    beam_waist_m  : Beam waist w₀ for Gaussian envelope (m).
    rng_seed      : Seed for reproducible noise.

    Returns
    -------
    np.ndarray : Intensity values ≥ 0, normalised so ideal peak = 1.
    """
    lam = wavelength_m
    L   = screen_dist_m
    d   = slit_sep_m
    a   = slit_width_m

    # ── Phase difference term (small-angle approx.) ────────────────────────
    # δ/2 = π·d·y/(λ·L).  The full phase δ = 2π·d·sinθ/λ ≈ 2π·d·y/(λ·L).
    half_delta = (np.pi * d * y) / (lam * L)

    # ── Diffraction envelope for each slit ────────────────────────────────
    # Single-slit Fraunhofer intensity: sinc²(β), β = π·a·y/(λ·L).
    # We compute it at the EFFECTIVE screen coordinate for each slit.
    def _sinc(beta: np.ndarray) -> np.ndarray:
        """Safe sinc: sin(β)/β with limit = 1 at β = 0."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(beta == 0, 1.0, np.sin(beta) / beta)

    beta1 = (np.pi * a * y) / (lam * L)          # slit 1 diffraction arg
    # Slit 2 shifted by misalign_m → effective y shifts
    y2    = y - misalign_m                         # coords relative to slit 2
    beta2 = (np.pi * a * y2) / (lam * L)          # slit 2 diffraction arg

    s1 = _sinc(beta1)
    s2 = _sinc(beta2)

    # ── Extra phase from misalignment ─────────────────────────────────────
    # If slit 2 is shifted by Δ in y, its path to point y on screen changes
    # by approximately Δ·y/L (small-angle), giving extra phase:
    #   Δφ = 2π·Δ·y / (λ·L)
    phi_mis = (np.pi * misalign_m * y) / (lam * L)

    # ── Full two-beam superposition ───────────────────────────────────────
    # I = |E₁ + E₂|² time-averaged
    #   = I₁·sinc²(β₁) + I₂·sinc²(β₂) + 2√(I₁I₂)·sinc₁·sinc₂·cos(δ + Δφ)
    E1_sq = I1 * s1**2
    E2_sq = I2 * s2**2
    cross  = 2.0 * np.sqrt(I1 * I2) * s1 * s2 * np.cos(half_delta + phi_mis)

    # Normalise so that the ideal peak (all terms at max) ≈ 1
    peak   = (np.sqrt(I1) + np.sqrt(I2)) ** 2       # maximum possible value
    I_out  = (E1_sq + E2_sq + cross) / peak

    # ── Gaussian beam envelope ────────────────────────────────────────────
    if use_gaussian:
        env    = gaussian_beam_envelope_1d(y, beam_waist_m, lam, L)
        I_out *= env

    # ── Additive Gaussian noise ───────────────────────────────────────────
    # Models detector shot noise / dark current.  Sigma is expressed as a
    # fraction of the normalised peak intensity.
    if noise_sigma > 0.0:
        rng    = np.random.default_rng(rng_seed)
        I_out  = I_out + rng.normal(0.0, noise_sigma, size=I_out.shape)

    return np.clip(I_out, 0.0, None)   # intensity must be non-negative


# ── 2.4  Core 2-D intensity ───────────────────────────────────────────────

def compute_intensity_2d(
    x:              np.ndarray,
    y:              np.ndarray,
    wavelength_m:   float,
    slit_sep_m:     float,
    screen_dist_m:  float,
    slit_width_m:   float  = 4.0e-5,
    I1:             float  = 1.0,
    I2:             float  = 1.0,
    misalign_m:     float  = 0.0,
    noise_sigma:    float  = 0.0,
    use_gaussian:   bool   = False,
    beam_waist_m:   float  = 5.0e-4,
    rng_seed:       int    = 42
) -> np.ndarray:
    """
    Compute 2-D double-slit intensity on an (x, y) grid.

    The fringe pattern varies only along y (perpendicular to slits).
    The x-dimension provides the width of the image — in the ideal
    case the pattern is uniform along x, but a Gaussian beam envelope
    or misalignment can break this symmetry.

    Parameters
    ----------
    x, y          : 1-D coordinate arrays (m).  Grid shape = (len(y), len(x)).
    (remaining)   : See compute_intensity_1d — same physics parameters.

    Returns
    -------
    np.ndarray : 2-D intensity array of shape (len(y), len(x)), values ≥ 0.
    """
    # Compute 1-D pattern along y with full physics
    I_1d = compute_intensity_1d(
        y, wavelength_m, slit_sep_m, screen_dist_m,
        slit_width_m, I1, I2, misalign_m, noise_sigma,
        use_gaussian, beam_waist_m, rng_seed
    )

    # For ideal / plane-wave illumination: I2d constant along x.
    # For Gaussian: also apply Gaussian envelope along x.
    if use_gaussian:
        w_screen = gaussian_beam_radius(beam_waist_m, wavelength_m,
                                        0.10 + screen_dist_m)
        env_x    = np.exp(-2.0 * x**2 / w_screen**2)
        # Broadcasting: I_1d is (Ny,), env_x is (Nx,) → outer product (Ny, Nx)
        I2d = np.outer(I_1d, env_x)
    else:
        # Simple broadcast: replicate 1-D pattern across x
        I2d = np.broadcast_to(I_1d[:, np.newaxis], (len(y), len(x))).copy()

    return np.clip(I2d, 0.0, None)


# ── 2.5  Fourier optics aperture approach (research extension) ────────────

def compute_fourier_optics_1d(
    y:              np.ndarray,
    wavelength_m:   float,
    slit_sep_m:     float,
    screen_dist_m:  float,
    slit_width_m:   float  = 4.0e-5,
    N_aperture:     int    = 8192
) -> np.ndarray:
    """
    Compute the Fraunhofer diffraction pattern via numerical Fourier transform.

    ╔═══════════════════════════════════════════════════════════╗
    ║  RESEARCH RELEVANCE                                       ║
    ║  ─────────────────────────────────────────────────────── ║
    ║  Fourier optics is the mathematical backbone of:          ║
    ║  • Holography and wavefront reconstruction                ║
    ║  • Coherent optical imaging systems (microscopy, LIDAR)   ║
    ║  • Spatial light modulators (SLMs) and beam shaping       ║
    ║  • Optical coherence tomography (OCT)                     ║
    ║  • Photonic integrated circuit design                     ║
    ║                                                           ║
    ║  The core idea: the far-field diffraction pattern equals   ║
    ║  the Fourier transform of the aperture function.          ║
    ║  This simulation validates our analytic formulae by       ║
    ║  computing the same pattern numerically — a standard      ║
    ║  cross-check in computational photonics.                  ║
    ╚═══════════════════════════════════════════════════════════╝

    Method:
        1. Sample the aperture transmission function t(x_ap) on a
           fine grid of width >> d + a.
        2. Set t = 1 inside each slit, 0 outside (binary mask).
        3. Apply np.fft.fft to get the Fourier spectrum U(f).
        4. Map spatial frequency f to screen coordinate y = f·λ·L.
        5. Interpolate onto the requested y grid.

    Parameters
    ----------
    y             : Output screen position array (m).
    wavelength_m  : Wavelength λ (m).
    slit_sep_m    : Slit centre-to-centre separation d (m).
    screen_dist_m : Screen distance L (m).
    slit_width_m  : Individual slit width a (m).
    N_aperture    : Number of aperture sample points (higher = more accurate).

    Returns
    -------
    np.ndarray : Normalised intensity, same shape as y.
    """
    # ── Step 1: define aperture plane grid ────────────────────────────────
    # Must span at least several slit widths for accuracy.
    ap_half = 5.0 * slit_sep_m               # aperture half-width
    x_ap    = np.linspace(-ap_half, ap_half, N_aperture)
    dx      = x_ap[1] - x_ap[0]

    # ── Step 2: build aperture transmission function ───────────────────────
    # Slit 1 centred at +d/2, Slit 2 centred at -d/2
    half_a = slit_width_m / 2.0
    t = (
        (np.abs(x_ap - slit_sep_m / 2.0) <= half_a).astype(float) +
        (np.abs(x_ap + slit_sep_m / 2.0) <= half_a).astype(float)
    )
    # Note: t is exactly 0 or 1 — a binary aperture mask.

    # ── Step 3: numerical Fourier transform ───────────────────────────────
    # U(f) = ∫ t(x)·exp(-i·2π·f·x) dx  ≈  FFT(t)·dx
    T      = np.fft.fftshift(np.fft.fft(t)) * dx

    # ── Step 4: map frequency axis to screen coordinates ──────────────────
    # Spatial frequency: f = x_ap / (λ·L)  → screen coord: y = f·λ·L
    freq   = np.fft.fftshift(np.fft.fftfreq(N_aperture, d=dx))
    y_fft  = freq * wavelength_m * screen_dist_m

    # ── Step 5: intensity and interpolation onto requested y grid ─────────
    I_fft  = np.abs(T) ** 2                          # intensity ∝ |U|²

    # Interpolate: only use the region where y_fft covers y
    mask   = (y_fft >= y.min()) & (y_fft <= y.max())
    I_interp = np.interp(y, y_fft[mask], I_fft[mask])

    # Normalise to [0, 1]
    mx = I_interp.max()
    return I_interp / mx if mx > 0 else I_interp


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 3 ── DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def compute_fringe_visibility(I: np.ndarray) -> float:
    """
    Compute the Michelson fringe visibility (contrast) of an intensity pattern.

        V = (I_max - I_min) / (I_max + I_min)

    Interpretation:
      V = 1.0  →  perfect contrast (fully coherent, equal-amplitude slits)
      V = 0.5  →  reduced contrast (unequal slits, partial coherence, or noise)
      V = 0.0  →  no fringes (incoherent superposition)

    For a clean sinusoidal pattern with noise, the peak/trough values are
    computed as the 99th and 1st percentile to avoid outlier sensitivity.

    Parameters
    ----------
    I : np.ndarray
        1-D intensity array.

    Returns
    -------
    float : Visibility V ∈ [0, 1].
    """
    I_max = np.percentile(I, 99)
    I_min = np.percentile(I, 1)
    denom = I_max + I_min
    return float((I_max - I_min) / denom) if denom > 0 else 0.0


def analyse_fringe_pattern(
    y:  np.ndarray,
    I:  np.ndarray
) -> dict:
    """
    Extract quantitative fringe metrics from a 1-D intensity array.

    Metrics computed:
      • Fringe visibility V (Michelson contrast)
      • Measured fringe spacing Δy (mm) via peak-finding
      • Theoretical fringe spacing Δy_theory (mm) — passed in separately
      • Peak positions (mm)
      • Number of visible fringes

    Parameters
    ----------
    y : np.ndarray  Screen positions (m).
    I : np.ndarray  Intensity values (same shape as y).

    Returns
    -------
    dict with keys: 'visibility', 'fringe_spacing_mm', 'n_fringes',
                    'peak_positions_mm', 'I_max', 'I_min'.
    """
    # Visibility
    V = compute_fringe_visibility(I)

    # Peak finding — simple: local maxima above half the global max
    threshold = 0.5 * np.max(I)
    peaks = []
    for i in range(1, len(I) - 1):
        if I[i] > I[i - 1] and I[i] > I[i + 1] and I[i] > threshold:
            peaks.append(i)

    peak_y_mm = y[peaks] * 1e3 if len(peaks) > 0 else np.array([])

    # Average spacing between consecutive peaks
    if len(peaks) >= 2:
        spacings = np.diff(y[peaks]) * 1e3      # mm
        delta_y  = float(np.mean(spacings))
    else:
        delta_y = float("nan")

    return {
        "visibility":        V,
        "fringe_spacing_mm": delta_y,
        "n_fringes":         len(peaks),
        "peak_positions_mm": peak_y_mm,
        "I_max":             float(np.max(I)),
        "I_min":             float(np.min(I)),
    }


def print_analysis_report(
    wavelength_nm:  float,
    slit_sep_mm:    float,
    screen_dist_m:  float,
    y:              np.ndarray,
    I_ideal:        np.ndarray,
    I_realistic:    np.ndarray
) -> None:
    """
    Print a formatted analysis report comparing ideal and realistic patterns.

    Parameters
    ----------
    wavelength_nm  : Wavelength (nm).
    slit_sep_mm    : Slit separation (mm).
    screen_dist_m  : Screen distance (m).
    y              : Screen position array (m).
    I_ideal        : Ideal intensity array.
    I_realistic    : Realistic intensity array (with noise/misalignment).
    """
    lam   = wavelength_nm * 1e-9
    d     = slit_sep_mm   * 1e-3
    L     = screen_dist_m

    delta_y_theory = fringe_spacing(lam, d, L) * 1e3   # mm

    a_ideal = analyse_fringe_pattern(y, I_ideal)
    a_real  = analyse_fringe_pattern(y, I_realistic)

    w = 62
    print("═" * w)
    print("   DOUBLE SLIT INTERFERENCE — ADVANCED ANALYSIS REPORT")
    print("═" * w)
    print(f"   Parameters")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   Wavelength           λ = {wavelength_nm} nm  ({lam:.2e} m)")
    print(f"   Slit separation      d = {slit_sep_mm} mm  ({d:.2e} m)")
    print(f"   Screen distance      L = {screen_dist_m} m")
    print(f"   Theoretical Δy (λL/d) = {delta_y_theory:.3f} mm")
    print(f"   Rayleigh range (a=0.04mm): z_R = "
          f"{(np.pi*(0.5e-3)**2/lam)*1e2:.1f} cm")
    print()
    print(f"   {'Metric':<30}{'Ideal':>12}{'Realistic':>12}")
    print(f"   {'─'*30}{'─'*12}{'─'*12}")
    V_i  = a_ideal['visibility']
    V_r  = a_real['visibility']
    dy_i = a_ideal['fringe_spacing_mm']
    dy_r = a_real['fringe_spacing_mm']
    n_i  = a_ideal['n_fringes']
    n_r  = a_real['n_fringes']

    dy_i_str = f"{dy_i:.3f}" if not np.isnan(dy_i) else "N/A"
    dy_r_str = f"{dy_r:.3f}" if not np.isnan(dy_r) else "N/A"
    print(f"   {'Fringe visibility V':<30}{V_i:>12.4f}{V_r:>12.4f}")
    print(f"   {'Measured Δy (mm)':<30}{dy_i_str:>12}{dy_r_str:>12}")
    print(f"   {'Visible fringe peaks':<30}{n_i:>12}{n_r:>12}")
    print(f"   {'I_max (normalised)':<30}"
          f"{a_ideal['I_max']:>12.4f}{a_real['I_max']:>12.4f}")
    print(f"   {'I_min (normalised)':<30}"
          f"{a_ideal['I_min']:>12.4f}{a_real['I_min']:>12.4f}")
    print()
    print(f"   Visibility reduction due to realistic effects:")
    if V_i > 0:
        print(f"   ΔV = {V_i - V_r:.4f}  ({100*(V_i-V_r)/V_i:.1f}% reduction)")
    print("═" * w)
    print()


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 4 ── STATIC PLOTTING SUITE
# ══════════════════════════════════════════════════════════════════════════

# ── Shared dark-theme style helper ────────────────────────────────────────

def _style_ax(ax, bg: str = "#080810") -> None:
    """Apply consistent dark-theme styling to a Matplotlib Axes."""
    ax.set_facecolor(bg)
    ax.tick_params(colors="white", labelsize=8, which="both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="minor", length=2, color="#555577")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.grid(True, color="#1c1c33", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", color="#111122", lw=0.3, ls=":", alpha=0.5)


def _style_fig(fig, bg: str = "#05050e") -> None:
    """Apply consistent dark-theme styling to a Matplotlib Figure."""
    fig.patch.set_facecolor(bg)


# ── Plot 1: Multi-wavelength 1-D comparison ───────────────────────────────

def plot_1d_multi_wavelength(
    wavelengths_nm:  list,
    slit_sep_mm:     float = 0.25,
    screen_dist_m:   float = 1.0,
    screen_half_cm:  float = 3.0,
    slit_width_mm:   float = 0.04,
    save_path:       str   = None
) -> plt.Figure:
    """
    Plot normalised 1-D interference intensity vs screen position
    for multiple wavelengths simultaneously, each curve coloured by
    its physical wavelength.

    Shows: How fringe spacing Δy = λL/d scales linearly with wavelength.

    Parameters
    ----------
    wavelengths_nm  : List of wavelengths (nm) to overlay.
    slit_sep_mm     : Slit separation d (mm).
    screen_dist_m   : Screen distance L (m).
    screen_half_cm  : Half-width of plotted screen region (cm).
    slit_width_mm   : Individual slit width a (mm).
    save_path       : Optional file path to save PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    d     = slit_sep_mm   * 1e-3
    L     = screen_dist_m
    a     = slit_width_mm * 1e-3
    y_max = screen_half_cm * 1e-2

    y = np.linspace(-y_max, y_max, 10000)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    _style_fig(fig); _style_ax(ax)

    for wl_nm in wavelengths_nm:
        lam = wl_nm * 1e-9
        rgb = wavelength_to_rgb(wl_nm)
        I   = compute_intensity_1d(y, lam, d, L, a)

        ax.plot(y * 1e3, I, color=rgb, linewidth=1.6, alpha=0.90,
                label=f"λ = {wl_nm} nm  (Δy = {fringe_spacing(lam,d,L)*1e3:.2f} mm)")
        ax.fill_between(y * 1e3, 0, I, color=rgb, alpha=0.07)

    # Central maximum marker
    ax.axvline(0, color="#aaaacc", lw=0.7, ls="--", alpha=0.5,
               label="Central maximum (m = 0)")

    # Annotation: fringe spacing for first wavelength
    lam0    = wavelengths_nm[0] * 1e-9
    dy0_mm  = fringe_spacing(lam0, d, L) * 1e3
    V0      = compute_fringe_visibility(
                  compute_intensity_1d(y, lam0, d, L, a))

    info_text = (
        f"d = {slit_sep_mm} mm  |  L = {screen_dist_m} m  |  a = {slit_width_mm} mm\n"
        f"Ref. λ₁ = {wavelengths_nm[0]} nm  →  Δy = {dy0_mm:.2f} mm  "
        f"|  V = {V0:.3f}"
    )
    ax.text(0.01, 0.97, info_text, transform=ax.transAxes,
            color="#ccccee", fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#10101e",
                      edgecolor="#444466", alpha=0.85))

    ax.set_xlabel("Screen Position  y  (mm)", color="white", fontsize=12)
    ax.set_ylabel("Normalised Intensity  I / I₀", color="white", fontsize=12)
    ax.set_title(
        "Young's Double Slit — Multi-Wavelength 1-D Interference Pattern\n"
        "Single-slit diffraction envelope included  |  Small-angle approximation",
        color="white", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xlim(-y_max * 1e3, y_max * 1e3)
    ax.set_ylim(-0.04, 1.14)
    ax.legend(facecolor="#0e0e1e", edgecolor="#444466",
              labelcolor="white", fontsize=9.5, loc="upper right")
    ax.xaxis.label.set_color("white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Plot 2: Ideal vs Realistic comparison ─────────────────────────────────

def plot_ideal_vs_realistic(
    wavelength_nm:  float = 532.0,
    slit_sep_mm:    float = 0.25,
    screen_dist_m:  float = 1.0,
    slit_width_mm:  float = 0.04,
    noise_sigma:    float = 0.04,
    misalign_um:    float = 30.0,
    I2_factor:      float = 0.75,
    screen_half_cm: float = 3.0,
    save_path:      str   = None
) -> plt.Figure:
    """
    Side-by-side comparison of ideal vs realistic interference patterns.

    Realistic effects modelled:
      1. Additive Gaussian detector noise  (noise_sigma)
      2. Slit 2 lateral misalignment  (misalign_um µm)
      3. Unequal slit intensities  (I2_factor < 1)

    Each effect reduces the fringe visibility V from the ideal value of ~1.
    This plot directly demonstrates how experimental imperfections degrade
    the interference pattern — critical knowledge for experimental optics.

    Parameters
    ----------
    wavelength_nm  : Wavelength (nm).
    slit_sep_mm    : Slit separation (mm).
    screen_dist_m  : Screen distance (m).
    slit_width_mm  : Slit width (mm).
    noise_sigma    : Noise std. dev. as fraction of peak.
    misalign_um    : Slit 2 vertical misalignment (µm).
    I2_factor      : Relative intensity of slit 2  (0 < I2 ≤ 1.0).
    screen_half_cm : Half-screen width (cm).
    save_path      : Optional PNG save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lam   = wavelength_nm * 1e-9
    d     = slit_sep_mm   * 1e-3
    L     = screen_dist_m
    a     = slit_width_mm * 1e-3
    Δ     = misalign_um   * 1e-6

    y = np.linspace(-screen_half_cm * 1e-2, screen_half_cm * 1e-2, 12000)
    rgb   = wavelength_to_rgb(wavelength_nm)

    # ── Compute patterns ──────────────────────────────────────────────────
    I_ideal = compute_intensity_1d(y, lam, d, L, a)

    I_noisy = compute_intensity_1d(y, lam, d, L, a,
                                   noise_sigma=noise_sigma)

    I_misaligned = compute_intensity_1d(y, lam, d, L, a,
                                        misalign_m=Δ)

    I_unequal = compute_intensity_1d(y, lam, d, L, a,
                                     I2=I2_factor)

    I_full_realistic = compute_intensity_1d(
        y, lam, d, L, a,
        I2=I2_factor, misalign_m=Δ, noise_sigma=noise_sigma
    )

    # Compute visibilities
    V_labels = [
        ("Ideal",             I_ideal,         rgb,           "-"),
        ("+ Noise only",      I_noisy,         (0.9,0.7,0.3), "--"),
        ("+ Misalignment",    I_misaligned,    (0.3,0.8,0.9), "-."),
        ("+ Unequal slits",   I_unequal,       (0.8,0.4,0.8), ":"),
        ("All effects",       I_full_realistic,(0.9,0.4,0.3), "-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _style_fig(fig)
    ax_left, ax_right = axes

    # ── Left: overlay of all five curves ─────────────────────────────────
    _style_ax(ax_left)
    for label, I_arr, col, ls in V_labels:
        V = compute_fringe_visibility(I_arr)
        ax_left.plot(y * 1e3, I_arr, color=col, lw=1.5, ls=ls,
                     label=f"{label}  (V = {V:.3f})", alpha=0.88)

    ax_left.set_xlabel("Screen Position  y  (mm)", color="white", fontsize=11)
    ax_left.set_ylabel("Normalised Intensity  I / I₀", color="white", fontsize=11)
    ax_left.set_title("Effect of Each Imperfection on Fringe Pattern",
                      color="white", fontsize=11, fontweight="bold")
    ax_left.set_xlim(-screen_half_cm * 10, screen_half_cm * 10)
    ax_left.set_ylim(-0.05, 1.20)
    ax_left.legend(facecolor="#0e0e1e", edgecolor="#444466",
                   labelcolor="white", fontsize=8.5, loc="upper right")

    # ── Right: ideal vs combined realistic ───────────────────────────────
    _style_ax(ax_right)
    V_ideal = compute_fringe_visibility(I_ideal)
    V_real  = compute_fringe_visibility(I_full_realistic)

    ax_right.plot(y * 1e3, I_ideal, color=rgb, lw=2.0,
                  label=f"Ideal  (V = {V_ideal:.3f})", alpha=0.9)
    ax_right.fill_between(y * 1e3, 0, I_ideal, color=rgb, alpha=0.08)

    ax_right.plot(y * 1e3, I_full_realistic, color=(0.9,0.4,0.3), lw=1.5,
                  ls="--", label=f"Realistic  (V = {V_real:.3f})", alpha=0.9)
    ax_right.fill_between(y * 1e3, 0, I_full_realistic,
                          color=(0.9,0.4,0.3), alpha=0.06)

    # Annotate visibility degradation
    ax_right.text(
        0.02, 0.96,
        f"λ = {wavelength_nm:.0f} nm  |  d = {slit_sep_mm} mm  |  L = {screen_dist_m} m\n"
        f"Noise σ = {noise_sigma:.0%}  |  Misalign = {misalign_um:.0f} µm  |  I₂/I₁ = {I2_factor:.2f}\n"
        f"Visibility loss:  {V_ideal:.3f} → {V_real:.3f}  (ΔV = {V_ideal-V_real:.3f})",
        transform=ax_right.transAxes, color="#ccccee",
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#10101e",
                  edgecolor="#4444aa", alpha=0.88)
    )
    ax_right.set_xlabel("Screen Position  y  (mm)", color="white", fontsize=11)
    ax_right.set_ylabel("Normalised Intensity  I / I₀", color="white", fontsize=11)
    ax_right.set_title("Ideal vs Full Realistic Comparison",
                       color="white", fontsize=11, fontweight="bold")
    ax_right.set_xlim(-screen_half_cm * 10, screen_half_cm * 10)
    ax_right.set_ylim(-0.05, 1.20)
    ax_right.legend(facecolor="#0e0e1e", edgecolor="#444466",
                    labelcolor="white", fontsize=9.5)

    fig.suptitle(
        "Ideal vs Realistic Double Slit Interference  —  Fringe Visibility Analysis",
        color="white", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Plot 3: 2-D fringe heatmap ────────────────────────────────────────────

def plot_2d_fringe_pattern(
    wavelength_nm:  float = 532.0,
    slit_sep_mm:    float = 0.25,
    screen_dist_m:  float = 1.0,
    slit_width_mm:  float = 0.04,
    screen_half_cm: float = 2.5,
    use_gaussian:   bool  = False,
    noise_sigma:    float = 0.0,
    save_path:      str   = None
) -> plt.Figure:
    """
    Render a high-resolution 2-D fringe pattern heatmap.

    Left panel: 1-D intensity profile (sideways).
    Right panel: 2-D colour-mapped image with diffraction envelope.

    Parameters
    ----------
    wavelength_nm  : Wavelength (nm).
    slit_sep_mm    : Slit separation (mm).
    screen_dist_m  : Screen distance (m).
    slit_width_mm  : Slit width (mm).
    screen_half_cm : Half-height of rendered screen (cm).
    use_gaussian   : If True, apply Gaussian beam envelope.
    noise_sigma    : Noise level (fraction of peak).
    save_path      : Optional PNG save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lam   = wavelength_nm * 1e-9
    d     = slit_sep_mm   * 1e-3
    L     = screen_dist_m
    a     = slit_width_mm * 1e-3
    y_h   = screen_half_cm * 1e-2
    x_h   = y_h * 0.30

    n_y, n_x = 1400, 480     # higher resolution for publication quality
    y = np.linspace(-y_h, y_h, n_y)
    x = np.linspace(-x_h, x_h, n_x)

    I2d  = compute_intensity_2d(x, y, lam, d, L, a,
                                 use_gaussian=use_gaussian,
                                 noise_sigma=noise_sigma)
    cmap  = make_wavelength_cmap(wavelength_nm)
    rgb   = wavelength_to_rgb(wavelength_nm)

    # Metrics
    I1d  = compute_intensity_1d(y, lam, d, L, a,
                                 use_gaussian=use_gaussian,
                                 noise_sigma=noise_sigma)
    V    = compute_fringe_visibility(I1d)
    dy   = fringe_spacing(lam, d, L) * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"width_ratios": [1, 3.5]})
    _style_fig(fig)

    # ── Left: 1-D profile ─────────────────────────────────────────────────
    ax1 = axes[0]
    _style_ax(ax1)
    ax1.plot(I1d, y * 1e3, color=rgb, lw=1.5, alpha=0.9)
    ax1.fill_betweenx(y * 1e3, 0, I1d, color=rgb, alpha=0.14)
    ax1.set_xlabel("Intensity", color="white", fontsize=9)
    ax1.set_ylabel("Screen Position  y  (mm)", color="white", fontsize=9)
    ax1.set_title(f"1-D\nProfile\nV = {V:.3f}", color="white",
                  fontsize=9, fontweight="bold")
    ax1.set_xlim(-0.05, 1.20); ax1.set_ylim(y[0]*1e3, y[-1]*1e3)
    ax1.invert_xaxis()
    ax1.text(0.95, 0.97, f"Δy = {dy:.2f} mm",
             transform=ax1.transAxes, ha="right", color="#aaffaa",
             fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1a0a",
                       edgecolor="#33aa33", alpha=0.85))

    # ── Right: 2-D heatmap ────────────────────────────────────────────────
    ax2 = axes[1]
    _style_ax(ax2)
    extent = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]
    im = ax2.imshow(
        I2d, extent=extent, origin="lower", cmap=cmap,
        aspect="auto", vmin=0, vmax=1, interpolation="bilinear"
    )
    cb = plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cb.set_label("Normalised Intensity  I / I₀", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    mode_str = " | Gaussian beam" if use_gaussian else " | Plane wave"
    noise_str = f" | noise σ={noise_sigma:.2f}" if noise_sigma > 0 else " | noiseless"
    ax2.set_xlabel("Horizontal Position  x  (mm)", color="white", fontsize=10)
    ax2.set_title(
        f"2-D Fringe Heatmap  |  λ = {wavelength_nm:.0f} nm  |  "
        f"d = {slit_sep_mm} mm  |  L = {screen_dist_m} m{mode_str}{noise_str}",
        color="white", fontsize=10.5, fontweight="bold"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Plot 4: White-light simulation ────────────────────────────────────────

def plot_white_light_pattern(
    slit_sep_mm:    float = 0.25,
    screen_dist_m:  float = 1.0,
    slit_width_mm:  float = 0.04,
    screen_half_cm: float = 2.5,
    save_path:      str   = None
) -> plt.Figure:
    """
    Simulate a white-light double-slit fringe pattern.

    White light is modelled as an incoherent superposition of
    31 monochromatic wavelengths from 400 to 700 nm (10 nm spacing).
    Each wavelength contributes its own fringe pattern at its own colour;
    the contributions are summed to give a composite RGB image.

    Physical consequence: the central maximum is white (all wavelengths
    in phase at y = 0), while outer orders show chromatic dispersion —
    the blue edge (short λ, small fringe spacing) is on the inside,
    red edge (long λ, larger spacing) on the outside.

    Parameters
    ----------
    slit_sep_mm    : Slit separation (mm).
    screen_dist_m  : Screen distance (m).
    slit_width_mm  : Slit width (mm).
    screen_half_cm : Half-screen height (cm).
    save_path      : Optional PNG save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    d     = slit_sep_mm   * 1e-3
    a     = slit_width_mm * 1e-3
    L     = screen_dist_m
    y_h   = screen_half_cm * 1e-2
    x_h   = y_h * 0.30

    n_y, n_x = 1400, 480
    y = np.linspace(-y_h, y_h, n_y)
    x = np.linspace(-x_h, x_h, n_x)

    wl_range = np.arange(400, 701, 10, dtype=float)

    R = np.zeros((n_y, n_x))
    G = np.zeros_like(R)
    B = np.zeros_like(R)

    for wl in wl_range:
        I2d      = compute_intensity_2d(x, y, wl*1e-9, d, L, a)
        r, g, b  = wavelength_to_rgb(wl)
        R += r * I2d
        G += g * I2d
        B += b * I2d

    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-15)

    rgb_img = np.stack([_norm(R), _norm(G), _norm(B)], axis=-1)

    # 1-D white-light profile (sum of 1-D patterns)
    I_wl = np.zeros(n_y)
    for wl in wl_range:
        I_wl += compute_intensity_1d(y, wl*1e-9, d, L, a)
    I_wl /= I_wl.max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"width_ratios": [1, 3.5]})
    _style_fig(fig)

    # Left: 1-D
    ax1 = axes[0]
    _style_ax(ax1)
    ax1.plot(I_wl, y * 1e3, color="white", lw=1.4)
    ax1.fill_betweenx(y * 1e3, 0, I_wl, color="white", alpha=0.10)
    ax1.set_xlabel("Intensity (norm.)", color="white", fontsize=9)
    ax1.set_ylabel("Screen Position  y  (mm)", color="white", fontsize=9)
    ax1.set_title("White Light\n1-D Profile", color="white",
                  fontsize=9, fontweight="bold")
    ax1.set_xlim(-0.05, 1.20); ax1.set_ylim(y[0]*1e3, y[-1]*1e3)
    ax1.invert_xaxis()

    # Right: 2-D
    ax2 = axes[1]
    _style_ax(ax2)
    extent = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]
    ax2.imshow(rgb_img, extent=extent, origin="lower",
               aspect="auto", interpolation="bilinear")
    ax2.set_xlabel("Horizontal Position  x  (mm)", color="white", fontsize=10)
    ax2.set_title(
        f"White-Light Fringe Pattern  (400–700 nm, 31 wavelengths)\n"
        f"d = {slit_sep_mm} mm  |  L = {screen_dist_m} m  |  "
        f"Central max = white  |  Outer orders = chromatic dispersion",
        color="white", fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Plot 5: Gaussian beam vs plane-wave comparison ────────────────────────

def plot_gaussian_beam_effect(
    wavelength_nm:  float = 532.0,
    slit_sep_mm:    float = 0.25,
    screen_dist_m:  float = 1.0,
    slit_width_mm:  float = 0.04,
    beam_waist_mm:  float = 0.50,
    screen_half_cm: float = 3.5,
    save_path:      str   = None
) -> plt.Figure:
    """
    Compare interference patterns for plane-wave vs Gaussian beam input.

    A Gaussian beam has a finite transverse extent (beam waist w₀).
    When it illuminates the double slit, the intensity envelope on the
    screen is no longer flat — it falls off as exp(-2y²/w_screen²).
    For a tightly focused beam (small w₀), only the central few fringes
    are bright; for a large beam (w₀ >> d) the pattern approaches the
    plane-wave limit.

    This comparison is directly relevant to:
    • Laser-based interference experiments
    • Free-space optical communications (beam truncation)
    • Optical trapping (Gaussian beam focusing)

    Parameters
    ----------
    wavelength_nm  : Wavelength (nm).
    slit_sep_mm    : Slit separation (mm).
    screen_dist_m  : Screen distance (m).
    slit_width_mm  : Slit width (mm).
    beam_waist_mm  : Laser beam waist w₀ (mm).
    screen_half_cm : Half-screen width (cm).
    save_path      : Optional PNG save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lam    = wavelength_nm * 1e-9
    d      = slit_sep_mm   * 1e-3
    L      = screen_dist_m
    a      = slit_width_mm * 1e-3
    w0     = beam_waist_mm * 1e-3
    y_max  = screen_half_cm * 1e-2

    y      = np.linspace(-y_max, y_max, 12000)

    I_plane   = compute_intensity_1d(y, lam, d, L, a, use_gaussian=False)
    I_gauss   = compute_intensity_1d(y, lam, d, L, a,
                                      use_gaussian=True, beam_waist_m=w0)
    envelope  = gaussian_beam_envelope_1d(y, w0, lam, L)

    # Beam radius on screen
    w_screen_mm = gaussian_beam_radius(w0, lam, 0.10 + L) * 1e3
    z_R_mm      = (np.pi * w0**2 / lam) * 1e3

    rgb = wavelength_to_rgb(wavelength_nm)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _style_fig(fig)

    # ── Left: plane wave ──────────────────────────────────────────────────
    _style_ax(axes[0])
    axes[0].plot(y*1e3, I_plane, color=rgb, lw=1.6,
                 label="Plane wave (ideal)")
    axes[0].fill_between(y*1e3, 0, I_plane, color=rgb, alpha=0.10)
    axes[0].set_title("Plane-Wave Illumination\n(uniform amplitude at slits)",
                      color="white", fontsize=11, fontweight="bold")
    V_plane = compute_fringe_visibility(I_plane)
    axes[0].text(0.02, 0.96, f"V = {V_plane:.4f}  |  Δy = {fringe_spacing(lam,d,L)*1e3:.2f} mm",
                 transform=axes[0].transAxes, color="#ccffcc", fontsize=9,
                 va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a150a",
                           edgecolor="#33aa33", alpha=0.85))

    # ── Right: Gaussian beam ──────────────────────────────────────────────
    _style_ax(axes[1])
    axes[1].plot(y*1e3, I_gauss, color=rgb, lw=1.6,
                 label=f"Gaussian beam (w₀ = {beam_waist_mm} mm)")
    axes[1].plot(y*1e3, envelope, color="white", lw=1.0, ls="--",
                 alpha=0.55, label=f"Gaussian envelope  w_screen = {w_screen_mm:.1f} mm")
    axes[1].fill_between(y*1e3, 0, I_gauss, color=rgb, alpha=0.10)
    axes[1].set_title(
        f"Gaussian Beam Illumination\n"
        f"w₀ = {beam_waist_mm} mm  |  z_R = {z_R_mm:.0f} mm",
        color="white", fontsize=11, fontweight="bold"
    )
    V_gauss = compute_fringe_visibility(I_gauss)
    axes[1].text(0.02, 0.96,
                 f"V = {V_gauss:.4f}  |  w_screen = {w_screen_mm:.1f} mm\n"
                 f"Rayleigh range z_R = {z_R_mm:.0f} mm",
                 transform=axes[1].transAxes, color="#ccffcc", fontsize=9,
                 va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a150a",
                           edgecolor="#33aa33", alpha=0.85))
    axes[1].legend(facecolor="#0e0e1e", edgecolor="#444466",
                   labelcolor="white", fontsize=9)

    for ax in axes:
        ax.set_xlabel("Screen Position  y  (mm)", color="white", fontsize=11)
        ax.set_ylabel("Normalised Intensity  I / I₀", color="white", fontsize=11)
        ax.set_xlim(-y_max*1e3, y_max*1e3)
        ax.set_ylim(-0.04, 1.14)

    fig.suptitle(
        f"Gaussian Beam vs Plane-Wave Illumination  |  λ = {wavelength_nm} nm  "
        f"|  d = {slit_sep_mm} mm  |  L = {screen_dist_m} m",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Plot 6: Fourier optics validation ─────────────────────────────────────

def plot_fourier_optics_validation(
    wavelength_nm:  float = 532.0,
    slit_sep_mm:    float = 0.25,
    screen_dist_m:  float = 1.0,
    slit_width_mm:  float = 0.04,
    screen_half_cm: float = 3.0,
    save_path:      str   = None
) -> plt.Figure:
    """
    Validate the analytic formula against the Fourier optics (FFT) method.

    This plot is the bridge between the analytic wave-optics treatment and
    the Fourier optics framework used in advanced photonics research.

    The Fourier optics result:
        U(y) ∝ FT[t(x)](y / λL)
    should match the analytic formula
        I(y) = sinc²(β) · cos²(δ/2)
    to within numerical precision.  Any discrepancy reveals the limits of
    the small-angle approximation or finite aperture effects.

    The third panel shows the aperture transmission function t(x) to make
    the connection between aperture and diffraction pattern explicit.

    Parameters
    ----------
    (all same as above)

    Returns
    -------
    matplotlib.figure.Figure
    """
    lam   = wavelength_nm * 1e-9
    d     = slit_sep_mm   * 1e-3
    L     = screen_dist_m
    a     = slit_width_mm * 1e-3
    y_h   = screen_half_cm * 1e-2

    y = np.linspace(-y_h, y_h, 10000)

    I_analytic = compute_intensity_1d(y, lam, d, L, a)
    I_fft      = compute_fourier_optics_1d(y, lam, d, L, a, N_aperture=16384)

    # Residual
    residual   = I_analytic - I_fft

    # Aperture function for visualisation
    ap_half    = 5.0 * d
    x_ap       = np.linspace(-ap_half, ap_half, 2000)
    half_a     = a / 2.0
    t_ap       = (
        (np.abs(x_ap - d / 2.0) <= half_a).astype(float) +
        (np.abs(x_ap + d / 2.0) <= half_a).astype(float)
    )

    rgb = wavelength_to_rgb(wavelength_nm)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    _style_fig(fig)

    # Panel 1: Aperture function
    _style_ax(axes[0])
    axes[0].fill_between(x_ap*1e3, 0, t_ap, color="#44aaff", alpha=0.7,
                         label="t(x) = aperture mask")
    axes[0].axhline(0, color="#555577", lw=0.5)
    axes[0].set_xlabel("Aperture position  x  (mm)", color="white", fontsize=10)
    axes[0].set_ylabel("Transmission  t(x)", color="white", fontsize=10)
    axes[0].set_title("Aperture Function\n(binary double-slit mask)",
                      color="white", fontsize=10, fontweight="bold")
    axes[0].set_xlim(-ap_half*1e3, ap_half*1e3)
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].legend(facecolor="#0e0e1e", edgecolor="#444466",
                   labelcolor="white", fontsize=8)
    # Annotate slit positions
    for pos_mm in [d/2*1e3, -d/2*1e3]:
        axes[0].axvline(pos_mm, color="#ffaaaa", lw=0.8, ls=":",
                        alpha=0.6)
    axes[0].text(d/2*1e3 + 0.01, 1.15, f"d/2={d/2*1e3:.2f}mm",
                 color="#ffaaaa", fontsize=7)

    # Panel 2: Analytic vs FFT overlay
    _style_ax(axes[1])
    axes[1].plot(y*1e3, I_analytic, color=rgb,       lw=2.0,
                 label="Analytic  I = sinc²(β)·cos²(δ/2)", alpha=0.85)
    axes[1].plot(y*1e3, I_fft,      color="white",   lw=1.2,
                 ls="--", label="Fourier optics  |FT[t]|² (FFT)",
                 alpha=0.75)
    axes[1].set_xlabel("Screen Position  y  (mm)", color="white", fontsize=10)
    axes[1].set_ylabel("Normalised Intensity  I / I₀", color="white", fontsize=10)
    axes[1].set_title("Analytic vs Fourier Optics\n(should be identical)",
                      color="white", fontsize=10, fontweight="bold")
    axes[1].set_xlim(-y_h*1e3, y_h*1e3)
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].legend(facecolor="#0e0e1e", edgecolor="#444466",
                   labelcolor="white", fontsize=8.5)
    rms = np.sqrt(np.mean(residual**2))
    axes[1].text(0.02, 0.96, f"RMS residual = {rms:.2e}",
                 transform=axes[1].transAxes, color="#aaffaa",
                 fontsize=8.5, va="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a150a",
                           edgecolor="#33aa33", alpha=0.85))

    # Panel 3: Residual
    _style_ax(axes[2])
    axes[2].plot(y*1e3, residual, color="#ff9944", lw=1.2)
    axes[2].axhline(0, color="#555577", lw=0.8, ls="--")
    axes[2].fill_between(y*1e3, 0, residual, color="#ff9944", alpha=0.25)
    axes[2].set_xlabel("Screen Position  y  (mm)", color="white", fontsize=10)
    axes[2].set_ylabel("Residual  (Analytic − FFT)", color="white", fontsize=10)
    axes[2].set_title(f"Residual Error\nRMS = {rms:.2e}",
                      color="white", fontsize=10, fontweight="bold")
    axes[2].set_xlim(-y_h*1e3, y_h*1e3)

    fig.suptitle(
        f"Fourier Optics Validation  |  λ = {wavelength_nm} nm  "
        f"|  d = {slit_sep_mm} mm  |  a = {slit_width_mm} mm  |  L = {screen_dist_m} m",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ── Plot 7: Parameter sensitivity dashboard ───────────────────────────────

def plot_parameter_sensitivity(
    save_path: str = None
) -> plt.Figure:
    """
    3×3 dashboard: how fringe pattern changes when one parameter varies.

    Row 1 — vary wavelength λ (d, L fixed)
    Row 2 — vary slit separation d (λ, L fixed)
    Row 3 — vary screen distance L (λ, d fixed)

    Each panel annotates the computed fringe spacing Δy and visibility V.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(17, 12))
    _style_fig(fig)
    gs   = gridspec.GridSpec(3, 3, figure=fig, hspace=0.60, wspace=0.35)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    y = np.linspace(-3e-2, 3e-2, 10000)

    param_sets = [
        # Row 0: vary λ
        [
            dict(lam_nm=440, d_mm=0.25, L_m=1.0, a_mm=0.04,
                 label="λ = 440 nm (blue)"),
            dict(lam_nm=532, d_mm=0.25, L_m=1.0, a_mm=0.04,
                 label="λ = 532 nm (green)"),
            dict(lam_nm=633, d_mm=0.25, L_m=1.0, a_mm=0.04,
                 label="λ = 633 nm (red)"),
        ],
        # Row 1: vary d
        [
            dict(lam_nm=532, d_mm=0.15, L_m=1.0, a_mm=0.04,
                 label="d = 0.15 mm"),
            dict(lam_nm=532, d_mm=0.25, L_m=1.0, a_mm=0.04,
                 label="d = 0.25 mm"),
            dict(lam_nm=532, d_mm=0.40, L_m=1.0, a_mm=0.04,
                 label="d = 0.40 mm"),
        ],
        # Row 2: vary L
        [
            dict(lam_nm=532, d_mm=0.25, L_m=0.5,  a_mm=0.04,
                 label="L = 0.5 m"),
            dict(lam_nm=532, d_mm=0.25, L_m=1.0,  a_mm=0.04,
                 label="L = 1.0 m"),
            dict(lam_nm=532, d_mm=0.25, L_m=1.5,  a_mm=0.04,
                 label="L = 1.5 m"),
        ],
    ]

    row_colours = [
        [wavelength_to_rgb(w) for w in [440, 532, 633]],
        [(0.95, 0.55, 0.15), (0.15, 0.90, 0.45), (0.55, 0.25, 0.95)],
        [(0.95, 0.20, 0.40), (0.20, 0.75, 0.95), (0.65, 0.95, 0.25)],
    ]
    row_titles = [
        "Varying Wavelength  λ  (d = 0.25 mm, L = 1.0 m)",
        "Varying Slit Separation  d  (λ = 532 nm, L = 1.0 m)",
        "Varying Screen Distance  L  (λ = 532 nm, d = 0.25 mm)",
    ]

    for row in range(3):
        for col, params in enumerate(param_sets[row]):
            ax  = axes[row][col]
            _style_ax(ax)
            lam = params["lam_nm"] * 1e-9
            d   = params["d_mm"]   * 1e-3
            L   = params["L_m"]
            a   = params["a_mm"]   * 1e-3
            c   = row_colours[row][col]

            I   = compute_intensity_1d(y, lam, d, L, a)
            V   = compute_fringe_visibility(I)
            dy  = fringe_spacing(lam, d, L) * 1e3   # mm

            ax.plot(y*1e3, I, color=c, lw=1.4)
            ax.fill_between(y*1e3, 0, I, color=c, alpha=0.10)

            ax.set_title(
                f"{params['label']}\nΔy = {dy:.2f} mm  |  V = {V:.3f}",
                color="white", fontsize=8.5, pad=4
            )
            ax.set_xlim(-30, 30); ax.set_ylim(-0.05, 1.15)

            if col == 0:
                ax.set_ylabel("I / I₀", color="white", fontsize=8)
            if row == 2:
                ax.set_xlabel("y (mm)", color="white", fontsize=8)

        # Row label rotated on the far left
        fig.text(0.005, 0.82 - row * 0.30, row_titles[row],
                 color="white", fontsize=9.5, fontweight="bold",
                 rotation=90, va="center")

    fig.suptitle(
        "Parameter Sensitivity Dashboard — Double Slit Interference\n"
        "All patterns include single-slit diffraction envelope",
        color="white", fontsize=14, fontweight="bold", y=0.99
    )
    plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.06)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 5 ── ANIMATION
# ══════════════════════════════════════════════════════════════════════════

def animate_wavelength_sweep(
    slit_sep_mm:    float = 0.25,
    screen_dist_m:  float = 1.0,
    slit_width_mm:  float = 0.04,
    screen_half_cm: float = 3.0,
    fps:            int   = 30,
    save_path:      str   = None
) -> animation.FuncAnimation:
    """
    Animate the interference pattern as wavelength sweeps 400 → 700 → 400 nm.

    Each frame shows:
      • Updated fringe pattern coloured by current wavelength
      • Updated title with λ, Δy, and V
      • Moving fringe-spacing markers

    The animation clearly demonstrates Δy ∝ λ: the fringes breathe
    in and out as wavelength changes.

    Parameters
    ----------
    slit_sep_mm    : Slit separation (mm).
    screen_dist_m  : Screen distance (m).
    slit_width_mm  : Slit width (mm).
    screen_half_cm : Half-screen width (cm).
    fps            : Frames per second (for saved video).
    save_path      : If given, save as GIF (requires pillow).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    d     = slit_sep_mm   * 1e-3
    L     = screen_dist_m
    a     = slit_width_mm * 1e-3
    y_max = screen_half_cm * 1e-2
    y     = np.linspace(-y_max, y_max, 8000)

    # Wavelength frames: 400→700→400 nm triangle wave
    wl_up   = np.linspace(400, 700, 120)
    wl_down = np.linspace(700, 400, 120)
    wl_frames = np.concatenate([wl_up, wl_down])

    fig, ax = plt.subplots(figsize=(12, 5))
    _style_fig(fig); _style_ax(ax)

    wl0  = wl_frames[0]
    rgb0 = wavelength_to_rgb(wl0)
    I0   = compute_intensity_1d(y, wl0*1e-9, d, L, a)

    [line]    = ax.plot(y*1e3, I0, color=rgb0, lw=1.8)
    fill_coll = [ax.fill_between(y*1e3, 0, I0, color=rgb0, alpha=0.12)]
    beta0     = fringe_spacing(wl0*1e-9, d, L) * 1e3
    vline_p   = ax.axvline( beta0, color="white", lw=0.8, ls=":", alpha=0.45)
    vline_n   = ax.axvline(-beta0, color="white", lw=0.8, ls=":", alpha=0.45)
    title_obj = ax.set_title("", color="white", fontsize=12, fontweight="bold")

    ax.set_xlabel("Screen Position  y  (mm)", color="white", fontsize=12)
    ax.set_ylabel("Normalised Intensity  I / I₀", color="white", fontsize=12)
    ax.set_xlim(-y_max*1e3, y_max*1e3)
    ax.set_ylim(-0.05, 1.18)

    # Spectrum colour bar at bottom (decorative)
    spec_ax = fig.add_axes([0.10, 0.03, 0.80, 0.025])
    spec_x  = np.linspace(400, 700, 500)
    spec_rgb = np.array([wavelength_to_rgb(w) for w in spec_x])
    spec_ax.imshow(spec_rgb[np.newaxis, :, :], aspect="auto",
                   extent=[400, 700, 0, 1])
    spec_ax.set_xlim(400, 700)
    spec_ax.set_yticks([])
    spec_ax.set_xlabel("Wavelength (nm)", color="white", fontsize=8)
    spec_ax.tick_params(colors="white", labelsize=7)
    for sp in spec_ax.spines.values():
        sp.set_edgecolor("#333355")
    # Moving wavelength marker
    wl_marker = spec_ax.axvline(wl0, color="white", lw=1.5, alpha=0.9)

    plt.subplots_adjust(bottom=0.18, top=0.90)

    def update(frame_idx: int):
        wl  = wl_frames[frame_idx]
        lam = wl * 1e-9
        rgb = wavelength_to_rgb(wl)
        I   = compute_intensity_1d(y, lam, d, L, a)
        V   = compute_fringe_visibility(I)
        dy  = fringe_spacing(lam, d, L) * 1e3

        line.set_ydata(I)
        line.set_color(rgb)

        fill_coll[0].remove()
        fill_coll[0] = ax.fill_between(y*1e3, 0, I, color=rgb, alpha=0.12)

        vline_p.set_xdata([dy, dy])
        vline_n.set_xdata([-dy, -dy])
        wl_marker.set_xdata([wl, wl])

        title_obj.set_text(
            f"Double Slit Interference  |  λ = {wl:.0f} nm  "
            f"|  Δy = {dy:.2f} mm  |  V = {V:.3f}"
        )
        title_obj.set_color(rgb)
        return line, vline_p, vline_n, title_obj, wl_marker

    anim = animation.FuncAnimation(
        fig, update, frames=len(wl_frames),
        interval=1000 // fps, blit=False
    )

    if save_path and save_path.endswith(".gif"):
        anim.save(save_path, writer="pillow", fps=fps)
    elif save_path and save_path.endswith(".mp4"):
        anim.save(save_path, writer="ffmpeg", fps=fps, dpi=120)

    return anim


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 6 ── ENHANCED INTERACTIVE WIDGET
# ══════════════════════════════════════════════════════════════════════════

def launch_interactive_widget() -> None:
    """
    Launch a feature-rich interactive Matplotlib widget with:

    Sliders
    ───────
    • λ (wavelength, 400–700 nm)
    • d (slit separation, 0.10–0.60 mm)
    • L (screen distance, 0.5–2.5 m)
    • Noise σ (0–15% of peak)
    • Misalignment Δ (0–100 µm)

    Toggle buttons
    ──────────────
    • Ideal / Realistic mode
    • Plane wave / Gaussian beam

    Real-time metrics display
    ──────────────────────────
    • Fringe spacing Δy (mm) — analytic formula
    • Fringe visibility V — computed from pattern
    • Gaussian beam radius on screen w_screen (mm)
    """
    # ── Default parameters ────────────────────────────────────────────────
    WL0       = 532.0    # nm
    D0        = 0.25     # mm
    L0        = 1.0      # m
    NOISE0    = 0.00
    MISALIGN0 = 0.0      # µm
    A_MM      = 0.04     # slit width (fixed in this widget)
    W0_MM     = 0.50     # Gaussian beam waist

    y = np.linspace(-4e-2, 4e-2, 10000)

    # ── State flags ───────────────────────────────────────────────────────
    state = {
        "realistic": False,
        "gaussian":  False,
    }

    # ── Build figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6.5))
    _style_fig(fig, bg="#04040c")
    _style_ax(ax, bg="#04040c")
    plt.subplots_adjust(left=0.09, right=0.97, bottom=0.42, top=0.90)

    def compute_current(wl_nm, d_mm, L_m, noise, misalign_um):
        lam = wl_nm      * 1e-9
        d   = d_mm       * 1e-3
        L   = L_m
        a   = A_MM       * 1e-3
        w0  = W0_MM      * 1e-3
        Δ   = misalign_um * 1e-6
        use_noise = state["realistic"]
        use_gauss = state["gaussian"]
        I = compute_intensity_1d(
            y, lam, d, L, a,
            misalign_m  = Δ   if use_noise else 0.0,
            noise_sigma = noise if use_noise else 0.0,
            use_gaussian= use_gauss,
            beam_waist_m= w0
        )
        return I

    lam0 = WL0 * 1e-9; d0 = D0 * 1e-3
    rgb0 = wavelength_to_rgb(WL0)
    I0   = compute_current(WL0, D0, L0, NOISE0, MISALIGN0)
    V0   = compute_fringe_visibility(I0)
    dy0  = fringe_spacing(lam0, d0, L0) * 1e3

    [line]    = ax.plot(y*1e3, I0, color=rgb0, lw=1.8, zorder=3)
    fill_ref  = [ax.fill_between(y*1e3, 0, I0, color=rgb0, alpha=0.12)]
    beta0     = dy0
    vline_p   = ax.axvline( beta0, color="white", lw=0.9, ls=":", alpha=0.5)
    vline_n   = ax.axvline(-beta0, color="white", lw=0.9, ls=":", alpha=0.5)

    # Metrics text box
    metrics_text = ax.text(
        0.98, 0.96,
        f"Δy = {dy0:.3f} mm\nV  = {V0:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        color="white", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d0d22",
                  edgecolor="#5555cc", alpha=0.90)
    )

    ax.set_xlabel("Screen Position  y  (mm)", color="white", fontsize=12)
    ax.set_ylabel("Normalised Intensity  I / I₀", color="white", fontsize=12)
    ax.set_title(
        "Advanced Interactive Double Slit Interference Explorer",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.set_xlim(-40, 40); ax.set_ylim(-0.05, 1.18)

    # ── Sliders ───────────────────────────────────────────────────────────
    slider_specs = [
        ("λ (nm)",    400,  700,  WL0,       1,    0.10),
        ("d (mm)",    0.10, 0.60, D0,        0.01, 0.165),
        ("L (m)",     0.50, 2.50, L0,        0.05, 0.23),
        ("Noise σ",   0.00, 0.15, NOISE0,    0.005,0.295),
        ("Δ (µm)",    0.0,  100,  MISALIGN0, 1.0,  0.36),
    ]
    sliders = []
    for label, vmin, vmax, vinit, vstep, ypos in slider_specs:
        s_ax = plt.axes([0.12, ypos, 0.60, 0.022], facecolor="#111128")
        s    = Slider(s_ax, label, vmin, vmax, valinit=vinit,
                      color="#2a2a6a", valstep=vstep)
        s.label.set_color("white"); s.valtext.set_color("#9999ff")
        sliders.append(s)
    s_wl, s_d, s_L, s_noise, s_mis = sliders

    # ── Realistic toggle ──────────────────────────────────────────────────
    rax_real = plt.axes([0.76, 0.29, 0.20, 0.10], facecolor="#0a0a1a")
    rax_real.set_title("Mode", color="white", fontsize=8, pad=2)
    radio_real = RadioButtons(rax_real, ["Ideal", "Realistic"],
                              activecolor="#5555dd")
    for lbl in radio_real.labels:
        lbl.set_color("white"); lbl.set_fontsize(8.5)

    rax_beam = plt.axes([0.76, 0.15, 0.20, 0.10], facecolor="#0a0a1a")
    rax_beam.set_title("Illumination", color="white", fontsize=8, pad=2)
    radio_beam = RadioButtons(rax_beam, ["Plane wave", "Gaussian beam"],
                              activecolor="#5555dd")
    for lbl in radio_beam.labels:
        lbl.set_color("white"); lbl.set_fontsize(8.5)

    # ── Reset button ──────────────────────────────────────────────────────
    ax_btn = plt.axes([0.44, 0.02, 0.10, 0.04])
    btn    = Button(ax_btn, "Reset", color="#18183a", hovercolor="#2a2a5a")
    btn.label.set_color("white")

    # ── Update function ───────────────────────────────────────────────────
    def update(_=None):
        wl_nm     = s_wl.val
        d_mm      = s_d.val
        L_m       = s_L.val
        noise     = s_noise.val
        mis_um    = s_mis.val

        lam  = wl_nm * 1e-9
        d    = d_mm  * 1e-3
        rgb  = wavelength_to_rgb(wl_nm)
        I    = compute_current(wl_nm, d_mm, L_m, noise, mis_um)
        V    = compute_fringe_visibility(I)
        dy   = fringe_spacing(lam, d, L_m) * 1e3

        line.set_ydata(I); line.set_color(rgb)
        fill_ref[0].remove()
        fill_ref[0] = ax.fill_between(y*1e3, 0, I, color=rgb, alpha=0.11)

        vline_p.set_xdata([dy, dy])
        vline_n.set_xdata([-dy, -dy])

        # Gaussian beam info for metrics box
        if state["gaussian"]:
            w_screen = gaussian_beam_radius(W0_MM*1e-3, lam, 0.10+L_m)*1e3
            metrics_text.set_text(
                f"Δy = {dy:.3f} mm\nV  = {V:.4f}\n"
                f"w_sc = {w_screen:.1f} mm"
            )
        else:
            metrics_text.set_text(f"Δy = {dy:.3f} mm\nV  = {V:.4f}")

        mode_str  = "REALISTIC" if state["realistic"] else "IDEAL"
        beam_str  = "Gaussian"  if state["gaussian"]  else "Plane wave"
        ax.set_title(
            f"Advanced Interactive Explorer  |  Mode: {mode_str}  "
            f"|  Beam: {beam_str}",
            color="white", fontsize=12, fontweight="bold"
        )
        fig.canvas.draw_idle()

    def on_real_radio(label):
        state["realistic"] = (label == "Realistic")
        update()

    def on_beam_radio(label):
        state["gaussian"] = (label == "Gaussian beam")
        update()

    def reset(_):
        for s in sliders:
            s.reset()

    for s in sliders:
        s.on_changed(update)
    radio_real.on_clicked(on_real_radio)
    radio_beam.on_clicked(on_beam_radio)
    btn.on_clicked(reset)

    plt.show()


# ══════════════════════════════════════════════════════════════════════════
#  MODULE 7 ── MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════

def print_physics_summary(
    wavelength_nm:  float = 532.0,
    slit_sep_mm:    float = 0.25,
    screen_dist_m:  float = 1.0,
    slit_width_mm:  float = 0.04,
    noise_sigma:    float = 0.04,
    misalign_um:    float = 30.0,
    I2_factor:      float = 0.75
) -> None:
    """
    Print a full physics summary and analysis report to the console.

    Computes and compares both ideal and realistic patterns, printing
    fringe spacing, visibility, and effect of each imperfection.
    """
    lam = wavelength_nm * 1e-9
    d   = slit_sep_mm   * 1e-3
    L   = screen_dist_m
    a   = slit_width_mm * 1e-3
    Δ   = misalign_um   * 1e-6

    y      = np.linspace(-3e-2, 3e-2, 12000)
    I_ideal = compute_intensity_1d(y, lam, d, L, a)
    I_real  = compute_intensity_1d(y, lam, d, L, a,
                                    I2=I2_factor, misalign_m=Δ,
                                    noise_sigma=noise_sigma)

    print_analysis_report(wavelength_nm, slit_sep_mm, screen_dist_m,
                          y, I_ideal, I_real)


def main() -> None:
    """
    Execute the complete advanced simulation pipeline:

    [1]  Physics analysis report (console)
    [2]  Multi-wavelength 1-D comparison
    [3]  Ideal vs Realistic side-by-side
    [4]  High-resolution 2-D fringe heatmap  (plane wave)
    [5]  High-resolution 2-D fringe heatmap  (Gaussian beam)
    [6]  White-light composite simulation
    [7]  Gaussian beam vs plane-wave effect
    [8]  Fourier optics validation
    [9]  Parameter sensitivity dashboard
    [10] Wavelength sweep animation (optional)
    [11] Interactive widget (optional)
    """
    print("\n" + "═"*62)
    print("  ADVANCED DOUBLE SLIT INTERFERENCE SIMULATION  v2.0")
    print("═"*62 + "\n")

    # ── [1] Physics report ────────────────────────────────────────────────
    print("  [1] Computing physics analysis report …\n")
    print_physics_summary()

    # ── [2] Multi-wavelength 1-D ──────────────────────────────────────────
    print("  [2] Multi-wavelength 1-D comparison …")
    plot_1d_multi_wavelength(
        wavelengths_nm=[440, 532, 589, 633],
        slit_sep_mm=0.25, screen_dist_m=1.0,
        save_path="plot1_multiwavelength_1d.png"
    )

    # ── [3] Ideal vs realistic ────────────────────────────────────────────
    print("  [3] Ideal vs Realistic comparison …")
    plot_ideal_vs_realistic(
        wavelength_nm=532, slit_sep_mm=0.25, screen_dist_m=1.0,
        noise_sigma=0.04, misalign_um=30.0, I2_factor=0.75,
        save_path="plot2_ideal_vs_realistic.png"
    )

    # ── [4] 2-D heatmap — plane wave ──────────────────────────────────────
    print("  [4] 2-D fringe heatmap (plane wave) …")
    plot_2d_fringe_pattern(
        wavelength_nm=532, slit_sep_mm=0.25, screen_dist_m=1.0,
        use_gaussian=False,
        save_path="plot3_2d_heatmap_planewave.png"
    )

    # ── [5] 2-D heatmap — Gaussian beam ───────────────────────────────────
    print("  [5] 2-D fringe heatmap (Gaussian beam) …")
    plot_2d_fringe_pattern(
        wavelength_nm=532, slit_sep_mm=0.25, screen_dist_m=1.0,
        use_gaussian=True, noise_sigma=0.02,
        save_path="plot4_2d_heatmap_gaussian.png"
    )

    # ── [6] White-light ───────────────────────────────────────────────────
    print("  [6] White-light composite simulation …")
    plot_white_light_pattern(
        slit_sep_mm=0.25, screen_dist_m=1.0,
        save_path="plot5_white_light.png"
    )

    # ── [7] Gaussian beam effect ──────────────────────────────────────────
    print("  [7] Gaussian beam vs plane-wave effect …")
    plot_gaussian_beam_effect(
        wavelength_nm=532, slit_sep_mm=0.25, screen_dist_m=1.0,
        beam_waist_mm=0.50,
        save_path="plot6_gaussian_beam_effect.png"
    )

    # ── [8] Fourier optics validation ─────────────────────────────────────
    print("  [8] Fourier optics validation (FFT vs analytic) …")
    plot_fourier_optics_validation(
        wavelength_nm=532, slit_sep_mm=0.25, screen_dist_m=1.0,
        slit_width_mm=0.04,
        save_path="plot7_fourier_optics_validation.png"
    )

    # ── [9] Parameter sensitivity ─────────────────────────────────────────
    print("  [9] Parameter sensitivity dashboard …")
    plot_parameter_sensitivity(
        save_path="plot8_parameter_sensitivity.png"
    )

    print("\n  ✓ All 8 plots saved successfully.\n")
    plt.show()

    # ── [10] Animation ────────────────────────────────────────────────────
    ans = input(
        "  [10] Generate wavelength sweep animation? (y/n): "
    ).strip().lower()
    if ans == "y":
        print("       Creating animation …")
        anim = animate_wavelength_sweep(
            slit_sep_mm=0.25, screen_dist_m=1.0,
            save_path="animation_wavelength_sweep.gif"
        )
        plt.show()
        print("       Animation saved as animation_wavelength_sweep.gif")

    # ── [11] Interactive widget ───────────────────────────────────────────
    ans2 = input(
        "  [11] Launch interactive slider widget? (y/n): "
    ).strip().lower()
    if ans2 == "y":
        launch_interactive_widget()

    print("\n  ✓ Simulation complete.\n")


if __name__ == "__main__":
    main()
