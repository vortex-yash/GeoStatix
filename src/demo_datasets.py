"""Built-in reproducible demo datasets for GeoStatix onboarding.

All datasets in this module are synthetic and generated locally with fixed seeds.
They are intentionally small enough to run instantly and are not intended to
represent real deposits or real geological measurements.
"""

import numpy as np
import pandas as pd

from .spatial_analysis import generate_demo_spatial_dataset


def synthetic_spatial_gold(n_samples=200, seed=42):
    """Synthetic spatial gold-grade dataset for mapping and geostatistics."""
    return generate_demo_spatial_dataset(n_samples=n_samples, seed=seed)


def synthetic_skewed_grade(n_samples=500, seed=7):
    """Synthetic positively skewed grade dataset for distribution diagnostics."""
    rng = np.random.default_rng(seed)
    grade = rng.lognormal(mean=-2.8, sigma=0.85, size=n_samples)
    high_grade_idx = rng.choice(n_samples, size=max(5, n_samples // 50), replace=False)
    grade[high_grade_idx] *= rng.uniform(2.0, 5.0, size=len(high_grade_idx))

    return pd.DataFrame(
        {
            "Sample_ID": [f"SG-{i:04d}" for i in range(1, n_samples + 1)],
            "Gold_Grade_gpt": grade,
        }
    )


def synthetic_multivariate_geology(n_samples=300, seed=21):
    """Synthetic multivariate geological/geochemical dataset."""
    rng = np.random.default_rng(seed)
    depth = np.sort(rng.uniform(5, 500, n_samples))
    lithology = rng.choice(
        ["Granite", "Basalt", "Schist", "Quartzite"],
        size=n_samples,
        p=[0.28, 0.24, 0.30, 0.18],
    )

    lithology_effect = pd.Series(lithology).map(
        {"Granite": 0.08, "Basalt": 0.16, "Schist": 0.11, "Quartzite": 0.05}
    ).to_numpy()

    copper = np.clip(
        0.05 + 0.00045 * depth + lithology_effect + rng.normal(0, 0.035, n_samples),
        0.001,
        None,
    )
    iron = np.clip(
        25 + 0.018 * depth + np.where(lithology == "Basalt", 8, 0) + rng.normal(0, 3.0, n_samples),
        1,
        None,
    )
    density = np.clip(
        2.55 + 0.0011 * depth + np.where(lithology == "Quartzite", 0.12, 0) + rng.normal(0, 0.06, n_samples),
        2.0,
        None,
    )
    moisture = np.clip(rng.normal(3.2, 1.0, n_samples), 0.2, None)

    return pd.DataFrame(
        {
            "Sample_ID": [f"GM-{i:04d}" for i in range(1, n_samples + 1)],
            "Depth_m": depth,
            "Lithology": lithology,
            "Cu_pct": copper,
            "Fe_pct": iron,
            "Density_gcc": density,
            "Moisture_pct": moisture,
        }
    )


def synthetic_spatial_multivariate(n_samples=220, seed=84):
    """Synthetic spatial dataset with several correlated grade variables."""
    rng = np.random.default_rng(seed)
    base = generate_demo_spatial_dataset(n_samples=n_samples, seed=seed)

    gold = base["Gold_Grade"].to_numpy(dtype=float)
    normalized_gold = (gold - gold.mean()) / (gold.std() or 1.0)

    copper = np.clip(
        0.55 + 0.12 * normalized_gold + rng.normal(0, 0.055, n_samples),
        0.01,
        None,
    )
    iron = np.clip(
        35 + 5.0 * normalized_gold + rng.normal(0, 2.5, n_samples),
        5,
        None,
    )
    density = np.clip(
        2.65 + 0.035 * normalized_gold + rng.normal(0, 0.025, n_samples),
        2.0,
        None,
    )

    return pd.DataFrame(
        {
            "Sample_ID": base["Sample_ID"],
            "Easting": base["Easting"],
            "Northing": base["Northing"],
            "Gold_Grade": gold,
            "Copper_pct": copper,
            "Iron_pct": iron,
            "Density_gcc": density,
        }
    )


DEMO_DATASETS = {
    "Synthetic Spatial Gold": {
        "description": "200 synthetic spatial samples with Easting, Northing and Gold_Grade. Ideal for maps, variograms, IDW and Kriging.",
        "generator": synthetic_spatial_gold,
        "source": "Synthetic — generated locally by GeoStatix",
    },
    "Synthetic Skewed Grade": {
        "description": "500 synthetic positively skewed grade observations. Ideal for histograms, distribution fits, Q-Q plots and probability plots.",
        "generator": synthetic_skewed_grade,
        "source": "Synthetic — generated locally by GeoStatix",
    },
    "Synthetic Multivariate Geology": {
        "description": "300 synthetic samples with depth, lithology, Cu, Fe, density and moisture variables. Ideal for profiling and relationship analysis.",
        "generator": synthetic_multivariate_geology,
        "source": "Synthetic — generated locally by GeoStatix",
    },
    "Synthetic Spatial + Multivariate": {
        "description": "220 synthetic spatial samples with multiple correlated grade/physical variables. Ideal for spatial and multivariate workflows.",
        "generator": synthetic_spatial_multivariate,
        "source": "Synthetic — generated locally by GeoStatix",
    },
}


def load_demo_dataset(name):
    """Generate one of the built-in demo datasets by name."""
    return DEMO_DATASETS[name]["generator"]()
