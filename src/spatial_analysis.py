import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, cdist


def _clean_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def _normalize_column_name(column):
    """Normalize a column name for robust, case-insensitive matching."""
    text = str(column).strip().lower()
    for char in [" ", "-", "/", "\\", ".", "(", ")", "[", "]"]:
        text = text.replace(char, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def generate_demo_spatial_dataset(n_samples=200, seed=42):
    """Generate a reproducible synthetic spatial dataset for testing GeoStatix.

    The demo intentionally uses generic coordinate/value names (Easting,
    Northing, Gold_Grade) because it is designed as a ready-to-run example,
    not as a hard-coded assumption about uploaded datasets.
    """
    n_samples = int(max(9, n_samples))
    rng = np.random.default_rng(seed)

    side = int(np.ceil(np.sqrt(n_samples)))
    easting_grid, northing_grid = np.meshgrid(
        np.linspace(0, 1000, side),
        np.linspace(0, 1000, side),
    )

    easting = easting_grid.ravel()[:n_samples]
    northing = northing_grid.ravel()[:n_samples]

    grade = (
        1.5
        + 0.9 * np.exp(-(((easting - 320) / 260) ** 2 + ((northing - 650) / 220) ** 2))
        + 0.6 * np.exp(-(((easting - 760) / 190) ** 2 + ((northing - 330) / 300) ** 2))
        + 0.18 * np.sin(easting / 115)
        + 0.12 * np.cos(northing / 145)
        + rng.normal(0, 0.06, n_samples)
    )

    grade = np.clip(grade, 0.01, None)

    return pd.DataFrame(
        {
            "Sample_ID": [f"DEMO-{i:03d}" for i in range(1, n_samples + 1)],
            "Easting": np.round(easting, 3),
            "Northing": np.round(northing, 3),
            "Gold_Grade": np.round(grade, 5),
        }
    )


def _coordinate_score(column, axis):
    """Score a column name for a likely X/Y spatial-coordinate role."""
    name = _normalize_column_name(column)
    tokens = set(name.split("_"))

    if any(token in tokens for token in {"id", "index", "row", "unnamed"}):
        return -100

    exact_x = {"x", "easting", "east", "longitude", "lon", "lng", "utm_x", "coord_x", "x_coord", "x_coordinate"}
    exact_y = {"y", "northing", "north", "latitude", "lat", "utm_y", "coord_y", "y_coord", "y_coordinate"}
    axis_words = {"x": exact_x, "y": exact_y}[axis]

    if name in axis_words:
        return 100

    score = 0
    joined = name
    if axis == "x":
        if any(word in joined for word in ["easting", "longitude", "utm_x", "coord_x", "x_coord"]):
            score += 75
        if "east" in tokens or "lon" in tokens or "lng" in tokens:
            score += 55
    else:
        if any(word in joined for word in ["northing", "latitude", "utm_y", "coord_y", "y_coord"]):
            score += 75
        if "north" in tokens or "lat" in tokens:
            score += 55

    if name.startswith(axis + "_") or name.endswith("_" + axis):
        score += 25

    return score


def _looks_like_identifier(series, column):
    """Detect columns that are likely IDs/indexes rather than measurements."""
    name = _normalize_column_name(column)
    if any(token in name.split("_") for token in ["id", "index", "row", "unnamed"]):
        return True

    numeric = _clean_numeric_series(series).dropna()
    if len(numeric) < 5:
        return False

    unique_ratio = numeric.nunique() / len(numeric)
    if unique_ratio > 0.98:
        values = numeric.to_numpy(dtype=float)
        if np.allclose(np.diff(np.sort(values)), 1.0):
            return True

    return False


def detect_coordinate_columns(df):
    """Detect likely spatial coordinates without assuming the first numeric columns.

    Returns a dictionary with x/y suggestions, detection mode, confidence and
    candidate lists. If no coordinate names are recognizable, x and y are None
    so GeoStatix does not silently treat a grade/depth/index column as a map axis.
    """
    numeric = df.select_dtypes(include="number").columns.tolist()
    if len(numeric) < 2:
        return None

    x_ranked = sorted(
        [(c, _coordinate_score(c, "x")) for c in numeric],
        key=lambda item: item[1],
        reverse=True,
    )
    y_ranked = sorted(
        [(c, _coordinate_score(c, "y")) for c in numeric],
        key=lambda item: item[1],
        reverse=True,
    )

    x_col, x_score = x_ranked[0]
    y_candidates = [(c, score) for c, score in y_ranked if c != x_col]
    y_col, y_score = y_candidates[0]

    geographic = (
        _normalize_column_name(x_col) in {"lon", "lng", "longitude", "x_lon"}
        and _normalize_column_name(y_col) in {"lat", "latitude", "y_lat"}
    )

    if x_score >= 55 and y_score >= 55:
        mode = "geographic" if geographic else "projected"
        confidence = "high" if x_score >= 75 and y_score >= 75 else "medium"
        return {
            "x": x_col,
            "y": y_col,
            "mode": mode,
            "confidence": confidence,
            "x_score": x_score,
            "y_score": y_score,
            "x_candidates": [c for c, s in x_ranked if s > 0],
            "y_candidates": [c for c, s in y_ranked if s > 0],
            "reason": "Coordinate-like column names were detected.",
        }

    return {
        "x": None,
        "y": None,
        "mode": "unknown",
        "confidence": "low",
        "x_score": max(0, x_score),
        "y_score": max(0, y_score),
        "x_candidates": [c for c, s in x_ranked if s > 0],
        "y_candidates": [c for c, s in y_ranked if s > 0],
        "reason": "No sufficiently strong coordinate naming pattern was detected.",
    }


def suggest_value_column(df, x_column=None, y_column=None):
    """Suggest a likely measured/value variable while avoiding coordinate/ID columns."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    excluded = {c for c in [x_column, y_column] if c is not None}
    candidates = [c for c in numeric if c not in excluded]
    if not candidates:
        return None

    value_words = {
        "value", "grade", "assay", "concentration", "content", "amount",
        "ppm", "ppb", "percent", "pct", "porosity", "permeability",
        "thickness", "density", "temperature", "pressure", "elevation",
        "depth", "hardness", "moisture", "salinity", "conductivity",
    }

    scored = []
    for col in candidates:
        name = _normalize_column_name(col)
        score = 0
        if _looks_like_identifier(df[col], col):
            score -= 100
        if "unnamed" in name:
            score -= 80
        if name in {"value", "measurement", "measure", "response"}:
            score += 80
        if any(word in name.split("_") for word in value_words):
            score += 60
        if any(word in name for word in value_words):
            score += 25
        if df[col].nunique(dropna=True) > 1:
            score += 10
        scored.append((col, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    best_col, best_score = scored[0]

    if best_score <= 0:
        return None
    return best_col


def prepare_spatial_data(df, x_column, y_column, value_column=None):
    """Prepare clean 1-D coordinate/value columns for spatial analysis."""
    if x_column is None or y_column is None or x_column == y_column:
        return None

    if value_column is not None and value_column in {x_column, y_column}:
        return None

    columns = [x_column, y_column]
    if value_column is not None:
        columns.append(value_column)

    if any(col not in df.columns for col in columns):
        return None

    data = df.loc[:, columns].copy()
    for col in columns:
        data[col] = _clean_numeric_series(data[col])

    data = data.dropna(subset=[x_column, y_column])
    if value_column is not None:
        data = data.dropna(subset=[value_column])

    if len(data) < 3:
        return None

    data = data.drop_duplicates(subset=[x_column, y_column], keep="first")
    if len(data) < 3:
        return None

    return data.reset_index(drop=True)


def create_spatial_map(df, x_column, y_column, value_column=None, geographic=False):
    """Create a spatial sample/value map."""
    data = prepare_spatial_data(df, x_column, y_column, value_column)
    if data is None:
        return None

    if value_column:
        fig = go.Figure(
            go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode="markers",
                marker=dict(
                    size=9,
                    color=data[value_column],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=value_column),
                ),
                text=[f"{value_column}: {v:.5g}" for v in data[value_column]],
                hovertemplate=(
                    f"{x_column}: %{{x}}<br>"
                    f"{y_column}: %{{y}}<br>"
                    "%{text}<extra></extra>"
                ),
                name="Samples",
            )
        )
        title = f"Spatial Value Map — {value_column}"
    else:
        fig = go.Figure(
            go.Scatter(
                x=data[x_column],
                y=data[y_column],
                mode="markers",
                marker=dict(size=9),
                name="Samples",
            )
        )
        title = "Spatial Sample Map"

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_dark",
    )
    return fig


def compute_variogram_cloud(df, x_column, y_column, value_column, max_pairs=50000):
    """Calculate pairwise distance and semivariance for a variogram cloud."""
    data = prepare_spatial_data(df, x_column, y_column, value_column)
    if data is None or len(data) < 3:
        return None

    coords = data[[x_column, y_column]].to_numpy(dtype=float)
    values = data[value_column].to_numpy(dtype=float)
    n = len(data)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        distances = pdist(coords)
        semivariances = 0.5 * pdist(values.reshape(-1, 1), metric="euclidean") ** 2
    else:
        rng = np.random.default_rng(42)
        target_pairs = min(int(max_pairs), total_pairs)
        pair_ids = rng.choice(total_pairs, size=target_pairs, replace=False)
        i = (
            n - 2
            - np.floor(
                np.sqrt(-8 * pair_ids + 4 * n * (n - 1) - 7) / 2.0 - 0.5
            ).astype(np.int64)
        )
        j = (
            pair_ids
            + i
            + 1
            - total_pairs
            + (n - i) * (n - i - 1) // 2
        ).astype(np.int64)
        dx = coords[i] - coords[j]
        distances = np.sqrt(np.sum(dx * dx, axis=1))
        semivariances = 0.5 * (values[i] - values[j]) ** 2

    cloud = pd.DataFrame({"Distance": distances, "Semivariance": semivariances})
    cloud = cloud.replace([np.inf, -np.inf], np.nan).dropna()
    cloud = cloud[cloud["Distance"] > 0]
    return cloud.sort_values("Distance").reset_index(drop=True)


def calculate_experimental_variogram(cloud, n_lags=12, max_distance=None):
    """Bin variogram-cloud pairs into lag classes."""
    if cloud is None or cloud.empty:
        return None

    working = cloud.copy()
    if max_distance is None:
        max_distance = float(working["Distance"].quantile(0.75))
    if max_distance <= 0:
        return None

    working = working[working["Distance"] <= max_distance].copy()
    if working.empty:
        return None

    edges = np.linspace(0, max_distance, int(n_lags) + 1)
    working["Lag"] = pd.cut(
        working["Distance"], bins=edges, include_lowest=True, labels=False
    )

    result = (
        working.groupby("Lag", observed=True)
        .agg(
            Lag_Distance=("Distance", "mean"),
            Semivariance=("Semivariance", "mean"),
            Pairs=("Semivariance", "size"),
        )
        .reset_index(drop=True)
    )
    return result[result["Pairs"] > 0].reset_index(drop=True)


def spherical_model(h, nugget, sill, rang):
    h = np.asarray(h, dtype=float)
    hr = h / max(rang, 1e-12)
    return np.where(h <= rang, nugget + sill * (1.5 * hr - 0.5 * hr ** 3), nugget + sill)


def exponential_model(h, nugget, sill, rang):
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1.0 - np.exp(-h / max(rang, 1e-12)))


def gaussian_model(h, nugget, sill, rang):
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1.0 - np.exp(-(h / max(rang, 1e-12)) ** 2))


def fit_variogram_model(experimental, model_name):
    """Fit a spherical, exponential, or Gaussian variogram model."""
    if experimental is None or len(experimental) < 3:
        return None

    x = experimental["Lag_Distance"].to_numpy(dtype=float)
    y = experimental["Semivariance"].to_numpy(dtype=float)
    if np.nanmax(y) <= 0 or np.nanmax(x) <= 0:
        return None

    model_map = {"Spherical": spherical_model, "Exponential": exponential_model, "Gaussian": gaussian_model}
    model = model_map.get(model_name)
    if model is None:
        return None

    nugget0 = max(0.0, float(np.nanmin(y)))
    sill0 = max(float(np.nanmax(y) - nugget0), np.finfo(float).eps)
    range0 = max(float(np.nanmedian(x)), np.finfo(float).eps)

    try:
        params, _ = curve_fit(
            model,
            x,
            y,
            p0=[nugget0, sill0, range0],
            bounds=([0.0, 0.0, np.finfo(float).eps], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
    except Exception:
        return None

    predictions = model(x, *params)
    residuals = y - predictions
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "model": model,
        "model_name": model_name,
        "nugget": float(params[0]),
        "sill": float(params[1]),
        "range": float(params[2]),
        "rmse": rmse,
    }


def create_variogram_plot(experimental, fitted=None, show_cloud=None):
    """Create an experimental variogram plot with an optional fitted model."""
    if experimental is None or experimental.empty:
        return None

    fig = go.Figure()
    if show_cloud is not None and not show_cloud.empty:
        cloud_to_plot = show_cloud
        if len(cloud_to_plot) > 5000:
            cloud_to_plot = cloud_to_plot.sample(5000, random_state=42)
        fig.add_trace(
            go.Scattergl(
                x=cloud_to_plot["Distance"],
                y=cloud_to_plot["Semivariance"],
                mode="markers",
                marker=dict(size=3, opacity=0.25),
                name="Variogram Cloud",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=experimental["Lag_Distance"],
            y=experimental["Semivariance"],
            mode="markers+lines",
            marker=dict(size=8),
            name="Experimental Variogram",
        )
    )

    if fitted is not None:
        h_max = float(experimental["Lag_Distance"].max())
        h = np.linspace(0, h_max, 300)
        gamma = fitted["model"](h, fitted["nugget"], fitted["sill"], fitted["range"])
        fig.add_trace(go.Scatter(x=h, y=gamma, mode="lines", name=fitted["model_name"] + " Model"))

    fig.update_layout(
        title="Experimental Variogram",
        xaxis_title="Lag Distance",
        yaxis_title="Semivariance γ(h)",
        template="plotly_dark",
    )
    return fig


def spatial_quality_report(df, x_column, y_column, value_column=None):
    """Return compact diagnostics for spatial-analysis readiness."""
    if x_column is None or y_column is None or x_column == y_column:
        return None
    columns = [x_column, y_column] + ([value_column] if value_column is not None else [])
    data = df.loc[:, columns].copy()
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[x_column, y_column])
    if value_column is not None:
        data = data.dropna(subset=[value_column])
    if data.empty:
        return None
    coords = data[[x_column, y_column]].to_numpy(dtype=float)
    duplicate_mask = pd.DataFrame(coords).duplicated(keep=False).to_numpy()
    report = {
        "valid_coordinate_rows": int(len(data)),
        "unique_locations": int(len(np.unique(coords, axis=0))),
        "duplicate_location_rows": int(duplicate_mask.sum()),
        "x_range": float(np.ptp(data[x_column].to_numpy(dtype=float))),
        "y_range": float(np.ptp(data[y_column].to_numpy(dtype=float))),
    }
    if value_column is not None:
        values = data[value_column].to_numpy(dtype=float)
        report["value_range"] = float(np.ptp(values))
        report["value_constant"] = bool(np.allclose(values, values[0]))
    return report

def idw_interpolate(df, x_column, y_column, value_column, grid_x, grid_y, power=2.0, max_neighbors=None):
    """Interpolate values on a grid using inverse-distance weighting."""
    data = prepare_spatial_data(df, x_column, y_column, value_column)
    if data is None:
        return None

    coords = data[[x_column, y_column]].to_numpy(dtype=float)
    values = data[value_column].to_numpy(dtype=float)
    query = np.column_stack([np.asarray(grid_x).ravel(), np.asarray(grid_y).ravel()])
    distances = cdist(query, coords)

    if max_neighbors is not None and 1 <= max_neighbors < len(coords):
        neighbor_idx = np.argpartition(distances, max_neighbors - 1, axis=1)[:, :max_neighbors]
        row_idx = np.arange(len(query))[:, None]
        local_d = distances[row_idx, neighbor_idx]
        local_v = values[neighbor_idx]
    else:
        local_d = distances
        local_v = np.broadcast_to(values, distances.shape)

    zero = local_d <= 1e-12
    weights = 1.0 / np.maximum(local_d, 1e-12) ** float(power)
    weighted = np.sum(weights * local_v, axis=1)
    weight_sum = np.sum(weights, axis=1)
    prediction = weighted / np.maximum(weight_sum, 1e-12)

    if np.any(zero):
        exact = np.argmax(zero, axis=1)
        has_exact = zero.any(axis=1)
        prediction[has_exact] = local_v[np.arange(len(query))[has_exact], exact[has_exact]]

    return prediction.reshape(np.asarray(grid_x).shape)


def _variogram_callable(fitted):
    if fitted is None:
        return None
    return lambda h: fitted["model"](h, fitted["nugget"], fitted["sill"], fitted["range"])


def ordinary_kriging_predict(
    df,
    x_column,
    y_column,
    value_column,
    grid_x,
    grid_y,
    fitted_variogram,
    max_neighbors=40,
):
    """Perform local ordinary kriging using a fitted variogram model.

    A local neighborhood is used to keep computation practical for interactive
    Streamlit datasets. Returns prediction and kriging-variance arrays.
    """
    data = prepare_spatial_data(df, x_column, y_column, value_column)
    if data is None or fitted_variogram is None:
        return None

    coords = data[[x_column, y_column]].to_numpy(dtype=float)
    values = data[value_column].to_numpy(dtype=float)
    query = np.column_stack([np.asarray(grid_x).ravel(), np.asarray(grid_y).ravel()])
    variogram = _variogram_callable(fitted_variogram)

    n = len(coords)
    k = min(max(3, int(max_neighbors)), n)
    predictions = np.full(len(query), np.nan, dtype=float)
    variances = np.full(len(query), np.nan, dtype=float)

    for q_idx, point in enumerate(query):
        distances_to_samples = np.sqrt(np.sum((coords - point) ** 2, axis=1))
        if k < n:
            idx = np.argpartition(distances_to_samples, k - 1)[:k]
        else:
            idx = np.arange(n)

        local_coords = coords[idx]
        local_values = values[idx]
        local_dist = cdist(local_coords, local_coords)
        gamma_matrix = variogram(local_dist)
        np.fill_diagonal(gamma_matrix, 0.0)
        gamma_vector = variogram(np.sqrt(np.sum((local_coords - point) ** 2, axis=1)))

        system = np.zeros((len(idx) + 1, len(idx) + 1), dtype=float)
        system[:-1, :-1] = gamma_matrix
        system[:-1, -1] = 1.0
        system[-1, :-1] = 1.0
        rhs = np.r_[gamma_vector, 1.0]

        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]

        weights = solution[:-1]
        multiplier = solution[-1]
        predictions[q_idx] = float(np.dot(weights, local_values))
        variances[q_idx] = max(0.0, float(np.dot(weights, gamma_vector) + multiplier))

    return {
        "prediction": predictions.reshape(np.asarray(grid_x).shape),
        "variance": variances.reshape(np.asarray(grid_x).shape),
    }


def cross_validate_idw(
    df,
    x_column,
    y_column,
    value_column,
    power=2.0,
    max_neighbors=None,
    max_validation_points=None,
    random_state=42,
):
    """Leave-one-out spatial validation for IDW.

    Each validation point is removed from the training set before its value is
    predicted. If max_validation_points is set below the sample count, a
    reproducible subset of points is validated instead of the full dataset.
    """
    data = prepare_spatial_data(df, x_column, y_column, value_column)
    if data is None or len(data) < 5:
        return None

    if max_validation_points is None or max_validation_points >= len(data):
        validation_idx = np.arange(len(data))
    else:
        rng = np.random.default_rng(random_state)
        validation_idx = np.sort(rng.choice(len(data), size=int(max_validation_points), replace=False))

    coords = data[[x_column, y_column]].to_numpy(dtype=float)
    values = data[value_column].to_numpy(dtype=float)
    rows = []

    for idx in validation_idx:
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[idx] = False
        train = data.iloc[np.flatnonzero(train_mask)].reset_index(drop=True)
        point_x = np.array([[coords[idx, 0]]])
        point_y = np.array([[coords[idx, 1]]])
        prediction = idw_interpolate(
            train,
            x_column,
            y_column,
            value_column,
            point_x,
            point_y,
            power=power,
            max_neighbors=max_neighbors,
        )
        predicted = float(prediction.ravel()[0]) if prediction is not None else np.nan
        observed = float(values[idx])
        rows.append({
            "Validation Index": int(idx),
            "X": float(coords[idx, 0]),
            "Y": float(coords[idx, 1]),
            "Observed": observed,
            "Predicted": predicted,
            "Residual": observed - predicted if np.isfinite(predicted) else np.nan,
        })

    return pd.DataFrame(rows)


def cross_validate_kriging(
    df,
    x_column,
    y_column,
    value_column,
    fitted_variogram,
    max_neighbors=20,
    max_validation_points=None,
    random_state=42,
):
    """Leave-one-out spatial validation for ordinary kriging.

    The variogram is fitted once from the supplied dataset and then held fixed
    while each validation point is predicted from the remaining samples.
    """
    data = prepare_spatial_data(df, x_column, y_column, value_column)
    if data is None or fitted_variogram is None or len(data) < 5:
        return None

    if max_validation_points is None or max_validation_points >= len(data):
        validation_idx = np.arange(len(data))
    else:
        rng = np.random.default_rng(random_state)
        validation_idx = np.sort(rng.choice(len(data), size=int(max_validation_points), replace=False))

    coords = data[[x_column, y_column]].to_numpy(dtype=float)
    values = data[value_column].to_numpy(dtype=float)
    rows = []

    for idx in validation_idx:
        train = data.drop(index=idx).reset_index(drop=True)
        point_x = np.array([[coords[idx, 0]]])
        point_y = np.array([[coords[idx, 1]]])
        result = ordinary_kriging_predict(
            train,
            x_column,
            y_column,
            value_column,
            point_x,
            point_y,
            fitted_variogram,
            max_neighbors=max_neighbors,
        )
        if result is None:
            predicted = np.nan
            variance = np.nan
        else:
            predicted = float(result["prediction"].ravel()[0])
            variance = float(result["variance"].ravel()[0])

        observed = float(values[idx])
        rows.append({
            "Validation Index": int(idx),
            "X": float(coords[idx, 0]),
            "Y": float(coords[idx, 1]),
            "Observed": observed,
            "Predicted": predicted,
            "Residual": observed - predicted if np.isfinite(predicted) else np.nan,
            "Kriging Variance": variance,
        })

    return pd.DataFrame(rows)


def summarize_cross_validation(validation_df):
    """Calculate common interpolation validation metrics."""
    if validation_df is None or validation_df.empty:
        return None

    valid = validation_df.dropna(subset=["Observed", "Predicted"]).copy()
    if valid.empty:
        return None

    observed = valid["Observed"].to_numpy(dtype=float)
    predicted = valid["Predicted"].to_numpy(dtype=float)
    residual = observed - predicted
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mean_error = float(np.mean(residual))
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "Validation Points": int(len(valid)),
        "MAE": mae,
        "RMSE": rmse,
        "Mean Error": mean_error,
        "R²": r_squared,
    }


def create_validation_plot(validation_results):
    """Create an observed-vs-predicted validation plot.

    validation_results may be a mapping of method name to validation DataFrame,
    or a single validation DataFrame.
    """
    if isinstance(validation_results, pd.DataFrame):
        validation_results = {"Prediction": validation_results}

    fig = go.Figure()
    all_values = []
    for method_name, result in validation_results.items():
        if result is None or result.empty:
            continue
        valid = result.dropna(subset=["Observed", "Predicted"])
        if valid.empty:
            continue
        all_values.extend(valid["Observed"].tolist())
        all_values.extend(valid["Predicted"].tolist())
        fig.add_trace(
            go.Scatter(
                x=valid["Observed"],
                y=valid["Predicted"],
                mode="markers",
                name=method_name,
                text=[f"Residual: {r:.5g}" for r in valid["Residual"]],
                hovertemplate="Observed: %{x:.5g}<br>Predicted: %{y:.5g}<br>%{text}<extra></extra>",
            )
        )

    if not all_values:
        return None

    minimum = float(np.min(all_values))
    maximum = float(np.max(all_values))
    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="1:1 Reference",
        )
    )
    fig.update_layout(
        title="Observed vs Predicted — Spatial Cross-Validation",
        xaxis_title="Observed",
        yaxis_title="Predicted",
        template="plotly_dark",
    )
    return fig


def create_prediction_grid(df, x_column, y_column, grid_size=60, padding=0.02):
    """Create a regular prediction grid covering the observed coordinate extent."""
    data = prepare_spatial_data(df, x_column, y_column)
    if data is None:
        return None

    x = data[x_column].to_numpy(dtype=float)
    y = data[y_column].to_numpy(dtype=float)
    x_range = max(float(np.ptp(x)), 1e-9)
    y_range = max(float(np.ptp(y)), 1e-9)
    x_min, x_max = x.min() - padding * x_range, x.max() + padding * x_range
    y_min, y_max = y.min() - padding * y_range, y.max() + padding * y_range

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, int(grid_size)),
        np.linspace(y_min, y_max, int(grid_size)),
    )
    return grid_x, grid_y


def create_interpolation_map(grid_x, grid_y, surface, x_column, y_column, title, colorbar_title, samples=None):
    """Create an interactive interpolation/uncertainty map with optional samples."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=surface,
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title),
            hovertemplate=f"{x_column}: %{{x:.4g}}<br>{y_column}: %{{y:.4g}}<br>{colorbar_title}: %{{z:.5g}}<extra></extra>",
        )
    )

    if samples is not None and not samples.empty:
        fig.add_trace(
            go.Scatter(
                x=samples[x_column],
                y=samples[y_column],
                mode="markers",
                marker=dict(size=5, color="white", line=dict(width=1)),
                name="Samples",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_dark",
    )
    return fig
