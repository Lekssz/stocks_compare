# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
import yfinance as yf

COLORS = {
    "ARIMA": "#2D6CDF",
    "Prophet": "#16A34A",
    "LSTM": "#9333EA",
    "Naive-last": "#64748B",
    "Naive-drift": "#F59E0B",
}
# Optional / heavy deps guarded by try/except
HAVE_ARIMA_PM = False
HAVE_ARIMA_SM = False
try:
    import pmdarima as pm
    HAVE_ARIMA_PM = True
except Exception:
    pass
try:
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    HAVE_ARIMA_SM = True
except Exception:
    pass

# Prophet (will auto-download CmdStan on first run if needed)
try:
    os.environ.setdefault("CMDSTANPY_FORCE_DOWNLOAD", "1")
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

# LSTM (TensorFlow)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAVE_TF = True
except Exception:
    HAVE_TF = False

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stocks: ARIMA vs Prophet vs LSTM", layout="wide")

# ============================================================
# Utilities
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def _normalize_ds_y(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a clean ['ds','y'] frame (ds tz-naive datetime, y float)."""
    if not {"ds", "y"}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'ds' and 'y' columns.")
    out = df[["ds", "y"]].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna(subset=["ds"])
    out["y"] = pd.to_numeric(out["y"], errors="coerce").astype("float64")
    out = out.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
    try:
        out["ds"] = out["ds"].dt.tz_localize(None)
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False)
def load_series(ticker: str, period_years: int = 5) -> pd.DataFrame:
    """Download daily prices and return normalized ['ds','y']."""
    df = yf.download(
        ticker, period=f"{period_years}y", interval="1d",
        auto_adjust=True, progress=False
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    price_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if price_col is None:
        return pd.DataFrame(columns=["ds", "y"])
    out = df.dropna().rename(columns={price_col: "y"}).copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out["ds"] = out.index
    try:
        out["ds"] = out["ds"].dt.tz_localize(None)
    except Exception:
        pass
    out = out[["ds", "y"]].reset_index(drop=True)
    return _normalize_ds_y(out)

# ============================================================
# Models
# ============================================================

def prophet_forecast_on_dates(train_df: pd.DataFrame, target_dates) -> pd.DataFrame:
    if not HAVE_PROPHET:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    df = _normalize_ds_y(train_df)
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.2,
    )
    m.fit(df[["ds", "y"]])

    future = pd.DataFrame({"ds": pd.to_datetime(pd.Index(target_dates), errors="coerce")}).dropna()
    try:
        future["ds"] = future["ds"].dt.tz_localize(None)
    except Exception:
        pass

    preds = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return preds

def arima_forecast(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = _normalize_ds_y(train_df).sort_values("ds").reset_index(drop=True)
    y = df.set_index("ds")["y"]  # DatetimeIndex
    if HAVE_ARIMA_PM:
        model = pm.auto_arima(
            y, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action="ignore"
        )
        preds, conf = model.predict(n_periods=horizon, return_conf_int=True)
        start = y.index[-1] + pd.offsets.BDay(1)
        idx = pd.bdate_range(start=start, periods=horizon)
        return pd.DataFrame({"ds": idx, "yhat": preds, "yhat_lower": conf[:, 0], "yhat_upper": conf[:, 1]})
    elif HAVE_ARIMA_SM:
        res = SM_ARIMA(y, order=(1, 1, 1)).fit()
        pred = res.get_forecast(steps=horizon)
        mean = pred.predicted_mean
        conf = pred.conf_int()
        start = y.index[-1] + pd.offsets.BDay(1)
        idx = pd.bdate_range(start=start, periods=horizon)
        return pd.DataFrame({
            "ds": idx,
            "yhat": mean.values,
            "yhat_lower": conf.iloc[:, 0].values,
            "yhat_upper": conf.iloc[:, 1].values,
        })
    else:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

def lstm_forecast(train_df: pd.DataFrame, horizon: int, lookback=60, epochs=20, batch=32, seed=42) -> pd.DataFrame:
    if not HAVE_TF:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    df = _normalize_ds_y(train_df).sort_values("ds").reset_index(drop=True)
    vals = df["y"].values.astype("float32").reshape(-1, 1)

    scaler = MinMaxScaler()
    vals_sc = scaler.fit_transform(vals)  # fit on train only

    X, y = [], []
    for i in range(lookback, len(vals_sc)):
        X.append(vals_sc[i - lookback:i, 0])
        y.append(vals_sc[i, 0])
    if len(X) == 0:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    X = np.array(X)[..., None]
    y = np.array(y)

    tf.keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch, verbose=0, validation_split=0.1, callbacks=[cb])

    # recursive forecast
    history = vals_sc.flatten().tolist()
    preds_sc = []
    for _ in range(horizon):
        x = np.array(history[-lookback:], dtype="float32")[None, :, None]
        pred = model.predict(x, verbose=0).ravel()[0]
        preds_sc.append(float(pred))
        history.append(float(pred))

    preds = scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).ravel()

    last_ts = pd.to_datetime(df["ds"].iloc[-1])
    start = last_ts + pd.offsets.BDay(1)
    idx = pd.bdate_range(start=start, periods=horizon)
    return pd.DataFrame({"ds": idx, "yhat": preds, "yhat_lower": np.nan, "yhat_upper": np.nan})

def naive_forecast_on_dates(train_df, target_dates, method="last"):
    df = _normalize_ds_y(train_df).sort_values("ds")
    td = pd.to_datetime(pd.Index(target_dates), errors="coerce").dropna()
    try:
        td = td.tz_localize(None)
    except Exception:
        pass
    n = len(td)
    if n == 0:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    last = df["y"].iloc[-1]
    if method == "last":
        yhat = np.full(n, last, float)
    else:  # drift
        y0, k = df["y"].iloc[0], max(1, len(df) - 1)
        drift = (last - y0) / k
        yhat = last + drift * np.arange(1, n + 1)
    return pd.DataFrame({"ds": td, "yhat": yhat, "yhat_lower": np.nan, "yhat_upper": np.nan})

# ============================================================
# Evaluation (with NaN protection + inner joins)
# ============================================================

def evaluate_models(
    df_full, horizon, test_days=60,
    use_arima=True, use_prophet=True, use_lstm=True,
    lookback=60, epochs=20, batch=32
):
    df_xy = _normalize_ds_y(df_full).sort_values("ds").reset_index(drop=True)
    if len(df_xy) <= test_days + 5:
        raise ValueError("Not enough data to create the requested test window.")

    train_df = df_xy.iloc[:-test_days].copy()
    test_df  = df_xy.iloc[-test_days:].copy()

    target_dates = pd.to_datetime(test_df["ds"].iloc[:horizon], errors="coerce").dropna()
    try:
        target_dates = target_dates.tz_localize(None)
    except Exception:
        pass

    outputs = {}

    # Prophet (already on exact target dates)
    if use_prophet and HAVE_PROPHET:
        try:
            outputs["Prophet"] = prophet_forecast_on_dates(train_df, target_dates)
        except Exception as e:
            st.warning(f"Prophet failed: {e}")
    elif use_prophet and not HAVE_PROPHET:
        st.info("Prophet not available in this environment.")

    # ARIMA → forecast then coerce to exact target dates via INNER join
    if use_arima and (HAVE_ARIMA_PM or HAVE_ARIMA_SM):
        try:
            ar = arima_forecast(train_df, horizon)
            ar = pd.DataFrame({"ds": target_dates}).merge(ar[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="inner")
            outputs["ARIMA"] = ar
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")
    elif use_arima and not (HAVE_ARIMA_PM or HAVE_ARIMA_SM):
        st.info("ARIMA not available; disabled.")

    # LSTM → forecast then INNER join to target dates
    if use_lstm and HAVE_TF:
        try:
            ls = lstm_forecast(train_df, horizon, lookback=lookback, epochs=epochs, batch=batch)
            ls = pd.DataFrame({"ds": target_dates}).merge(ls[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="inner")
            outputs["LSTM"] = ls
        except Exception as e:
            st.warning(f"LSTM failed: {e}")
    elif use_lstm and not HAVE_TF:
        st.info("TensorFlow not available; LSTM disabled.")

    # Naive baselines always on target dates
    outputs["Naive-last"]  = naive_forecast_on_dates(train_df, target_dates, "last")
    outputs["Naive-drift"] = naive_forecast_on_dates(train_df, target_dates, "drift")

    # Score with NaN protection
    scores, aligned = {}, {}
    for name, fc in outputs.items():
        try:
            merged = (
                fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                .merge(test_df[["ds", "y"]].iloc[:horizon], on="ds", how="inner")
                .sort_values("ds").reset_index(drop=True)
            )
            if merged.empty:
                st.warning(f"{name}: no overlapping dates with test window.")
                continue

            before = len(merged)
            merged = merged.dropna(subset=["yhat", "y"])
            if len(merged) < before:
                st.info(f"{name}: dropped {before - len(merged)} NaN rows after alignment.")
            if merged.empty:
                st.warning(f"{name}: no valid rows after dropping NaNs.")
                continue

            scores[name] = {
                "rmse": rmse(merged["y"], merged["yhat"]),
                "mae":  mae(merged["y"], merged["yhat"]),
            }
            aligned[name] = merged
        except Exception as e:
            st.warning(f"{name}: scoring skipped — {e}")

    return scores, aligned, test_df

# ============================================================
# Plot
# ============================================================

def plot_forecasts_multi(history_df, fcsts_dict, title="Forecast comparison"):
    import pandas as pd
    import plotly.graph_objects as go

    fig = go.Figure()

    # ---- Actuals (history) ----
    hist = history_df.sort_values("ds").copy()

    # pick the y column (prefer 'y', fall back to common price names)
    target_col = "y"
    if target_col not in hist.columns:
        for c in ["adj_close", "Adj Close", "close", "Close", "y_true"]:
            if c in hist.columns:
                target_col = c
                break
        else:
            raise ValueError("history_df must have 'y' or a price column like 'Adj Close'.")

    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist[target_col],
        name="Actual", mode="lines", line=dict(width=2)
    ))

    # ---- Forecasts ----
    for name, fdf in fcsts_dict.items():
        if fdf is None or fdf.empty:
            continue

        f = fdf.sort_values("ds")
        if not {"ds", "yhat"}.issubset(f.columns):
            raise ValueError(f"Forecast '{name}' must include 'ds' and 'yhat' columns.")

        line_style = {"width": 2}
        # optional color mapping if you have a COLORS dict
        if "COLORS" in globals():
            try:
                color = COLORS.get(name)
                if color:
                    line_style["color"] = color
            except Exception:
                pass

        fig.add_trace(go.Scatter(
            x=f["ds"], y=f["yhat"],
            name=f"{name} forecast", mode="lines", line=line_style
        ))

        # Confidence interval band (optional)
        if {"yhat_lower", "yhat_upper"}.issubset(f.columns):
            if f["yhat_lower"].notna().any() and f["yhat_upper"].notna().any():
                band_x = pd.concat([f["ds"], f["ds"][::-1]])
                band_y = pd.concat([f["yhat_upper"], f["yhat_lower"][::-1]])
                fig.add_trace(go.Scatter(
                    x=band_x, y=band_y,
                    fill="toself", name=f"{name} CI",
                    line=dict(width=0), opacity=0.15, showlegend=True
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Adj Close",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


# ============================================================
# UI
# ============================================================

st.title("Stocks: ARIMA vs Prophet vs LSTM (7/30-day)")

with st.sidebar:
    st.header("Controls")
    tickers = st.multiselect(
        "Tickers",
        ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA"],
        default=["AAPL", "MSFT", "TSLA"]
    )
    horizon = st.selectbox("Forecast horizon (days)", [7, 30], index=0)
    test_days = st.slider("Test window (days)", 30, 120, 60, step=5)

    use_arima   = st.checkbox("Use ARIMA", (HAVE_ARIMA_PM or HAVE_ARIMA_SM), disabled=not (HAVE_ARIMA_PM or HAVE_ARIMA_SM))
    use_prophet = st.checkbox("Use Prophet", False and HAVE_PROPHET, disabled=not HAVE_PROPHET)  # default off (first run)
    use_lstm    = st.checkbox("Use LSTM", HAVE_TF, disabled=not HAVE_TF)

    lookback = st.slider("LSTM lookback", 30, 120, 60, step=5, disabled=not use_lstm)
    epochs   = st.slider("LSTM epochs", 5, 50, 20, step=1, disabled=not use_lstm)
    batch    = st.selectbox("LSTM batch size", [16, 32, 64, 128], index=1, disabled=not use_lstm)

run_btn = st.button("Run")

if run_btn:
    results_rows = []
    for t in tickers:
        st.subheader(t)
        df = load_series(t)
        if df.empty:
            st.warning(f"{t}: no data returned.")
            continue

        try:
            scores, aligned, test_df = evaluate_models(
                df, horizon, test_days=test_days,
                use_arima=use_arima, use_prophet=use_prophet, use_lstm=use_lstm,
                lookback=lookback, epochs=epochs, batch=batch
            )
        except Exception as e:
            st.error(f"{t}: evaluation failed — {e}")
            continue

        # Scores table per ticker
        if scores:
            sc_df = (
                pd.DataFrame([{"ticker": t, "model": k, **v} for k, v in scores.items()])
                .sort_values(["rmse", "mae"])
                .reset_index(drop=True)
            )
            st.dataframe(sc_df, use_container_width=True)

            # add to combined
            for k, v in scores.items():
                results_rows.append({"ticker": t, "horizon": horizon, "model": k, "rmse": v["rmse"], "mae": v["mae"]})
        else:
            st.info("No model produced scores for this ticker.")

        # Plot overlay (use actuals from test_df first `horizon` rows)
        fig = plot_forecasts_multi(test_df.iloc[:horizon], aligned, title=f"{t} • {horizon}-day forecasts")
        st.plotly_chart(fig, use_container_width=True)

    # Combined results + winners + downloads
    if results_rows:
        combined = pd.DataFrame(results_rows).sort_values(["ticker", "horizon", "rmse"]).reset_index(drop=True)
        st.markdown("### Combined Results")
        st.dataframe(combined, use_container_width=True)

        winners = combined.loc[combined.groupby(["ticker", "horizon"])["rmse"].idxmin()].sort_values(["ticker", "horizon"])
        st.markdown("### Winners by ticker × horizon")
        st.dataframe(winners[["ticker", "horizon", "model", "rmse", "mae"]], use_container_width=True)

        st.download_button("Download combined CSV", combined.to_csv(index=False).encode(), "combined_scores.csv", "text/csv")
        st.download_button("Download winners CSV", winners.to_csv(index=False).encode(), "winners.csv", "text/csv")
    else:
        st.info("No results to show. Check your options and try again.")
else:
    st.info("Select your options on the left, then click **Run**.")
with st.expander("About this app"):
    st.markdown(
        """
        **Author:** GBEMILEKE MICAH   
        **Purpose:** Compare quick, short-horizon forecasts (7/30 days) with ARIMA/Prophet/LSTM vs simple baselines.  
        **Notes:** First Prophet run may download CmdStan; if slow, start with ARIMA/LSTM.  
        **Caveats:** This is a lightweight demo—no walk-forward CV, and LSTM is intentionally small to keep things fast.
        """
    )
