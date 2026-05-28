import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
EXCEL_PATH = r"C:\Users\ummek\python practice\okra_veg.xlsx"
TRAIN_SHEET = "Sheet5"
TEST_SHEET = "Sheet6"
WINDOW_SIZE = 3
RANDOM_STATE = 42
VALIDATION_FRAC = 0.10
BATCH_SIZE = 8
EPOCHS = 200

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_sheet(df_raw):
    """
    - Normalize column names
    - Convert Date to datetime
    - Ensure Time column exists
    - Sort, interpolate per Subject
    - Harmonize plant-height and count columns
    - Apply isotonic regression / cummax fixes per Subject
    Returns processed df with guaranteed columns:
      - 'Plant height(mm)', 'Stem Diameter(mm)', 'No of internodes', 'No of pods'
    """
    df = df_raw.copy()
    # normalize column names
    df.columns = df.columns.str.replace('\n', ' ', regex=False).str.strip()

    # Convert Date column (format YYYYMMDD integer/string)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')

    # Ensure Time exists:
    if 'Time' not in df.columns:
        df['Time'] = 0

    # Sort and interpolate per Subject
    df = df.sort_values(['Subject', 'Date', 'Time']).reset_index(drop=True)
    df = df.groupby('Subject').apply(lambda g: g.interpolate(method='linear')).reset_index(drop=True)

    # If training sheet already has 'Plant height (mm)' keep it; else convert 'Stem Length (mm)'*25.4 when present
    if 'Plant height (mm)' not in df.columns and 'Stem Length (mm)' in df.columns:
        df['Plant height (mm)'] = df['Stem Length (mm)'] * 25.4

    # Harmonize various names to our canonical names
    cols_lower = {c.lower(): c for c in df.columns}

    def find_col(variants):
        for v in variants:
            key = v.lower()
            if key in cols_lower:
                return cols_lower[key]
        return None

    # Map potential variants
    ph_col = find_col(['Plant height (mm)', 'Plant height(mm)', 'plant height (mm)', 'plant height(mm)'])
    stem_d_col = find_col(['Stem Diameter (mm)', 'Stem Diameter(mm)', 'Stem diameter (mm)'])
    internodes_col = find_col(['Number of internodes', 'No. of Internodes', 'No of internodes', 'Number of Internodes'])
    pods_col = find_col(['Number of pods', 'No. of Fruits', 'Number of Fruits', 'No of pods'])

    # Make canonical columns (create if necessary)
    if ph_col:
        df['Plant height(mm)'] = df[ph_col].values
    # else earlier conversion from 'Stem Length (mm)' already created 'Plant height (mm)' then ph_col points to that

    if stem_d_col:
        df['Stem Diameter(mm)'] = df[stem_d_col].values
    else:
        df['Stem Diameter(mm)'] = df.get('Stem Diameter (mm)', np.nan)

    if internodes_col:
        df['No of internodes'] = df[internodes_col].values
    else:
        df['No of internodes'] = df.get('No of internodes', np.nan)

    if pods_col:
        df['No of pods'] = df[pods_col].values
    else:
        df['No of pods'] = df.get('No of pods', np.nan)

    # Apply isotonic (for height/diameter) and cummax for counts, per Subject
    iso = IsotonicRegression(increasing=True)

    # For safety set columns present
    if 'Plant height(mm)' not in df.columns:
        df['Plant height(mm)'] = np.nan
    if 'Stem Diameter(mm)' not in df.columns:
        df['Stem Diameter(mm)'] = np.nan

    for subject, group in df.groupby('Subject'):
        idx = group.index
        x = np.arange(len(group))

        # fetch series or NaNs
        y1 = group['Plant height(mm)'].astype(float).values
        y2 = group['Stem Diameter(mm)'].astype(float).values
        y3 = group['No of internodes'].astype(float).values
        y4 = group['No of pods'].astype(float).values

        # Handle isotonic safely: need finite numeric arrays; fallback to interpolation if necessary
        if np.isfinite(y1).any():
            # fill small gaps for fit
            y1_filled = pd.Series(y1).interpolate().fillna(method='ffill').fillna(method='bfill').values
            try:
                y_fixed1 = iso.fit_transform(x, y1_filled)
            except Exception:
                y_fixed1 = y1_filled
        else:
            y_fixed1 = y1  # all NaN likely

        if np.isfinite(y2).any():
            y2_filled = pd.Series(y2).interpolate().fillna(method='ffill').fillna(method='bfill').values
            try:
                y_fixed2 = iso.fit_transform(x, y2_filled)
            except Exception:
                y_fixed2 = y2_filled
        else:
            y_fixed2 = y2

        # counts -> cumulative max
        try:
            y_fixed3 = pd.Series(y3).cummax().values
        except Exception:
            y_fixed3 = y3
        try:
            y_fixed4 = pd.Series(y4).cummax().values
        except Exception:
            y_fixed4 = y4

        df.loc[idx, 'Plant height(mm)'] = y_fixed1
        df.loc[idx, 'Stem Diameter(mm)'] = y_fixed2
        df.loc[idx, 'No of internodes'] = y_fixed3
        df.loc[idx, 'No of pods'] = y_fixed4

    # final sort and return
    df = df.sort_values(['Subject', 'Date', 'Time']).reset_index(drop=True)
    return df

def create_sequences(group, features, target_col, window_size=3):
    """
    From a single subject-group DataFrame, build sliding windows:
    Returns X (n_windows, window_size, n_features), y (n_windows,), dates (n_windows,) where dates are
    the date corresponding to the target (i+window_size).
    """
    X, y, dates = [], [], []
    data = group[features + [target_col]].values
    date_values = group['Date'].values
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :-1])         # features for window
        y.append(data[i+window_size, -1])           # target at next timestep
        dates.append(date_values[i+window_size])    # date for that target
    if len(X) == 0:
        return np.empty((0, window_size, len(features))), np.empty((0,)), np.empty((0,), dtype='datetime64[ns]')
    return np.array(X), np.array(y), np.array(dates, dtype='datetime64[ns]')

def make_subject_dicts(df, features, target_col, window_size=3):
    """
    Iterate over subjects and build dictionaries:
       X_dict[subj] -> array (n_windows, window_size, n_features)
       y_dict[subj] -> array (n_windows,)
       date_dict[subj] -> array of datetime for each prediction window (n_windows,)
    """
    X_dict = {}
    y_dict = {}
    date_dict = {}

    for subj, group in df.groupby('Subject'):
        group = group.sort_values(['Date', 'Time'])
        X_seq, y_seq, d_seq = create_sequences(group, features, target_col, window_size=window_size)
        if X_seq.shape[0] > 0:
            X_dict[subj] = X_seq
            y_dict[subj] = y_seq
            date_dict[subj] = d_seq
    return X_dict, y_dict, date_dict

# -----------------------------
# 1️⃣ Load and preprocess training sheet
# -----------------------------
df_train_raw = pd.read_excel(EXCEL_PATH, sheet_name=TRAIN_SHEET)
df_train = preprocess_sheet(df_train_raw)
print("Training sheet loaded. Columns:", df_train.columns.tolist())

# -----------------------------
# 2️⃣ Define features and target (canonical names used by preprocess_sheet)
# -----------------------------
env_features = [
    "Temp (F)", "CO2 (ppm)", "Relative Humidity (%)",
    "Luminous Flux (lux)", "Soil Temperature (F)",
    "Soil PH", "Soil moisture content (%)", "Plant height(mm)"
]
target_original_col = "Plant height(mm)"

# sanity check presence
missing_train = [c for c in env_features if c not in df_train.columns]
if missing_train:
    raise ValueError(f"Missing required feature(s) in training data: {missing_train}")

# create target_original for inverse-scaling later
df_train["target_original"] = df_train[target_original_col].values

# -----------------------------
# 3️⃣ Fit scalers on training data only
# -----------------------------
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit on entire training rows (per-feature)
df_train[env_features] = feature_scaler.fit_transform(df_train[env_features])
df_train["target_scaled"] = target_scaler.fit_transform(df_train[["target_original"]])
target_col_scaled = "target_scaled"

# -----------------------------
# 4️⃣ Create sequences for training (per-plant)
# -----------------------------
X_train_dict, y_train_dict, date_train_dict = make_subject_dicts(df_train, env_features, target_col_scaled, window_size=WINDOW_SIZE)

if len(X_train_dict) == 0:
    raise ValueError("No training sequences could be generated. Decrease WINDOW_SIZE or check training data.")

# Combine all training windows into arrays (for random splitting into train/val)
X_train_all = np.vstack(list(X_train_dict.values()))
y_train_all = np.hstack(list(y_train_dict.values()))
print(f"Total training windows: {X_train_all.shape[0]} (X shape {X_train_all.shape}, y shape {y_train_all.shape})")

# -----------------------------
# 5️⃣ Split out validation (10% of windows)
# -----------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_all, y_train_all, test_size=VALIDATION_FRAC, random_state=RANDOM_STATE, shuffle=True
)
print(f"Train windows: {X_tr.shape[0]}, Val windows: {X_val.shape[0]}")

# -----------------------------
# 6️⃣ Load and preprocess test sheet
# -----------------------------
df_test_raw = pd.read_excel(EXCEL_PATH, sheet_name=TEST_SHEET)
df_test = preprocess_sheet(df_test_raw)
print("Test sheet loaded. Columns:", df_test.columns.tolist())

# ensure required columns present
missing_test = [c for c in env_features if c not in df_test.columns]
if missing_test:
    raise ValueError(f"Missing required feature(s) in test data: {missing_test}")

# create target_original on test (real units) and then scale using training scalers
df_test["target_original"] = df_test[target_original_col].values

# -----------------------------
# 7️⃣ Transform test features & target using the training scalers
# -----------------------------
df_test[env_features] = feature_scaler.transform(df_test[env_features])
df_test["target_scaled"] = target_scaler.transform(df_test[["target_original"]])
target_col_scaled = "target_scaled"

# -----------------------------
# 8️⃣ Create sequences for test (per-plant)
# -----------------------------
X_test_dict, y_test_dict, date_test_dict = make_subject_dicts(df_test, env_features, target_col_scaled, window_size=WINDOW_SIZE)

if len(X_test_dict) == 0:
    print("Warning: no test sequences created. Maybe window_size too large or test plants too short.")
else:
    X_test_all = np.vstack(list(X_test_dict.values()))
    y_test_all = np.hstack(list(y_test_dict.values()))
    print(f"Total test windows: {X_test_all.shape[0]} (X_test shape {X_test_all.shape}, y_test shape {y_test_all.shape})")

# -----------------------------
# 9️⃣ Build LSTM model
# -----------------------------
n_timesteps = X_tr.shape[1]
n_features = X_tr.shape[2]

model = Sequential([
    LSTM(128, input_shape=(n_timesteps, n_features)),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# -----------------------------
# 10️⃣ Train model (with validation)
# -----------------------------
history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# -----------------------------
# 11️⃣ Evaluate on test set (global metrics)
# -----------------------------
if len(X_test_dict) > 0:
    y_pred_test = model.predict(X_test_all, verbose=0)
    # inverse transform to real units
    y_pred_test_real = target_scaler.inverse_transform(y_pred_test)
    y_test_real = target_scaler.inverse_transform(y_test_all.reshape(-1,1))

    mse = mean_squared_error(y_test_real, y_pred_test_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_real, y_pred_test_real)
    r2 = r2_score(y_test_real, y_pred_test_real)

    print("\nGlobal Test metrics (across all test windows):")
    print(f"R²:   {r2:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

# -----------------------------
# 12️⃣ Per-plant plots & metrics (test). Use actual dates from date_test_dict
# -----------------------------
pred_dict = {}
if len(X_test_dict) > 0:
    for subj, X_sub in X_test_dict.items():
        y_sub = y_test_dict[subj]
        d_sub = date_test_dict[subj]  # corresponding dates for each predicted y
        pred_sub = model.predict(X_sub, verbose=0)
        pred_sub_real = target_scaler.inverse_transform(pred_sub)
        y_sub_real = target_scaler.inverse_transform(y_sub.reshape(-1,1))
        pred_dict[subj] = (y_sub_real, pred_sub_real, d_sub)

    # compute per-plant metrics and plot
    mse_all = []
    rmse_all = []
    mae_all = []
    r2_all = []
    for subj, (y_real_plant, pred_real_plant, dates) in pred_dict.items():
        y_true = y_real_plant.flatten()
        y_pred = pred_real_plant.flatten()
        if len(y_true) == 0:
            continue

        # metrics (if only one sample, r2 is nan)
        r2_p = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        mse_p = mean_squared_error(y_true, y_pred)
        rmse_p = np.sqrt(mse_p)
        mae_p = mean_absolute_error(y_true, y_pred)

        r2_all.append(r2_p)
        mse_all.append(mse_p)
        rmse_all.append(rmse_p)
        mae_all.append(mae_p)

        # Plot using exact dates
        plt.figure(figsize=(5,8))
        plt.plot(dates, y_true, marker='o', label='Actual')
        plt.plot(dates, y_pred, marker='x', label='Predicted')
        plt.title(f'Plant {subj}')
        plt.xlabel('Date')
        plt.ylabel('Plant height (mm)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    print("\nPer-plant test averages:")
    print(f"R² (mean over plants):   {np.nanmean(r2_all):.4f}")
    print(f"MSE (mean over plants):  {np.mean(mse_all):.4f}")
    print(f"RMSE (mean over plants): {np.mean(rmse_all):.4f}")
    print(f"MAE (mean over plants):  {np.mean(mae_all):.4f}")
else:
    print("No test plants to evaluate (no sequences).")

# -----------------------------
# End
# -----------------------------
