from datetime import date
from pathlib import Path
import ast
import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from IPython.display import display, HTML
except Exception:
    def display(*_args, **_kwargs):
        return None

    def HTML(value):
        return value

try:
    import ipywidgets as widgets
except Exception:
    widgets = None
import librosa
import time
import matplotlib.gridspec as gridspec
try:
    import contextily as cx
except Exception:
    cx = None
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from math import gcd
from threading import Timer
from dataclasses import dataclass
import sounddevice as sd


def _can_use_ipywidgets_canvas(fig) -> bool:
    """True when running with an ipympl canvas that supports widget layout."""
    try:
        _ = fig.canvas.layout
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
class DataFrameSchema:
    """Holds a column-to-dtype mapping and can coerce a DataFrame to match it.
    
    Columns absent from the schema are left untouched. Values that cannot be
    coerced are set to NaN/NaT silently.
    """

    def __init__(self, schema: dict[str, str]):
        self.schema = schema
        self.headers = list(schema.keys())

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, dtype in self.schema.items():
            if col not in df.columns:
                continue
            try:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'string':
                    df[col] = df[col].astype('string')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass
        return df


_metadata_schema = DataFrameSchema({
    'filename':         'string',
    'collection':       'string',
    'secondary_labels': 'string',
    'url':              'string',
    'latitude':         'float64',
    'longitude':        'float64',
    'author':           'string',
    'license':          'string',
    'recorded_on':      'datetime64[ns]',
    'reviewed_by':      'string',
    'reviewed_on':      'datetime64[ns]',
    'source_filename':  'string',
    'source_start_s':   'float64',
    'source_end_s':     'float64',
    'models_used':      'string',
})

_label_schema = DataFrameSchema({
    'Filename':                     'string',
    'Start Time (s)':               'float64',
    'End Time (s)':                 'float64',
    'Low Freq (Hz)':                'float64',
    'High Freq (Hz)':               'float64',
    'Label':                        'string',
    'Type':                         'string',
    'Sex':                          'string',
    'Score':                        'float64',
    'Life Stage':                   'string',
    'Indv ID':                      'string',
    'Delta Time (s)':               'float64',
    'Delta Freq (Hz)':              'float64',
    'Avg Power Density (dB FS/Hz)': 'float64',
})




def calc_signal_pwr(wav, chunk_len, sr=32000):
    power = wav ** 2 
    power = np.pad(power, (0, int(np.ceil(len(power) / chunk_len) * chunk_len - len(power))))
    power = power.reshape((-1, chunk_len)).sum(axis=1)
    return power

def calc_band_power(spec, ymin_idx, ymax_idx):
    band_spec = spec[ymin_idx:ymax_idx+1, :]
    band_power = band_spec.sum(axis=0)
    return 10 * np.log10(band_power + 1e-12)


def normalize_secondary_labels(x):
    # Missing values
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # Already correct
    if isinstance(x, list):
        return [str(i) for i in x]

    # Tuple → list
    if isinstance(x, tuple):
        return [str(i) for i in x]

    # NumPy array → list
    if isinstance(x, np.ndarray):
        return [str(i) for i in x.tolist()]

    # Stringified list (most important case)
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)

            # parsed could still be weird (single string, etc.)
            if isinstance(parsed, (list, tuple)):
                return [str(i) for i in parsed]
            else:
                return [str(parsed)]

        except Exception:
            # fallback: treat as single label string
            return [x]

    # Fallback (unknown type)
    return [str(x)]




class CQTSpectrogramMaker:
    def __init__(self, sr=32000, n_bins=84, fmin=20, bins_per_octave=12):
        self.sr = sr
        self.n_bins = n_bins
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave

    def create_cqt(self, waveform):
        cqt = librosa.cqt(
            waveform,
            sr=self.sr,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt))
        num_frames = cqt_db.shape[1]
        duration = len(waveform) / self.sr
        time_axis = np.linspace(0, duration, num=num_frames)
        freqs = librosa.cqt_frequencies(
            n_bins=self.n_bins,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave
        )
        return cqt_db, time_axis, freqs


class MelSpecMaker:
    def __init__(
        self,
        sr=32000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,          # explicit and fixed
        f_min=20,
        f_max=14000,
        pcen=False
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pcen = pcen


    def create_melspec(self, waveform):
        """
        waveform: 1D numpy array
        returns:
            mel_spec (numpy array) shape (n_mels, time_frames)
            time_axis (seconds)
            frequencies (Hz) - centre frequency of each mel filter
        """
        waveform = waveform.astype(np.float32)

        # --- mel spectrogram (power) ---
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )  # shape (n_mels, frames), linear power scale

        # --- optional PCEN ---
        if self.pcen:
            mel_spec = librosa.pcen(
                mel_spec,
                sr=self.sr,
                hop_length=self.hop_length,
                gain=0.99,
                bias=2,
                power=0.5,
                time_constant=0.4,
            )
        else:
            mel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10)

            # --- robust dynamic range normalisation ---
            mel_spec = mel_spec - mel_spec.max()

            p_low  = np.percentile(mel_spec, 5)
            p_high = np.percentile(mel_spec, 99.5)
            mel_spec = np.clip(mel_spec, p_low, p_high)

        # --- time axis ---
        num_frames = mel_spec.shape[1]
        time_axis = np.arange(num_frames) * self.hop_length / self.sr

        # --- centre frequency of each mel filter via filterbank argmax ---
        # This matches what the torchaudio version was doing
        fb = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )  # shape (n_mels, n_fft//2+1)

        fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        frequencies = fft_freqs[fb.argmax(axis=1)]  # centre freq per filter

        return mel_spec, time_axis, frequencies




class STFTMaker():
    def __init__(self, sr=32000, n_fft=2048, hop_length = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.freq_resolution = sr / n_fft
        self.hop_length = hop_length
        
    def create_stft(self, waveform):
        stft = librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann"
        )

        power = np.abs(stft) ** 2
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        times = librosa.frames_to_time(
                                    np.arange(power.shape[1]),
                                    sr=self.sr,
                                    hop_length=self.hop_length
                                )
        return power, times, freqs


class FastMap:

    # Default extents in WGS84
    default_extents = {
        'max_longitude': 174.0,
        'min_longitude': 173.0,
        'max_latitude': -41.0,
        'min_latitude': -41.5
    }

    def __init__(
        self,
        provider=None,
        #provider: dict = cx.providers["OpenStreetMap"]["Mapnik"],
        figsize=(3, 3),
        ax=None,
        map_extents=None,
        web_mercator_extents=None        # e.g. {'min_x': ..., 'max_x': ..., 'min_y': ..., 'max_y': ...}
    ):
        if provider is None and cx is not None:
            provider = cx.providers.OpenStreetMap.Mapnik
        self.provider = provider

        if web_mercator_extents is not None:
            # User supplied Web Mercator directly — use as-is
            self.min_x = web_mercator_extents['min_x']
            self.max_x = web_mercator_extents['max_x']
            self.min_y = web_mercator_extents['min_y']
            self.max_y = web_mercator_extents['max_y']
            # Back-convert to WGS84 for bounds checking in update()
            self.min_long, self.min_lat = self._webmercator_to_wgs84(self.min_x, self.min_y)
            self.max_long, self.max_lat = self._webmercator_to_wgs84(self.max_x, self.max_y)
        else:
            # Use WGS84 extents (default path)
            extents = map_extents if map_extents is not None else self.default_extents
            self.min_long = extents['min_longitude']
            self.max_long = extents['max_longitude']
            self.min_lat  = extents['min_latitude']
            self.max_lat  = extents['max_latitude']
            self.min_x, self.min_y = self._wgs84_to_webmercator(self.min_long, self.min_lat)
            self.max_x, self.max_y = self._wgs84_to_webmercator(self.max_long, self.max_lat)

        self._interactive_state = plt.isinteractive()
        plt.ioff()

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, num="Location (if available)")
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.ax.set_position([0, 0, 1, 1])
            self._owns_figure = True
        else:
            self.ax = ax
            self.fig = ax.figure
            self._owns_figure = False

        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal', adjustable='box')

        if cx is not None and self.provider is not None:
            cx.add_basemap(self.ax, source=self.provider, zoom='auto')

        self.ax.plot(
            [self.min_x, self.max_x, self.max_x, self.min_x, self.min_x],
            [self.min_y, self.min_y, self.max_y, self.max_y, self.min_y],
            color='black', linewidth=0.5
        )

        self.marker, = self.ax.plot([], [], 'ro', markersize=4)

    # ------------------------------------------------------------------ #
    #  Coordinate helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wgs84_to_webmercator(lon, lat):
        """Convert WGS84 degrees to Web Mercator (EPSG:3857) metres."""
        x = math.radians(lon) * 6378137.0
        y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 6378137.0
        return x, y

    @staticmethod
    def _webmercator_to_wgs84(x, y):
        """Convert Web Mercator (EPSG:3857) metres back to WGS84 degrees."""
        lon = math.degrees(x / 6378137.0)
        lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
        return lon, lat

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def update(self, lat=None, lon=None):
        if (
            lat is not None and lon is not None
            and self.min_lat <= lat <= self.max_lat
            and self.min_long <= lon <= self.max_long
        ):
            x, y = self._wgs84_to_webmercator(lon, lat)
            self.marker.set_data([x], [y])
        else:
            self.marker.set_data([], [])

        self.fig.canvas.draw_idle()

    def display(self):
        if self._owns_figure:
            plt.show(block=False)
        if self._interactive_state:
            plt.ion()




def normalise_labels_df(df, schema):
    # ensure all expected columns exist
    for c in schema:
        if c not in df.columns:
            df[c] = pd.NA

    # enforce column order
    df = df[schema]

    # IMPORTANT: prevent "all-NA dtype ambiguity"
    for c in df.columns:
        if df[c].isna().all():
            df[c] = df[c].astype("object")
    df = df.astype(object)
    

    return df


# ----------------------------
# Annotation State
# ----------------------------
class AnnotationState:
    def __init__(self, all_classes, max_visible=30):
        self.all_classes = list(sorted(all_classes))            # master list
        self.max_visible = max_visible
        self.visible_classes = ['Unknown', 'Unknown Insect']
        self.current_label = self.visible_classes[0] if self.visible_classes else None
        self.sex = None
        self.life_stage = None
        self.score = None
        self.call_type = None

    # Set the current active class for annotation
    def set_label(self, label):
        if label in self.visible_classes:
            self.current_label = label
        else:
            raise ValueError(f"Label '{label}' not in visible_classes")

    # Update the subset of classes currently visible
    def set_visible_classes(self, new_visible):
        self.visible_classes = new_visible[:self.max_visible]

        # Reset current_label if it's no longer visible
        if self.current_label not in self.visible_classes:
            self.current_label = self.visible_classes[0] if self.visible_classes else None

    # Helpers for widgets
    def get_visible_classes(self):
        return self.visible_classes

    def get_all_classes(self):
        return self.all_classes
    
    def autoselect_class_from_session(self, session):
        filepath = getattr(session, "current_filepath", None)
        if filepath is None:
            return

        folder = Path(filepath).parent.name
        cls = self.ebird_to_common.get(folder)

        if cls is None:
            return

        visible = self.get_visible_classes()

        if cls not in visible:
            visible.append(cls)
            self.set_visible_classes(visible)  # enforces max_visible truncation

        # update radio widget options to match the (possibly truncated) visible list
        options = [
            (f"{c} ({self.common_to_ebird.get(c, '')})", c)
            for c in self.visible_classes  # use self.visible_classes, not visible, in case it was truncated
        ]
        self.radio_class.options = options

        # only set if cls survived truncation
        if cls in self.visible_classes:
            self.radio_class.value = cls
            self.current_label = cls


def create_class_widgets(annotation_state,
                         fastmap: FastMap=None,
                         n_columns: int=5,
                         common_to_ebird: dict=None
                         ):
    # --- Checkbox grid for class visibility ---


    checkboxes = []
    for cls in annotation_state.get_all_classes():
        cb = widgets.Checkbox(
            value=cls in annotation_state.get_visible_classes(),
            description=cls,
            indent=False,
            layout=widgets.Layout(width="auto")
        )
        checkboxes.append(cb)

    grid = widgets.GridBox(
        checkboxes,
        layout=widgets.Layout(
            grid_template_columns=f"repeat({n_columns}, 1fr)",
            grid_gap="5px 25px",
            width="100%"
        )
    )

    # --- Radio buttons ---
    visible = annotation_state.get_visible_classes()

    options = [
        (f"{cls} ({common_to_ebird.get(cls, '')})", cls)
        for cls in visible
    ]

    current = annotation_state.current_label
    if current not in visible:
        current = visible[0] if visible else None

    radio_class = widgets.RadioButtons(
        options=options,
        value=current,
        description="Current Class:"
    )

    radio_sex = widgets.RadioButtons(
        options=['Leave Empty', 'Male', 'Female'],
        value='Leave Empty',
        description="Sex:"
    )

    radio_life_stage = widgets.RadioButtons(
        options=['Leave Empty', 'Juvenile', 'Adult'],
        value='Leave Empty',
        description="Life Stage:"
    )

    radio_call_type = widgets.RadioButtons(
        options=['Leave Empty', 'Call', 'Song', 'Alarm', 'Duet', 'Begging', 
                 'Flight call', 'Echolocation', 'Other'],
        value='Leave Empty',
        description="Call Type:"
    )

    radio_score = widgets.RadioButtons(
        options=['Leave Empty', 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        value='Leave Empty',
        description="Confidence Score:"
    )

    # --- Handlers for dynamic updates ---
    def on_checkbox_change(change):
        visible = [cb.description for cb in checkboxes if cb.value]
        annotation_state.set_visible_classes(visible)

        options = [
        (f"{cls} ({common_to_ebird.get(cls, '')})", cls)
        for cls in visible
    ]
        radio_class.options = options
    
        
        if annotation_state.current_label not in visible and visible:
            radio_class.value = visible[0]

    for cb in checkboxes:
        cb.observe(on_checkbox_change, names="value")

    radio_class.observe(lambda change: annotation_state.set_label(change["new"]), names="value")
    radio_sex.observe(lambda change: setattr(annotation_state, 'sex', change["new"]), names="value")
    radio_life_stage.observe(lambda change: setattr(annotation_state, 'life_stage', change["new"]), names="value")
    radio_call_type.observe(lambda change: setattr(annotation_state, 'call_type', change["new"]), names="value")
    radio_score.observe(lambda change: setattr(annotation_state, 'score', change["new"]), names="value")

    # --- Layout for radio buttons ---
    radio_columns = [
        widgets.Box([radio_class], layout=widgets.Layout(width="20%")),
        widgets.Box([radio_sex], layout=widgets.Layout(width="20%")),
        widgets.Box([radio_life_stage], layout=widgets.Layout(width="20%")),
        widgets.Box([radio_call_type], layout=widgets.Layout(width="20%")),
        widgets.Box([radio_score], layout=widgets.Layout(width="20%")),
    ]

    # --- Map display inline (ipympl canvas) ---
    if fastmap is not None:
        canvas_widget = fastmap.fig.canvas
        canvas_widget.layout.width = '40%'  # adjust as desired
        radio_columns.append(canvas_widget)

    # --- Bottom row ---
    bottom_row = widgets.HBox(
        radio_columns,
        layout=widgets.Layout(align_items="flex-start", width="100%")
    )

    # --- Top row (checkbox grid) stacked above bottom row ---
    container = widgets.VBox(
        [grid, bottom_row],
        layout=widgets.Layout(width="100%")
    )

    annotation_state.ebird_to_common = {v: k for k, v in common_to_ebird.items()}
    annotation_state.common_to_ebird = common_to_ebird
    annotation_state.radio_class = radio_class

    display(container)


def avg_power_from_box(spec: np.ndarray) -> float:
    """
    Compute Avg Power Density (dB FS/Hz) from a precomputed spectrogram crop.
    """
    mean_power = spec.mean().item()
    bandwidth = spec.shape[0]
    power_density = mean_power / bandwidth
    return round(10 * np.log10(power_density + 1e-12), 1)


def zoom_in_on_wav(
    wav: np.ndarray,
    x_left: float,
    f_min_hz: float,
    f_max_hz: float,
    times: np.ndarray,
    window_width: float = 5.0,
    sr: float = 32000,
    filter_order: int = 4,
    nyquist_margin: float = 1.1,
    ):

    # 1. BANDPASS FILTER on full wav (avoids edge artifacts from cropping first)
    low  = f_min_hz / (sr / 2)
    high = f_max_hz / (sr / 2)
    b, a = butter(filter_order, [low, high], btype='band')
    filtered = filtfilt(b, a, wav)

    # 2. RESAMPLE
    target_sr = nyquist_margin * 2 * f_max_hz
    new_sr    = min(sr, target_sr)

    if abs(new_sr - sr) < 1:
        new_sr    = sr
        resampled = filtered
    else:
        up, down = int(new_sr), int(sr)
        g        = gcd(up, down)
        up      //= g
        down    //= g
        resampled = resample_poly(filtered, up, down)
        new_sr    = sr * up / down

    # 3. TIME CROP after resample so indices use new_sr
    t_start = max(0.0, x_left)
    t_end   = min(times[-1], x_left + window_width)
    s0      = int(t_start * new_sr)
    s1      = int(t_end   * new_sr)

    return resampled[s0:s1], new_sr


def optimal_stft_params(
    sr: float,
    f_min: float,
    cycles: int = 8,
    hop_fraction: float = 0.25,
    min_fft: int = 64,
    max_fft: int = 8192,
):
    """
    Optimise STFT parameters for a band-limited signal.

    Args:
        sr: sample rate after resampling
        f_min: lowest frequency of interest (Hz)
        cycles: number of cycles of f_min to include in window
        hop_fraction: hop length as fraction of n_fft
        min_fft: lower clamp for n_fft
        max_fft: upper clamp for n_fft

    Returns:
        n_fft
        hop_length
        window_duration_sec
        freq_resolution_hz
    """

    # Avoid pathological cases
    f_min = max(f_min, 1.0)

    # ----------------------------------
    # 1. Physics-based window estimate
    # ----------------------------------

    n_fft_est = cycles * sr / f_min

    # Round to nearest power of 2
    n_fft = int(2 ** np.round(np.log2(n_fft_est)))

    # Clamp to bounds
    n_fft = int(np.clip(n_fft, min_fft, max_fft))

    # ----------------------------------
    # 2. Hop length
    # ----------------------------------

    hop_length = int(n_fft * hop_fraction)

    # ----------------------------------
    # 3. Diagnostics
    # ----------------------------------

    window_duration = n_fft / sr
    freq_resolution = sr / n_fft

    return n_fft, hop_length, window_duration, freq_resolution


def play_audio_standalone(wav_segment, sr, peak_reference=None):
    """
    Play a numpy array via system audio, non-blocking.
    wav_segment: 1D numpy float32 array
    sr: sample rate
    """
    wav_segment = wav_segment.astype(np.float32)
    if peak_reference is None:
        peak_reference = np.abs(wav_segment).max()
    if peak_reference > 0:
        wav_segment = wav_segment / peak_reference

    sd.stop()  # kill any current playback
    sd.play(wav_segment, samplerate=sr)


@dataclass
class SpectrogramData:
    """All computed data for a single audio file. No matplotlib dependencies."""
    wav: np.ndarray
    sr: int
    duration_seconds: float
    mel_spec_db: np.ndarray
    time_axis: np.ndarray
    frequencies: np.ndarray
    stft_power: np.ndarray
    stft_time: np.ndarray
    stft_freqs: np.ndarray
    freq_resolution: float
    power: np.ndarray

    @property
    def n_rows(self):
        return self.mel_spec_db.shape[0]

    @property
    def n_cols(self):
        return self.mel_spec_db.shape[1]

    def hz_to_row(self, y_hz):
        """Map physical Hz to mel spectrogram row index."""
        return np.interp(y_hz, self.frequencies, np.arange(self.n_rows))

    def row_to_hz(self, y_row):
        """Map mel spectrogram row index to physical Hz."""
        return np.interp(y_row, np.arange(len(self.frequencies)), self.frequencies)

    def hz_to_stft_row(self, hz):
        return np.searchsorted(self.stft_freqs, hz)

    def time_to_stft_col(self, t):
        return np.searchsorted(self.stft_time, t)

    def avg_power_density(self, fmin, fmax, tmin, tmax):
        """Compute average power density (dB FS/Hz) over a time-frequency box."""
        r0 = self.hz_to_stft_row(fmin)
        r1 = self.hz_to_stft_row(fmax)
        c0 = self.time_to_stft_col(tmin)
        c1 = self.time_to_stft_col(tmax)
        r0, r1 = sorted([r0, r1])
        c0, c1 = sorted([c0, c1])

        cropped = self.stft_power[r0:r1, c0:c1]
        if cropped.size == 0:
            return None

        mean_power = np.mean(cropped)
        bandwidth_hz = (r1 - r0) * self.freq_resolution
        power_density = mean_power / bandwidth_hz
        return round(10 * np.log10(power_density + 1e-12), 1)

    @classmethod
    def from_file(cls,
                  filepath,
                  n_mels=128,
                  f_min=20,
                  f_max=14000,
                  power_time_steps=0.1):
        """Load audio and compute all derived data."""
        wav, sr = librosa.load(filepath, sr=None)
        duration_seconds = len(wav) / sr

        specmaker = MelSpecMaker(sr=sr, n_mels=n_mels, f_min=f_min, f_max=f_max, pcen=False)
        stftmaker = STFTMaker(sr=sr)
        mel_spec_db, time_axis, frequencies = specmaker.create_melspec(wav)
        stft_power, stft_time, stft_freqs = stftmaker.create_stft(wav)  #need to ad

        power = calc_signal_pwr(wav, chunk_len=int(sr * power_time_steps))

        return cls(
            wav=wav,
            sr=sr,
            duration_seconds=duration_seconds,
            mel_spec_db=mel_spec_db,
            time_axis=time_axis,
            frequencies=frequencies,
            stft_power=stft_power,
            stft_time=stft_time,
            stft_freqs=stft_freqs,
            freq_resolution=stftmaker.freq_resolution,
            power=power,
        )


# =============================================================================
# AnnotationStore
# =============================================================================

class AnnotationStore:
    """
    Owns annotation data and the corresponding matplotlib artists.
    Keeps boxes, box_artists and text_artists in sync.
    """

    def __init__(self):
        self.boxes: list[dict] = []
        self._box_artists: list[plt.Rectangle] = []
        self._text_artists: list[plt.Text] = []

    def add(self, box_dict: dict, rect: plt.Rectangle, text: plt.Text):
        self.boxes.append(box_dict)
        self._box_artists.append(rect)
        self._text_artists.append(text)

    def undo(self):
        """Remove the most recently added annotation. Returns True if anything was removed."""
        if not self.boxes:
            return False
        self.boxes.pop()
        self._box_artists.pop().remove()
        self._text_artists.pop().remove()
        return True

    def clear(self):
        while self._box_artists:
            self._box_artists.pop().remove()
        while self._text_artists:
            self._text_artists.pop().remove()
        self.boxes.clear()

    def __len__(self):
        return len(self.boxes)


# =============================================================================
# SpectrogramAnnotator
# =============================================================================

class SpectrogramAnnotator:
    """
    Spectrogram annotator that maps boxes in physical Hz to display coordinates.
    Works for mel, CQT, PEN, STFT etc.
    """

    def __init__(self,
                 annotation_state,
                 common_to_ebird: dict,
                 plot_size: tuple = (18, 4),
                 power_time_steps: float = 0.1,
                 f_min: int = 20,
                 f_max: int = 14000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 zoom_window_width: float = 5,
                 zoom_window_height: float = 0.4,
                 min_freq_hz: int = 200,
                 full_width_box_min_hz: int = 200, 
                 full_width_box_max_hz: int = 14000,
                 min_drag_rows: int = 10,
                 min_drag_time_s: float = 0.5,
                 min_separation: float = 4,
                 similarness_threshold: float = 0.5
                 ):

        self.annotation_state = annotation_state
        self.common_to_ebird = common_to_ebird
        self.plot_size = plot_size
        self.power_time_steps = power_time_steps
        self.right_axis_limits = (-0.2, 1.2)
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.zoom_window_width = zoom_window_width
        self.zoom_window_height = zoom_window_height
        self.min_freq_hz = min_freq_hz
        self.full_width_box_min_hz = full_width_box_min_hz
        self.full_width_box_max_hz = full_width_box_max_hz
        self._last_box_time = None
        self._last_box_rows = None
        self.min_drag_time = min_drag_time_s      # seconds
        self.min_drag_rows = min_drag_rows        # spectrogram rows
        self.min_separation = min_separation
        self.similarness_threshold = similarness_threshold

        # --- data and annotations ---
        self.data: SpectrogramData | None = None
        self.annotations = AnnotationStore()
        self._last_box_freq = None  # (ymin_hz, ymax_hz)
        self.band_power = None

        # --- drag state ---
        self._drag_rect = None
        self._drag_start = None
        self._drag_button = None

        # --- zoom rect (drawn on ax_spec) ---
        self.zoom_rect = None

        # --- playhead artist handles ---
        self.centre_dot = None
        self.playhead_power = None
        self.playhead_spec = None
        self.playhead_zoom = None
        self._playhead_timer = None

        # --- file state ---
        self.file_loaded = False
        self.filepath = None
        self.meta_row = None
        self.label_rows = None
        self._audio_peak_reference = 1.0
        self.play_selected_on_right_click = False

        # --- build figure ---
        self._interactive_state = plt.isinteractive()
        plt.ioff()
        self.fig = plt.figure(figsize=self.plot_size)

        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[6, 1],
            width_ratios=[5, 2.5],
            hspace=0.05,
            wspace=0.1
        )
        self.ax_spec  = self.fig.add_subplot(gs[0, 0])
        self.ax_power = self.fig.add_subplot(gs[1, 0], sharex=self.ax_spec)
        self.ax_side  = self.fig.add_subplot(gs[:, 1])

        plt.setp(self.ax_spec.get_xticklabels(), visible=False)

        # --- connect events ---
        self.fig.canvas.mpl_connect('button_press_event',   self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',      self._on_keypress)
        self.fig.canvas.mpl_connect('button_press_event',   self._on_drag_start)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_drag_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_drag_release)

        if self._interactive_state:
            plt.ion()

        # --- notebook-only widget container (desktop fallback below) ---
        self.audio_output = None
        self.container = None
        self._notebook_ui = _can_use_ipywidgets_canvas(self.fig)

        if self._notebook_ui and widgets is not None:
            self.audio_output = widgets.Output(
                layout=widgets.Layout(
                    width="800px",
                    margin="0 0 0 35px"
                )
            )
            self.container = widgets.VBox([self.fig.canvas, self.audio_output])
            display(self.container)
            display(HTML("<style>audio { width: 800px; margin-left: 35px; }</style>"))
        else:
            plt.show(block=False)

    # ----------------------------
    # File loading
    # ----------------------------

    def load_file(self, filepath, meta_row, label_rows):
        self.filepath = filepath
        self.meta_row = meta_row
        self.label_rows = label_rows

        self._prepare_data()
        self._render()
        self._load_existing_annotations()
        self._render_zoom()
        
        self.file_loaded = True
        self._play_audio(start_time=0.0)

    def _prepare_data(self):
        self.data = SpectrogramData.from_file(
            self.filepath,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power_time_steps=self.power_time_steps,
        )
        self.t_marker = self.data.duration_seconds / 2
        self.f_marker = self.data.n_rows / 2
        self._audio_peak_reference = max(float(np.abs(self.data.wav).max()), 1e-12)

    # ----------------------------
    # Audio / playhead
    # ----------------------------

    def _play_audio(self, start_time=0.0, end_time=None):
        if self.data is None or self.meta_row is None:
            return

        sr = self.meta_row.get("sample_rate", None)
        if sr is None:
            sr = librosa.get_samplerate(self.filepath)

        start_sample = int(start_time * sr)
        if end_time is None:
            wav_segment = self.data.wav[start_sample:]
        else:
            end_sample = int(max(start_time, end_time) * sr)
            wav_segment = self.data.wav[start_sample:end_sample]
        if wav_segment.size == 0:
            return

        if self.audio_output is not None:
            with self.audio_output:
                self.audio_output.clear_output(wait=True)
                play_audio_standalone(
                    wav_segment=wav_segment,
                    sr=sr,
                    peak_reference=self._audio_peak_reference,
                )
        else:
            play_audio_standalone(
                wav_segment=wav_segment,
                sr=sr,
                peak_reference=self._audio_peak_reference,
            )

        self._start_playhead(start_time=start_time, end_time=end_time)

    def _start_playhead(self, start_time=0.0, end_time=None, interval=0.2):
        self._stop_playhead_timer()

        # Generation counter — stale callbacks exit cleanly without shared mutable flags
        self._playhead_gen = getattr(self, "_playhead_gen", 0) + 1
        my_gen = self._playhead_gen
        play_end = self.data.duration_seconds if end_time is None else min(end_time, self.data.duration_seconds)

        self._play_start_wall  = time.time()
        self._play_start_audio = start_time

        interval_ms = max(30, int(interval * 1000))
        self._playhead_timer = self.fig.canvas.new_timer(interval=interval_ms)

        def update_loop():
            if self._playhead_gen != my_gen:
                self._stop_playhead_timer()
                return
            if self.playhead_spec is None or self.playhead_power is None or self.fig is None:
                self._stop_playhead_timer()
                return

            elapsed = time.time() - self._play_start_wall
            current_time = self._play_start_audio + elapsed
            done = current_time >= play_end
            if done:
                current_time = play_end

            try:
                self.playhead_spec.set_xdata([current_time])
                self.playhead_power.set_xdata([current_time])
                if self.playhead_zoom is not None:
                    self.playhead_zoom.set_xdata([current_time])
                self.fig.canvas.draw_idle()
            except Exception:
                self._stop_playhead_timer()
                return

            if done:
                self._stop_playhead_timer()

        self._playhead_timer.add_callback(update_loop)
        self._playhead_timer.start()
        update_loop()

    def _stop_playhead_timer(self):
        if self._playhead_timer is not None:
            try:
                self._playhead_timer.stop()
            except Exception:
                pass
            self._playhead_timer = None

    # ----------------------------
    # Rendering
    # ----------------------------

    def display(self):
        if self.container is not None:
            return self.container
        plt.show(block=False)
        return self.fig

    def _render(self):
        self._stop_playhead_timer()
        self.annotations.clear()
        self.ax_spec.clear()
        self.ax_power.clear()
        self.centre_dot    = None
        self.playhead_power = None
        self.playhead_spec  = None
        self.zoom_rect      = None

        d = self.data
        t_power  = np.linspace(0, d.duration_seconds, len(d.power))
        n_rows, n_cols = d.mel_spec_db.shape
        extent = [0, d.duration_seconds, 0, n_rows]

        spec = d.mel_spec_db.copy()
        vmin_global = np.percentile(spec, .5)
        vmax_global = np.percentile(spec, 99.5)
        spec = np.clip(spec, vmin_global, vmax_global)

        noise_profile   = np.percentile(spec, .5, axis=1, keepdims=True)
        spec_suppressed = np.maximum(spec - noise_profile, 0)

        self.ax_spec.imshow(
            spec_suppressed,
            origin='lower',
            aspect='auto',
            cmap='magma',
            extent=extent,
        )

        t_max = d.duration_seconds
        for t in np.arange(0, t_max, 5):
            self.ax_spec.axvline(
                t,
                color='white',
                linewidth=1,
                alpha=0.2,
                zorder=5
            )


        # --- playheads ---
        self.playhead_spec = self.ax_spec.axvline(
            0, color='cyan', linewidth=1.5, alpha=0.9, zorder=30
        )
        self.playhead_power = self.ax_power.axvline(
            0, color='cyan', linewidth=1.5, alpha=0.9, zorder=30
        )

        # --- centre dot ---
        self.centre_dot, = self.ax_spec.plot(
            self.t_marker, self.f_marker,
            'o', color='red', markersize=5, zorder=20
        )

        # --- y-axis ticks in Hz ---
        n_ticks    = 6
        yticks_row = np.linspace(0, n_rows - 1, n_ticks)
        yticks_hz  = np.interp(yticks_row, np.arange(n_rows), d.frequencies)
        yticks_hz[0]  = self.f_min   # cosmetic change only to force display label at bottom
        yticks_hz[-1] = self.f_max   # cosmetic change only to force display label at top

        self.ax_spec.set_ylabel('Mel Frequency (Hz)')
        self.ax_spec.set_yticks(yticks_row)
        self.ax_spec.set_yticklabels([f"{f:.0f}" for f in yticks_hz])
        self.ax_spec.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # --- title ---
        secondaries = self.meta_row.get('secondary_labels', '')
        recorded_on = self.meta_row.get('recorded_on', '')
        parts = [p for p in [str(secondaries), str(recorded_on)] if p]
        title = f"{self.filepath.parent.name}/{self.filepath.name}"
        if parts:
            title += ": " + " | ".join(parts)
        self.ax_spec.set_title(title)

        if hasattr(self, 'band_power') and self.band_power is not None:
            power_to_plot = self.band_power
            t_plot = self.band_power_time_axis
        else:
            eps       = 1e-12
            power_db  = 10 * np.log10(d.power + eps)
            pmin, pmax = power_db.min(), power_db.max()
            power_to_plot = (power_db - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(power_db)
            t_plot = t_power

        # store the line for later updates
        self.power_line, = self.ax_power.plot(
            t_plot, power_to_plot, label='Power', color='black', linewidth=1, zorder=5
        )

        self.ax_power.set_xlabel("Time (s)")
        self.ax_power.tick_params(axis='y', labelcolor='m')
        self.ax_power.set_ylim(self.right_axis_limits)
        self.ax_power.legend(
            loc="upper right",
            bbox_to_anchor=(.9, -0.35),
            frameon=False,
            ncol=2
        )

        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        self.ax_spec.set_xlim(0, d.duration_seconds)
        self.fig.canvas.draw_idle()


    def _render_zoom(self):
        self.ax_side.clear()

        self.playhead_zoom = self.ax_side.axvline(
            self.t_marker, color='cyan', linewidth=1.5, alpha=0.9, zorder=30
        )
        self.ax_side.set_title(
            f'Zoomed & Filtered STFT: {self.zoom_window_width} sec x {self.zoom_window_height} f-range'
        )

        # --------------------------------------------------
        # Calculate frequency window in mel rows, then convert to Hz
        # --------------------------------------------------
        n_mels           = self.data.n_rows
        half_height_bins = int((self.zoom_window_height * n_mels) / 2)
        f_centre_row     = int(np.clip(self.f_marker, 0, n_mels - 1))

        r0 = max(0,          f_centre_row - half_height_bins)
        r1 = min(n_mels - 1, f_centre_row + half_height_bins)

        f_min_hz = float(self.data.frequencies[r0])
        f_max_hz = float(self.data.frequencies[r1])

        # --------------------------------------------------
        # Zoom rect on main spectrogram (row coordinates)
        # --------------------------------------------------
        height = r1 - r0
        y0     = r0

        if self.zoom_rect is None:
            self.zoom_rect = plt.Rectangle(
                                            (self.t_marker, y0),
                                            self.zoom_window_width,
                                            height,
                                            edgecolor='white',
                                            facecolor='none',
                                            linewidth=1,
                                            linestyle='--',
                                            zorder=10,
                                        )
            self.ax_spec.add_patch(self.zoom_rect)
        else:
            self.zoom_rect.set_x(self.t_marker)
            self.zoom_rect.set_y(y0)
            self.zoom_rect.set_width(self.zoom_window_width)
            self.zoom_rect.set_height(height)

        # --------------------------------------------------
        # Get filtered, resampled wav for this window
        # --------------------------------------------------
        zoomed_wav, new_sr = zoom_in_on_wav(
                                            self.data.wav,
                                            self.t_marker,
                                            f_min_hz,
                                            f_max_hz,
                                            self.data.time_axis,
                                            window_width=self.zoom_window_width,
                                            sr=self.data.sr,
                                        )

        # --------------------------------------------------
        # STFT and frequency crop
        # --------------------------------------------------
        n_fft, hop_length, win_dur, f_res = optimal_stft_params(new_sr, f_min_hz, cycles=64, hop_fraction=.1)
        zoomed_specmaker = STFTMaker(sr=new_sr, n_fft=n_fft, hop_length=hop_length)
        spec, time_axis, freqs = zoomed_specmaker.create_stft(zoomed_wav)

        freq_mask    = (freqs >= f_min_hz) & (freqs <= f_max_hz)
        spec         = spec[freq_mask, :]
        freqs_sliced = freqs[freq_mask]

        # --------------------------------------------------
        # 
        # 
        #  and display
        # --------------------------------------------------
        vmin_global  = np.percentile(spec, 2)
        vmax_global  = np.percentile(spec, 99.5)
        spec_clipped = np.clip(spec, vmin_global, vmax_global)
        spec_norm    = (spec_clipped - vmin_global) / (vmax_global - vmin_global)      
        spec_vis     = spec_norm ** 0.6

        extent = [
            time_axis[0]  + self.t_marker,
            time_axis[-1] + self.t_marker,
            freqs_sliced[0],
            freqs_sliced[-1],
        ]

        self.ax_side.imshow(
            spec_vis,
            origin='lower',
            aspect='auto',
            cmap='gray_r',
            vmin=0,
            vmax=1,
            extent=extent,
        )
        self.ax_side.set_xlim(extent[0], extent[1])
        self.ax_side.set_ylim(freqs_sliced[0], freqs_sliced[-1])

        # Force top and bottom ticks to exactly match the zoom box bounds
        yticks = self.ax_side.get_yticks()
        yticks = yticks[(yticks > f_min_hz) & (yticks < f_max_hz)]  # keep only interior ticks
        yticks = np.concatenate([[f_min_hz], yticks, [f_max_hz]])   # add exact bounds
        self.ax_side.set_yticks(yticks)
        self.ax_side.set_yticklabels([f"{f:.0f}" for f in yticks])
        self.ax_side.set_xlabel("Time (s)")
        if hasattr(self.fig.canvas, "header_visible"):
            self.fig.canvas.header_visible = False
        self.fig.canvas.draw_idle()



    # ----------------------------
    # Annotation loading & drawing
    # ----------------------------

    def _load_existing_annotations(self):
        """Draw existing labels from label_rows, mapping Hz -> row coordinates."""
        if self.label_rows is None or self.label_rows.empty:
            return

        def _safe_round(value, ndigits):
            return round(value, ndigits) if value is not None else None

        for _, row in self.label_rows.iterrows():
            xmin     = round(row["Start Time (s)"], 1)
            xmax     = round(row["End Time (s)"], 1)
            min_hz = max(row["Low Freq (Hz)"], self.min_freq_hz)
            max_hz = max(row["High Freq (Hz)"], self.min_freq_hz)
            ymin_row = self.data.hz_to_row(min_hz)
            ymax_row = self.data.hz_to_row(max_hz)
            ebird_label = row["Label"]

            pwr_density = row.get('Avg Power Density (dB FS/Hz)')
            if pwr_density is None:
                pwr_density = self.data.avg_power_density(
                    min_hz, max_hz,
                    row["Start Time (s)"], row["End Time (s)"],
                )

            time_delta = row.get('Delta Time (s)') or _safe_round(
                row["End Time (s)"] - row["Start Time (s)"], 1
            )
            freq_delta = row.get('Delta Freq (Hz)') or _safe_round(
                row["High Freq (Hz)"] - row["Low Freq (Hz)"], 1
            )

            box_dict = {
                'Filename':                    row['Filename'],
                'Start Time (s)':              _safe_round(row["Start Time (s)"], 1),
                'End Time (s)':                _safe_round(row["End Time (s)"], 1),
                'Low Freq (Hz)':               _safe_round(min_hz, 0),
                'High Freq (Hz)':              _safe_round(max_hz, 0),
                'Delta Time (s)':              time_delta,
                'Delta Freq (Hz)':             freq_delta,
                'Avg Power Density (dB FS/Hz)': pwr_density,
                'Label':                       row.get('Label'),
                'Type':                        row.get('Type'),
                'Sex':                         row.get('Sex'),
                'Score':                       row.get('Score'),
                'Life Stage':                  row.get('Life Stage'),
            }

            rect = plt.Rectangle(
                (xmin, ymin_row), xmax - xmin, ymax_row - ymin_row,
                edgecolor='lime', facecolor='none', linewidth=2, zorder=10
            )
            self.ax_spec.add_patch(rect)

            y_offset = 0.01 * (self.ax_spec.get_ylim()[1] - self.ax_spec.get_ylim()[0])
            text = self.ax_spec.text(
                xmax, ymax_row + y_offset, ebird_label,
                color='white', fontsize=9,
                ha='right', va='bottom',
                clip_on=False, zorder=11,
            )

            self.annotations.add(box_dict, rect, text)

    # ----------------------------
    # Drag preview callbacks
    # ----------------------------
    
    def _event_to_data(self, event):
        """
        Convert mouse event to spectrogram data coordinates,
        always relative to ax_spec, even if cursor is over another axis.
        """
        # transform screen coordinates to data coordinates of ax_spec
        x, y = self.ax_spec.transData.inverted().transform((event.x, event.y))

        # clip vertical to spectrogram rows
        y = np.clip(y, 0, self.data.n_rows)

        # optional: allow small negative x overshoot
        x = max(x, -0.1 * self.data.duration_seconds)  # small left overshoot allowed

        return x, y
    
    def _snap_box(self, x0, y0, x1, y1):

        t_axis = self.data.time_axis
        freqs  = self.data.frequencies

        n_frames = len(t_axis)
        n_rows   = self.data.n_rows

        # --- snap using searchsorted (stable) ---
        xmin_idx = np.searchsorted(t_axis, x0)
        xmax_idx = np.searchsorted(t_axis, x1)

        xmin_idx = np.clip(xmin_idx, 0, n_frames - 1)
        xmax_idx = np.clip(xmax_idx, 0, n_frames - 1)

        ymin_idx = int(np.clip(round(y0), 0, n_rows - 1))
        ymax_idx = int(np.clip(round(y1), 0, n_rows - 1))

        xmin_idx, xmax_idx = sorted((xmin_idx, xmax_idx))
        ymin_idx, ymax_idx = sorted((ymin_idx, ymax_idx))

        # enforce minimum size
        if xmax_idx <= xmin_idx:
            xmax_idx = min(xmin_idx + 1, n_frames - 1)

        if ymax_idx <= ymin_idx:
            ymax_idx = min(ymin_idx + 1, n_rows - 1)

        xmin = float(t_axis[xmin_idx])
        xmax = float(t_axis[xmax_idx])

        ymin_hz = float(freqs[ymin_idx])
        ymax_hz = float(freqs[ymax_idx])

        return xmin, xmax, ymin_idx, ymax_idx, ymin_hz, ymax_hz

    def _on_drag_start(self, event):
        if event.inaxes != self.ax_spec:
            return
        if event.button not in (1, 3):
            return
        self._drag_button = event.button
        self._drag_start = self._event_to_data(event)
        self._drag_start_screen = (event.x, event.y)  # store screen coords too
        self._drag_moved = False


    def _on_drag_motion(self, event):

        if self._drag_start is None:
            return

        self._drag_moved = True

        x0, y0 = self._drag_start
        x1, y1 = self._event_to_data(event)

        xmin, xmax, ymin_idx, ymax_idx, _, _ = self._snap_box(x0, y0, x1, y1)

        if self._drag_rect is not None:
            self._drag_rect.remove()

        self._drag_rect = plt.Rectangle(
            (xmin, ymin_idx),
            xmax - xmin,
            ymax_idx - ymin_idx,
            edgecolor='cyan' if self._drag_button == 3 else 'lime',
            facecolor='none' if self._drag_button == 3 else 'lime',
            alpha=1.0 if self._drag_button == 3 else 0.25,
            linewidth=2,
            linestyle='--' if self._drag_button == 3 else '-',
            zorder=15,
        )

        self.ax_spec.add_patch(self._drag_rect)
        self.fig.canvas.draw_idle()

    def _update_band_power(self, ymin_hz, ymax_hz):
        """
        Compute 1D normalized power signal from the spectrogram within
        the frequency bounds of the last drawn box.
        """
        ymin_idx = int(np.floor(self.data.hz_to_row(ymin_hz)))
        ymax_idx = int(np.ceil(self.data.hz_to_row(ymax_hz)))

        # Sum across frequency band
        band_spec = self.data.mel_spec_db[ymin_idx:ymax_idx+1, :]
        power = band_spec.sum(axis=0)

        # Normalize 0..1
        power = (power - power.min()) / (power.max() - power.min() + 1e-12)

        self.band_power = power
        self.band_power_time_axis = self.data.time_axis

        if hasattr(self, 'power_line') and self.power_line is not None:
            self.power_line.set_ydata(power)
            self.power_line.set_xdata(self.band_power_time_axis)
            self.fig.canvas.draw_idle()


    def _on_drag_release(self, event):

        if self._drag_start is None:
            self._drag_moved = False
            self._drag_start = None
            self._drag_button = None
            return

        x0, y0 = self._drag_start
        drag_button = self._drag_button

        # --- Determine click vs drag from actual pixel distance, not motion events ---
        dx = abs(event.x - self._drag_start_screen[0])
        dy = abs(event.y - self._drag_start_screen[1])
        is_click = (dx < 5 and dy < 5)  # 5px tolerance

        x1, y1 = self._event_to_data(event)

        if drag_button == 3:
            if is_click:
                x_click, y_click = self._event_to_data(event)
                max_marker = self.data.duration_seconds - self.zoom_window_width
                t_marker   = np.clip(x_click, 0, max_marker)

                half_height = self.zoom_window_height * self.n_mels / 2
                f_marker    = np.clip(y_click, half_height, self.n_mels - half_height)

                self.t_marker = t_marker
                self.f_marker = f_marker

                self.centre_dot.set_data([self.t_marker], [self.f_marker])
                self._render_zoom()
                self.fig.canvas.draw_idle()

                if self.play_selected_on_right_click:
                    self.play_selected_section()
                else:
                    self.play_from_marker()
            else:
                xmin, xmax, _, _, ymin_hz, ymax_hz = self._snap_box(x0, y0, x1, y1)
                f_mid_row = float(self.data.hz_to_row((ymin_hz + ymax_hz) / 2))
                self.t_marker = float(np.clip(xmin, 0, self.data.duration_seconds))
                self.f_marker = float(np.clip(f_mid_row, 0, self.data.n_rows - 1))
                if self.centre_dot is not None:
                    self.centre_dot.set_data([self.t_marker], [self.f_marker])
                self._render_zoom()
                self.play_time_frequency_box(xmin=xmin, xmax=xmax, ymin_hz=ymin_hz, ymax_hz=ymax_hz)

            if self._drag_rect is not None:
                self._drag_rect.remove()
                self._drag_rect = None
            self._drag_start = None
            self._drag_moved = False
            self._drag_button = None
            self.fig.canvas.draw_idle()
            return

        if is_click:
            # treat as a click: place last-used box centred on cursor
            if self._last_box_time is None or self._last_box_rows is None:
                self._drag_start = None
                self._drag_moved = False
                self._drag_button = None
                return

            cx, cy = self._event_to_data(event)
            half_t = self._last_box_time / 2
            half_r = self._last_box_rows / 2
            x0, x1 = cx - half_t, cx + half_t
            y0, y1 = cy - half_r, cy + half_r

        else:
            # genuine drag: reject if below minimum size
            if abs(x1 - x0) < self.min_drag_time or abs(y1 - y0) < self.min_drag_rows:
                if self._drag_rect is not None:
                    self._drag_rect.remove()
                    self._drag_rect = None
                self._drag_start = None
                self._drag_moved = False
                self._drag_button = None
                self.fig.canvas.draw_idle()
                return

        xmin, xmax, ymin_idx, ymax_idx, ymin_hz, ymax_hz = \
            self._snap_box(x0, y0, x1, y1)

        # --- update band power line ---
        self._last_box_freq = (ymin_hz, ymax_hz)
        self._update_band_power(ymin_hz, ymax_hz)

        # remove preview rect
        if self._drag_rect is not None:
            self._drag_rect.remove()
            self._drag_rect = None

        # --- annotation metadata ---
        avg_pwr      = self.data.avg_power_density(ymin_hz, ymax_hz, xmin, xmax)
        common_label = self.annotation_state.current_label
        ebird_label  = self.common_to_ebird[common_label]

        box_dict = {
            'Filename':                     self.meta_row['filename'],
            'Start Time (s)':               xmin,
            'End Time (s)':                 xmax,
            'Low Freq (Hz)':                ymin_hz,
            'High Freq (Hz)':               ymax_hz,
            'Delta Time (s)':               round(xmax - xmin, 1),
            'Delta Freq (Hz)':              round(ymax_hz - ymin_hz, 0),
            'Avg Power Density (dB FS/Hz)': avg_pwr,
            'Label':                        ebird_label,
            'Type':                         self.annotation_state.call_type,
            'Sex':                          self.annotation_state.sex,
            'Score':                        self.annotation_state.score,
            'Life Stage':                   self.annotation_state.life_stage,
        }

        rect = plt.Rectangle(
            (xmin, ymin_idx),
            xmax - xmin,
            ymax_idx - ymin_idx,
            edgecolor='lime',
            facecolor='none',
            linewidth=2,
            zorder=10,
        )
        self.ax_spec.add_patch(rect)

        y_offset = 0.01 * (self.ax_spec.get_ylim()[1] - self.ax_spec.get_ylim()[0])
        text = self.ax_spec.text(
            xmax,
            ymax_idx + y_offset,
            ebird_label,
            color='white',
            fontsize=9,
            ha='right',
            va='bottom',
            clip_on=False,
            zorder=11,
        )

        self.annotations.add(box_dict, rect, text)

        self._drag_start = None
        self._drag_moved = False
        self._drag_button = None

        # store shape for future click-placement
        self._last_box_time = xmax - xmin
        self._last_box_rows = ymax_idx - ymin_idx

        self.fig.canvas.draw_idle()


    def _propagate_boxes_from_power(self):
        """
        Place copies of the last drawn box centred on local peaks in band power.
        Box size and frequency bounds are fixed from the last drawn box.
        If two peaks are closer than min_separation, extend the left box to cover both.
        """
        if self.data is None or self._last_box_freq is None or self._last_box_time is None:
            return

        ymin_hz, ymax_hz = self._last_box_freq
        half_t = self._last_box_time / 2

        # --- band-limited power ---
        ymin_idx = int(np.floor(self.data.hz_to_row(ymin_hz)))
        ymax_idx = int(np.ceil(self.data.hz_to_row(ymax_hz)))
        band_spec = self.data.mel_spec_db[ymin_idx:ymax_idx+1, :]
        band_power = band_spec.sum(axis=0)

        # --- normalize to 0..1 ---
        band_power = (band_power - band_power.min()) / (band_power.max() - band_power.min() + 1e-12)

        t_axis = self.data.time_axis

        # --- find local peaks above threshold ---
        thresh = band_power.mean() + 0.5 * band_power.std()
        frames_per_sec = len(t_axis) / self.data.duration_seconds
        min_distance = max(1, int(self._last_box_time * frames_per_sec / 2))

        peak_indices, _ = find_peaks(
            band_power,
            height=thresh,
            distance=min_distance,
        )

        # --- merge close peaks into time intervals ---
        # Each interval is (xmin, xmax) in seconds
        intervals = []
        for peak_idx in peak_indices:
            t_centre = float(t_axis[peak_idx])
            xmin = t_centre - half_t
            xmax = t_centre + half_t

            if xmin < 0 or xmax > self.data.duration_seconds:
                continue

            if intervals and (xmin - intervals[-1][1]) < self.min_separation:
                # too close to previous box — extend it rightward
                intervals[-1] = (intervals[-1][0], xmax)
            else:
                intervals.append((xmin, xmax))

        # --- place a box for each interval ---
        for xmin, xmax in intervals:
            xmin, xmax, ymin_idx_s, ymax_idx_s, ymin_hz_s, ymax_hz_s = \
                self._snap_box(xmin, ymin_idx, xmax, ymax_idx)

            avg_pwr      = self.data.avg_power_density(ymin_hz_s, ymax_hz_s, xmin, xmax)
            common_label = self.annotation_state.current_label
            ebird_label  = self.common_to_ebird[common_label]

            box_dict = {
                'Filename':                     self.meta_row['filename'],
                'Start Time (s)':               xmin,
                'End Time (s)':                 xmax,
                'Low Freq (Hz)':                ymin_hz_s,
                'High Freq (Hz)':               ymax_hz_s,
                'Delta Time (s)':               round(xmax - xmin, 1),
                'Delta Freq (Hz)':              round(ymax_hz_s - ymin_hz_s, 0),
                'Avg Power Density (dB FS/Hz)': avg_pwr,
                'Label':                        ebird_label,
                'Type':                         self.annotation_state.call_type,
                'Sex':                          self.annotation_state.sex,
                'Score':                        self.annotation_state.score,
                'Life Stage':                   self.annotation_state.life_stage,
            }

            rect = plt.Rectangle(
                (xmin, ymin_idx_s),
                xmax - xmin,
                ymax_idx_s - ymin_idx_s,
                edgecolor='cyan',
                facecolor='none',
                linewidth=2,
                linestyle='--',
                zorder=10,
            )
            self.ax_spec.add_patch(rect)

            y_offset = 0.01 * (self.ax_spec.get_ylim()[1] - self.ax_spec.get_ylim()[0])
            text = self.ax_spec.text(
                xmax,
                ymax_idx_s + y_offset,
                ebird_label,
                color='cyan',
                fontsize=9,
                ha='right',
                va='bottom',
                clip_on=False,
                zorder=11,
            )
            self.annotations.undo()
            self.annotations.add(box_dict, rect, text)
            

        self.fig.canvas.draw_idle()



    def _propagate_boxes_from_template(self):
        """
        Slide a window the same size as the seed patch along the band,
        compute normalised cosine similarity at each position,
        then place boxes at positions above a fraction of the peak score.
        """
        self._debug = {'reached': True}

        if self.data is None or self._last_box_freq is None or self._last_box_time is None:
            self._debug['failed_at'] = 'guard clause'
            return

        if not self.annotations.boxes:
            self._debug['failed_at'] = 'no annotation to use as seed'
            return

        # --- read seed coords BEFORE undoing ---
        last = self.annotations.boxes[-1]
        seed_start = last['Start Time (s)']
        seed_end   = last['End Time (s)']

        self.annotations.undo()

        # --- frequency band ---
        ymin_hz, ymax_hz = self._last_box_freq
        ymin_idx = int(np.floor(self.data.hz_to_row(ymin_hz)))
        ymax_idx = int(np.ceil(self.data.hz_to_row(ymax_hz)))
        band = self.data.mel_spec_db[ymin_idx:ymax_idx+1, :]  # (n_rows, n_frames)

        t_axis = self.data.time_axis

        # --- extract template ---
        seed_xmin_idx = int(np.clip(np.searchsorted(t_axis, seed_start), 0, band.shape[1] - 1))
        seed_xmax_idx = int(np.clip(np.searchsorted(t_axis, seed_end),   0, band.shape[1] - 1))
        template = band[:, seed_xmin_idx:seed_xmax_idx]
        w = template.shape[1]

        self._debug.update({
            'band_shape': band.shape,
            'template_shape': template.shape,
            'seed_xmin_idx': seed_xmin_idx,
            'seed_xmax_idx': seed_xmax_idx,
            'w': w,
        })

        if w == 0 or w >= band.shape[1]:
            self._debug['failed_at'] = 'template too wide or empty'
            return

        # --- normalise template ---
        t_flat = template.ravel()
        t_norm = (t_flat - t_flat.mean()) / (t_flat.std() + 1e-12)

        # --- slide window and compute cosine similarity at each position ---
        n_steps = band.shape[1] - w + 1
        scores = np.zeros(n_steps)
        for i in range(n_steps):
            window = band[:, i:i+w].ravel()
            w_norm = (window - window.mean()) / (window.std() + 1e-12)
            scores[i] = np.dot(t_norm, w_norm) / len(t_norm)

        # --- mask out seed position so it doesn't set the threshold ---
        scores_masked = scores.copy()
        scores_masked[seed_xmin_idx:seed_xmin_idx + w] = 0.0

        peak_score = scores_masked.max()
        thresh = peak_score * self.similarness_threshold
        half_w = w // 2
        min_distance = max(1, half_w)

        self._debug.update({
            'n_steps': n_steps,
            'scores_min': scores.min(),
            'scores_max': scores.max(),
            'scores_mean': scores.mean(),
            'score_at_seed': scores[seed_xmin_idx],
            'peak_score': peak_score,
            'thresh': thresh,
            'min_distance': min_distance,
        })

        peak_indices, _ =find_peaks(
            scores_masked,
            height=thresh,
            distance=min_distance,
        )

        self._debug['peak_indices'] = peak_indices

        # --- build intervals ---
        half_t = self._last_box_time / 2
        intervals = []
        last_centre = -self.min_separation
        for peak_idx in peak_indices:
            centre_frame = peak_idx + half_w
            t_centre = float(t_axis[min(centre_frame, len(t_axis) - 1)])
            xmin = t_centre - half_t
            xmax = t_centre + half_t

            xmin = np.max([0, xmin])
            xmax = np.min([self.data.duration_seconds, xmax])

            min_sep = max([self.min_separation, 2*half_t])

            if intervals and (t_centre - last_centre) < min_sep:
                intervals[-1] = (intervals[-1][0], xmax)
            else:
                intervals.append((xmin, xmax))
            last_centre = t_centre

        self._debug['intervals'] = intervals

        # --- draw boxes ---
        for xmin, xmax in intervals:
            xmin, xmax, ymin_idx_s, ymax_idx_s, ymin_hz_s, ymax_hz_s = \
                self._snap_box(xmin, ymin_idx, xmax, ymax_idx)

            avg_pwr      = self.data.avg_power_density(ymin_hz_s, ymax_hz_s, xmin, xmax)
            common_label = self.annotation_state.current_label
            ebird_label  = self.common_to_ebird[common_label]

            box_dict = {
                'Filename':                     self.meta_row['filename'],
                'Start Time (s)':               xmin,
                'End Time (s)':                 xmax,
                'Low Freq (Hz)':                ymin_hz_s,
                'High Freq (Hz)':               ymax_hz_s,
                'Delta Time (s)':               round(xmax - xmin, 1),
                'Delta Freq (Hz)':              round(ymax_hz_s - ymin_hz_s, 0),
                'Avg Power Density (dB FS/Hz)': avg_pwr,
                'Label':                        ebird_label,
                'Type':                         self.annotation_state.call_type,
                'Sex':                          self.annotation_state.sex,
                'Score':                        self.annotation_state.score,
                'Life Stage':                   self.annotation_state.life_stage,
            }

            rect = plt.Rectangle(
                (xmin, ymin_idx_s),
                xmax - xmin,
                ymax_idx_s - ymin_idx_s,
                edgecolor='orange',
                facecolor='none',
                linewidth=2,
                linestyle='--',
                zorder=10,
            )
            self.ax_spec.add_patch(rect)

            y_offset = 0.01 * (self.ax_spec.get_ylim()[1] - self.ax_spec.get_ylim()[0])
            text = self.ax_spec.text(
                xmax, ymax_idx_s + y_offset, ebird_label,
                color='orange', fontsize=9,
                ha='right', va='bottom',
                clip_on=False, zorder=11,
            )

            self.annotations.add(box_dict, rect, text)

        self.fig.canvas.draw_idle()




    # ----------------------------
    # Selection callback
    # ----------------------------


    def _on_select(self, eclick, erelease):

        if self.data is None:
            return

        # --- raw cursor coordinates ---
        xmin_raw, ymin_raw = self._event_to_data(eclick)
        xmax_raw, ymax_raw = self._event_to_data(erelease)

        # --- clip to spectrogram bounds ---
        xmin_raw = np.clip(xmin_raw, 0, self.data.duration_seconds)
        xmax_raw = np.clip(xmax_raw, 0, self.data.duration_seconds)

        ymin_raw = np.clip(ymin_raw, 0, self.data.n_rows - 1)
        ymax_raw = np.clip(ymax_raw, 0, self.data.n_rows - 1)

        # --- snap to grid ---
        xmin, xmax, ymin_idx, ymax_idx, ymin_hz, ymax_hz = \
            self._snap_box(xmin_raw, ymin_raw, xmax_raw, ymax_raw)

        # --- annotation metadata ---
        avg_pwr      = self.data.avg_power_density(ymin_hz, ymax_hz, xmin, xmax)
        common_label = self.annotation_state.current_label
        ebird_label  = self.common_to_ebird[common_label]

        box_dict = {
            'Filename':                     self.meta_row['filename'],
            'Start Time (s)':               xmin,
            'End Time (s)':                 xmax,
            'Low Freq (Hz)':                ymin_hz,
            'High Freq (Hz)':               ymax_hz,
            'Delta Time (s)':               round(xmax - xmin,       1),
            'Delta Freq (Hz)':              round(ymax_hz - ymin_hz, 0),
            'Avg Power Density (dB FS/Hz)': avg_pwr,
            'Label':                        ebird_label,
            'Type':                         self.annotation_state.call_type,
            'Sex':                          self.annotation_state.sex,
            'Score':                        self.annotation_state.score,
            'Life Stage':                   self.annotation_state.life_stage,
        }

        # --- draw rectangle ---
        rect = plt.Rectangle(
            (xmin, ymin_idx),
            xmax - xmin,
            ymax_idx - ymin_idx,
            edgecolor='lime',
            facecolor='none',
            linewidth=2,
            zorder=10,
        )
        self.ax_spec.add_patch(rect)

        # --- label ---
        y_offset = 0.01 * (self.ax_spec.get_ylim()[1] - self.ax_spec.get_ylim()[0])
        text = self.ax_spec.text(
            xmax,
            ymax_idx + y_offset,
            ebird_label,
            color='white',
            fontsize=9,
            ha='right',
            va='bottom',
            clip_on=False,
            zorder=11,
        )

        # --- store annotation ---
        self.annotations.add(box_dict, rect, text)

        self.fig.canvas.draw_idle()


    # ----------------------------
    # Click / keypress callbacks
    # ----------------------------

    def _on_click(self, event):
        # Right-click handling is done in drag-release so we can distinguish click vs drag.
        return


    def _on_keypress(self, event):
        print(f"Some key was pressed: {event.key}")
        if event.key == 'd':
            self.annotations.clear()
            self.fig.canvas.draw_idle()
        elif event.key == 'u':
            if self.annotations.undo():
                self.fig.canvas.draw_idle()
        elif event.key == 'b':
            self._propagate_boxes_from_power()
            self.fig.canvas.draw_idle()
        elif event.key == 't':
            self._propagate_boxes_from_template()
            self.fig.canvas.draw_idle()

    # ----------------------------
    # Public utilities
    # ----------------------------

    def clear_annotations(self):
        self.annotations.clear()
        self.fig.canvas.draw_idle()

    def play_from_marker(self):
        if self.data is None:
            return
        start_time = float(np.clip(self.t_marker, 0, self.data.duration_seconds))
        self._play_audio(start_time=start_time)

    def play_selected_section(self):
        if self.data is None:
            return
        start_time = float(np.clip(self.t_marker, 0, self.data.duration_seconds))
        end_time = min(start_time + self.zoom_window_width, self.data.duration_seconds)
        self._play_audio(start_time=start_time, end_time=end_time)

    def play_time_frequency_box(self, xmin: float, xmax: float, ymin_hz: float, ymax_hz: float):
        if self.data is None:
            return
        if xmax <= xmin:
            return

        duration = max(0.01, xmax - xmin)
        wav_segment, new_sr = zoom_in_on_wav(
            self.data.wav,
            x_left=xmin,
            f_min_hz=ymin_hz,
            f_max_hz=ymax_hz,
            times=self.data.time_axis,
            window_width=duration,
            sr=self.data.sr,
        )
        if wav_segment.size == 0:
            return

        if self.audio_output is not None:
            with self.audio_output:
                self.audio_output.clear_output(wait=True)
                play_audio_standalone(
                    wav_segment=wav_segment,
                    sr=new_sr,
                    peak_reference=self._audio_peak_reference,
                )
        else:
            play_audio_standalone(
                wav_segment=wav_segment,
                sr=new_sr,
                peak_reference=self._audio_peak_reference,
            )
        self._start_playhead(start_time=xmin, end_time=xmax)

    def stop_audio(self):
        sd.stop()
        self._playhead_gen = getattr(self, "_playhead_gen", 0) + 1
        self._stop_playhead_timer()

    def get_boxes(self):
        return self.annotations.boxes

    def close(self):
        plt.close(self.fig)


class AnnotationSession:
    """
    Stateful, crash-resilient annotation manager.

    Backed by a single parquet file with a 'status' column.
    """

    def __init__(self,
                 df_meta: pd.DataFrame,
                 df_labels: pd.DataFrame,
                 new_meta_filepath: Path,
                 new_labels_filepath: Path,
                 reviewer: str,
                 author: str,
                 label_schema: DataFrameSchema = _label_schema,
                 metadata_schema: DataFrameSchema = _metadata_schema,
                 sort_by=None, 
                 ):

        self.new_labels_filepath = Path(new_labels_filepath)
        self.new_meta_filepath = Path(new_meta_filepath)
        self.reviewer = reviewer
        self.author = author
        self.label_schema = label_schema
        self.meta_schema = metadata_schema

        self.label_headers = label_schema.headers
        self.meta_headers = self.meta_schema.headers

        #label_headers = ['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',
        #                 'Label', 'Type', 'Sex', 'Score', 'Life Stage', 'Indv ID', 'Delta Time (s)',
        #                 'Delta Freq (Hz)', 'Avg Power Density (dB FS/Hz)']

        #meta_headers = ['filename', 'collection', 'primary_label', 'secondary_labels', 'url', 'latitude', 
        #                'longitude', 'author', 'license', 'recorded_on', 'reviewed_by', 'reviewed_on', 
        #                'source_filename', 'source_start_s', 'source_end_s', 'models_used']

        # Copy to avoid modifying original and coerce metadata dtypes
        self.df_meta = df_meta.copy()
        for col in self.meta_headers:
            if col not in self.df_meta.columns:
                self.df_meta[col] = pd.NA
        self.df_meta = self.meta_schema.apply(self.df_meta)
        self.df_meta["status"] = "pending"

        self.df_labels = df_labels.copy() if df_labels is not None else pd.DataFrame(columns=self.label_headers)

        # Restore completed files if saved
        done_filenames = []
        if self.new_meta_filepath.exists():
            df_meta_done = pd.read_parquet(self.new_meta_filepath)
            done_filenames = df_meta_done['filename']
            mask = self.df_meta['filename'].isin(done_filenames)
            self.df_meta.loc[mask, 'status'] = 'done'


        # Restore completed labels if saved (robust merge)
        if self.new_labels_filepath.exists():
            df_labels_done = pd.read_parquet(self.new_labels_filepath)

            # Identify completed files (from saved metadata if available)
            if self.new_meta_filepath.exists():
                done_files = done_filenames
            else:
                done_files = df_labels_done['Filename'].unique()

            # Remove any existing labels for completed files from base labels
            self.df_labels = self.df_labels[~self.df_labels['Filename'].isin(done_files)]

            # Append saved annotations (source of truth)
            if not df_labels_done.empty:
                for col in self.label_headers:
                    if col not in df_labels_done.columns:
                        df_labels_done[col] = pd.NA

                df_labels_done = df_labels_done[self.label_headers]

                # force dtype stability for all-NA columns
                for col in df_labels_done.columns:
                    if df_labels_done[col].isna().all():
                        df_labels_done[col] = df_labels_done[col].astype("object")



                if self.df_labels.empty:
                    self.df_labels = df_labels_done.copy()
                else:
                    df_labels_done = normalise_labels_df(df_labels_done, self.label_headers)
                    self.df_labels = normalise_labels_df(self.df_labels, self.label_headers)

                    self.df_labels = pd.concat(
                        [self.df_labels, df_labels_done],
                        axis=0,
                        ignore_index=True
                    )

        # Optional sorting
        if sort_by is not None:
            self.df_meta = self.df_meta.sort_values(sort_by)

        # Maintain original dataset order
        self.df_meta = self.df_meta.set_index('filename', drop=False)
        self._current_index = None
        self.advance_pointer()

        # Keep a stack of completed files for undo
        self._completed_stack = []

    # -----------------------------
    # Core navigation
    # -----------------------------
    def advance_pointer(self):
        pending = self.df_meta[self.df_meta.status == "pending"]
        self._current_index = pending.index[0] if not pending.empty else None

    @property
    def current(self):
        if self._current_index is None:
            return None, None

        filename = self._current_index
        current_row = self.df_meta.loc[filename]

        # Get labels safely
        current_labels = self.df_labels[self.df_labels['Filename'] == filename]

        return current_row, current_labels

    @property
    def finished(self):
        return self.df_meta[self.df_meta.status == "done"]

    @property
    def remaining(self):
        return self.df_meta[self.df_meta.status == "pending"]

    # -----------------------------
    # State transitions
    # -----------------------------
    def skip_row(self):
        if self._current_index is None:
            raise RuntimeError("No pending rows left.")
        self.df_meta.at[self._current_index, "status"] = "skipped"
        self.advance_pointer()

    def complete(self, boxes: list[dict] | None = None):
        """
        Mark current file as done and replace its label rows
        with the provided annotation boxes.
        """

        if self._current_index is None:
            raise RuntimeError("No pending rows left.")

        file_idx = self._current_index

        # ---- Update labels ----
        if boxes:
            new_labels = pd.DataFrame(boxes)
            new_labels['Filename'] = file_idx

            new_labels = normalise_labels_df(new_labels, self.label_headers)
            self.df_labels = normalise_labels_df(self.df_labels, self.label_headers)

            self.df_labels = pd.concat(
                [self.df_labels, new_labels],
                axis=0,
                ignore_index=True
                )

        # ---- Update metadata ----
        self.df_meta.at[file_idx, "status"] = "done"
        self.df_meta.at[file_idx, "author"] = self.author
        self.df_meta.at[file_idx, "reviewed_on"] = pd.Timestamp(date.today())
        self.df_meta.at[file_idx, "reviewed_by"] = self.reviewer

        # ---- Persist only completed rows ----
        df_meta_done = self.df_meta.loc[self.df_meta["status"] == "done", self.meta_headers]
        df_labels_done = self.df_labels[self.df_labels['Filename'].isin(df_meta_done['filename'])]

        self._save(self.new_meta_filepath, df_meta_done)
        self._save(self.new_labels_filepath, df_labels_done)

        # Push to completed stack for undo
        self._completed_stack.append(file_idx)

        self.advance_pointer()

    # -----------------------------
    # Undo last annotation
    # -----------------------------
    def undo_last(self):
        if not self._completed_stack:
            print("No completed files to undo.")
            return

        last_file = self._completed_stack.pop()

        # Reset metadata status
        self.df_meta.at[last_file, "status"] = "pending"
        self.df_meta.at[last_file, "reviewed_on"] = pd.NaT
        self.df_meta.at[last_file, "reviewed_by"] = pd.NA

        # Remove labels for that file
        self.df_labels = self.df_labels[self.df_labels['Filename'] != last_file]

        # Persist updated files
        df_meta_done = self.df_meta.loc[self.df_meta["status"] == "done", self.meta_headers]
        df_labels_done = self.df_labels[self.df_labels['Filename'].isin(df_meta_done['filename'])]

        self._save(self.new_meta_filepath, df_meta_done)
        self._save(self.new_labels_filepath, df_labels_done)

        # Re-position pointer to undone file
        self._current_index = last_file

    # -----------------------------
    # Persistence
    # -----------------------------
    def _save(self, filepath, df):
        tmp = filepath.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(filepath)

    # -----------------------------
    # Info
    # -----------------------------
    def summary(self):
        total = len(self.df_meta)
        total_boxes = len(self.df_labels)
        done = (self.df_meta["status"] == "done").sum()
        pending = (self.df_meta["status"] == "pending").sum()
        finished_files = pd.read_parquet(self.new_meta_filepath)["filename"].nunique() if self.new_meta_filepath.exists() else 0

        return {
            "total_files": total,
            "finished_files_in_new_meta": finished_files,
            "total_annotations": total_boxes,
            "done_in_current_session": done,
            "pending_in_current_session": pending,
            "total_minus_pending": total - pending,
        }



def load_current_sample(session, annotator, paths, map_widget):
    meta_row, label_rows = session.current

    if meta_row is None:
        print("No current sample.")
        return

    filepath = paths.audio_folder / meta_row['filename']
    session.current_filepath = filepath

    annotator.load_file(filepath, meta_row, label_rows)

    annotation_state = annotator.annotation_state

    # --- derive primary class from folder ---
    folder = Path(filepath).parent.name
    primary_cls = annotation_state.ebird_to_common.get(folder)

    # --- derive secondary classes from metadata ---
    secondary_ebirds = meta_row.get('secondary_labels', [])

    # ensure it's a list (handle NaN / string edge cases)
    if isinstance(secondary_ebirds, str):
        # optional: only if stored as stringified list
        
        try:
            secondary_ebirds = ast.literal_eval(secondary_ebirds)
        except Exception:
            secondary_ebirds = []
    elif secondary_ebirds is None:
        secondary_ebirds = []

    # map to common names, filtering unknowns
    secondary_classes = [
        annotation_state.ebird_to_common[e]
        for e in secondary_ebirds
        if e in annotation_state.ebird_to_common
    ]

    # --- merge all classes into visible list ---
    visible = annotation_state.get_visible_classes()

    def add_if_missing(cls):
        if cls is not None and cls not in visible:
            visible.append(cls)

    # add primary
    add_if_missing(primary_cls)

    # add secondary
    for cls in secondary_classes:
        add_if_missing(cls)

    # --- update visible classes if changed ---
    annotation_state.set_visible_classes(visible)

    if hasattr(annotation_state, 'radio_class') and annotation_state.radio_class is not None:
        # update radio options
        options = [
            (f"{c} ({annotation_state.common_to_ebird.get(c, '')})", c)
            for c in visible
        ]
        annotation_state.radio_class.options = options

        # --- safely assign selected value (primary only) ---
        if primary_cls is not None:
            def set_value():
                annotation_state.radio_class.value = primary_cls
                annotation_state.current_label = primary_cls

            Timer(0.01, set_value).start()
    elif primary_cls is not None:
        annotation_state.current_label = primary_cls

    # --- update map ---
    map_widget.update(
        lat=meta_row.at['latitude'],
        lon=meta_row.at['longitude']
    )

    return annotator.display()


class AnnotationControls:

    def __init__(self):
        self.session = None
        self.annotator = None
        self.paths = None
        self.map = None

        # Buttons
        self.load_btn = widgets.Button(
            description="Reload",
            button_style='primary',
            icon='refresh'
        )

        self.next_btn = widgets.Button(
            description="Next",
            button_style='success',
            icon='arrow-right'
        )

        self.skip_btn = widgets.Button(
            description="Skip",
            button_style='warning',
            icon='arrow-right'
        )

        self.undo_btn = widgets.Button(
            description="Undo Last",
            button_style='danger',
            icon='undo'
        )

        # Hook callbacks
        self.load_btn.on_click(self._on_load_clicked)
        self.next_btn.on_click(self._on_next_clicked)
        self.skip_btn.on_click(self._on_skip_clicked)
        self.undo_btn.on_click(self._on_undo_clicked)

        # Container for buttons (add undo button)
        self.controls = widgets.HBox([self.load_btn, self.skip_btn, self.next_btn, self.undo_btn])

        # Output widget for session summary
        self.summary_out = widgets.Output()

        # Combine controls + summary display
        self.container = widgets.VBox([self.controls, self.summary_out])

    # ---- Dependency Injection ----
    def bind(self, session, annotator, paths, map_obj):
        self.session = session
        self.annotator = annotator
        self.paths = paths
        self.map = map_obj

        # Display initial summary if session already loaded
        self._update_summary()

    # ---- Callbacks ----
    def _on_load_clicked(self, b):
        if self.session is None:
            return
        load_current_sample(self.session, self.annotator, self.paths, self.map)
        self._update_summary()

    def _on_skip_clicked(self, b):
        if self.session is None:
            return
        self.session.skip_row()
        load_current_sample(self.session, self.annotator, self.paths, self.map)
        self._update_summary()

    def _on_next_clicked(self, b=None):
        if self.session is None:
            return
        # Complete current file
        if self.annotator is not None:
            self.session.complete(self.annotator.get_boxes())
        load_current_sample(self.session, self.annotator, self.paths, self.map)
        self._update_summary()

    def _on_undo_clicked(self, b):
        if self.session is None:
            return
        self.session.undo_last()
        load_current_sample(self.session, self.annotator, self.paths, self.map)
        self._update_summary()

    # ---- Update summary display ----
    def _update_summary(self):
        if self.session is None:
            return
        with self.summary_out:
            self.summary_out.clear_output(wait=True)
            s = self.session.summary()
            df = pd.DataFrame([{
                "Finished (new)": s["finished_files_in_new_meta"],
                "Done": s["done_in_current_session"],
                "Pending": s["pending_in_current_session"],
                "Total": s["total_files"],
            }])
            display(df.style.hide(axis="index"))

    # ---- Display everything ----
    def display(self):
        display(self.container)


class MiniBirdNamer:
    '''Handles bird name conversions from a csv file with an eBird column and a CommonName column
       This is a cut-down version of BirdNamer from WildPyTools'''
    def __init__(self,
                 naming_csv_path: Path,
                 common_col_name: str='CommonName',
                 ebird_col_name: str='eBird',
                 ):
        _mapping_df = pd.read_csv(naming_csv_path)
        self.common_to_ebird_dict = dict(zip(_mapping_df[common_col_name], _mapping_df[ebird_col_name]))
        self.e_names = list(set(_mapping_df[ebird_col_name]))
        self.common_names = list(set(_mapping_df[common_col_name]))