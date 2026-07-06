from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
import matplotlib.pyplot as plt

from anqa.annotation import (
    AnnotationSession,
    AnnotationState,
    FastMap,
    MiniBirdNamer,
    SpectrogramAnnotator,
    cx,
    load_current_sample,
    normalize_secondary_labels,
)


@dataclass
class SessionPaths:
    source_dataset: Path
    audio_folder: Path
    reviewed_dataset: Path
    naming_csv: Path

    @property
    def original_labels(self) -> Path:
        return self.source_dataset / "annotations.parquet"

    @property
    def original_metadata(self) -> Path:
        return self.source_dataset / "metadata.parquet"

    @property
    def out_labels(self) -> Path:
        return self.reviewed_dataset / "annotations.parquet"

    @property
    def out_metadata(self) -> Path:
        return self.reviewed_dataset / "metadata.parquet"


class _MapPlaceholder:
    def update(self, lat=None, lon=None):
        return None


class AnnotationDesktopWindow(tk.Toplevel):
    SEX_OPTIONS = ("Leave Empty", "Male", "Female")
    LIFE_STAGE_OPTIONS = ("Leave Empty", "Juvenile", "Adult")
    CALL_TYPE_OPTIONS = (
        "Leave Empty",
        "Call",
        "Song",
        "Alarm",
        "Duet",
        "Begging",
        "Flight call",
        "Echolocation",
        "Other",
    )
    SCORE_OPTIONS = ("Leave Empty", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")

    def __init__(self, master: tk.Tk, paths: SessionPaths, author: str | None, reviewer: str | None):
        super().__init__(master)
        self.title("Anqa Annotator")
        self.geometry("940x640")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_session(paths=paths, author=author, reviewer=reviewer)
        self._build_controls()
        self._load_current_sample()

    def _build_session(self, paths: SessionPaths, author: str | None, reviewer: str | None):
        df_meta = pd.read_parquet(paths.original_metadata)
        if "secondary_labels" in df_meta.columns:
            df_meta["secondary_labels"] = df_meta["secondary_labels"].apply(normalize_secondary_labels)

        if paths.original_labels.exists():
            df_labels = pd.read_parquet(paths.original_labels)
        else:
            df_labels = pd.DataFrame(columns=["Filename"])

        namer = MiniBirdNamer(paths.naming_csv)
        all_classes = sorted(set(namer.common_names))
        max_visible = len(all_classes)

        annotation_state = AnnotationState(all_classes=all_classes, max_visible=max_visible)
        default_class = "Unknown" if "Unknown" in all_classes else (all_classes[0] if all_classes else None)
        if default_class is not None:
            annotation_state.set_visible_classes([default_class])
            annotation_state.current_label = default_class
        else:
            annotation_state.set_visible_classes([])
        annotation_state.common_to_ebird = namer.common_to_ebird_dict
        annotation_state.ebird_to_common = {v: k for k, v in namer.common_to_ebird_dict.items()}

        self.paths = paths
        provider = cx.providers.Esri.WorldImagery if cx is not None else None
        try:
            self.map_widget = FastMap(provider=provider)
            self.map_widget.display()
        except Exception:
            self.map_widget = _MapPlaceholder()

        self.annotation_state = annotation_state
        self.annotator = SpectrogramAnnotator(
            annotation_state=annotation_state,
            common_to_ebird=namer.common_to_ebird_dict,
            plot_size=(16, 4),
            f_min=20,
            f_max=16000,
            zoom_window_height=0.4,
            zoom_window_width=5,
            min_drag_rows=5,
            min_drag_time_s=0.1,
            min_separation=2,
            similarness_threshold=0.5,
            min_freq_hz=300,
        )
        self.session = AnnotationSession(
            df_meta=df_meta,
            df_labels=df_labels,
            new_meta_filepath=paths.out_metadata,
            new_labels_filepath=paths.out_labels,
            reviewer=reviewer,
            author=author,
        )

        self.annotator.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _build_controls(self):
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)

        self.current_file_var = tk.StringVar(value="Current: -")
        ttk.Label(frame, textvariable=self.current_file_var).pack(anchor="w")

        self.summary_var = tk.StringVar(value="Finished: 0 | Pending: 0 | Total: 0")
        ttk.Label(frame, textvariable=self.summary_var).pack(anchor="w", pady=(4, 10))

        ttk.Label(frame, text="New Class").pack(anchor="w")
        self.label_combo = ttk.Combobox(
            frame,
            values=self.annotation_state.get_visible_classes(),
            state="readonly",
            width=80,
        )
        self.label_combo.bind("<<ComboboxSelected>>", self._on_label_changed)
        self.label_combo.pack(fill="x", pady=(0, 10))

        class_frame = ttk.LabelFrame(frame, text="Classes")
        class_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.class_listbox = tk.Listbox(
            class_frame,
            selectmode="multiple",
            exportselection=False,
            height=10,
        )
        self.class_listbox.pack(fill="both", expand=True, padx=8, pady=(8, 6))
        for cls in self.annotation_state.get_all_classes():
            self.class_listbox.insert("end", cls)
        self.class_listbox.bind("<<ListboxSelect>>", self._on_visible_classes_changed)

        meta_frame = ttk.LabelFrame(frame, text="Annotation metadata")
        meta_frame.pack(fill="x", pady=(0, 10))

        self.sex_combo = self._meta_combobox(meta_frame, "Sex", self.SEX_OPTIONS, self._on_sex_changed)
        self.life_stage_combo = self._meta_combobox(
            meta_frame, "Life Stage", self.LIFE_STAGE_OPTIONS, self._on_life_stage_changed
        )
        self.call_type_combo = self._meta_combobox(
            meta_frame, "Call Type", self.CALL_TYPE_OPTIONS, self._on_call_type_changed
        )
        self.score_combo = self._meta_combobox(meta_frame, "Score", self.SCORE_OPTIONS, self._on_score_changed)

        audio_row = ttk.Frame(frame)
        audio_row.pack(fill="x", pady=(0, 10))
        ttk.Button(audio_row, text="Play from cursor", command=self._play_from_cursor).pack(side="left", padx=(0, 6))
        ttk.Button(audio_row, text="Play selected section", command=self._play_selected_section).pack(side="left", padx=(0, 6))
        ttk.Button(audio_row, text="Stop audio", command=self._stop_audio).pack(side="left", padx=(0, 12))

        self.play_selected_on_rclick_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            audio_row,
            text="Right-click plays selected section",
            variable=self.play_selected_on_rclick_var,
            command=self._on_audio_options_changed,
        ).pack(side="left")

        button_row = ttk.Frame(frame)
        button_row.pack(fill="x")

        ttk.Button(button_row, text="Reload", command=self._load_current_sample).pack(side="left", padx=(0, 6))
        ttk.Button(button_row, text="Skip", command=self._skip_current).pack(side="left", padx=(0, 6))
        ttk.Button(button_row, text="Next", command=self._complete_current).pack(side="left", padx=(0, 6))
        ttk.Button(button_row, text="Undo Last", command=self._undo_last_file).pack(side="left")

        ttk.Label(
            frame,
            text="Shortcuts: Enter/Space = Next, N = Skip, Ctrl+Z = Undo last file",
        ).pack(anchor="w", pady=(10, 0))
        ttk.Label(
            frame,
            text="Use right-click on spectrogram to set cursor/zoom. Map opens in its own window.",
        ).pack(anchor="w", pady=(2, 0))

        self._sync_visible_class_listbox()
        self._set_metadata_controls()
        self._on_audio_options_changed()

    def _meta_combobox(self, parent, label, values, callback):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=16).pack(side="left")
        combo = ttk.Combobox(row, values=values, state="readonly", width=25)
        combo.set(values[0])
        combo.bind("<<ComboboxSelected>>", callback)
        combo.pack(side="left", fill="x", expand=True)
        return combo

    def _on_label_changed(self, _event=None):
        selected = self.label_combo.get()
        if selected:
            self.annotation_state.set_label(selected)

    def _on_sex_changed(self, _event=None):
        value = self.sex_combo.get()
        self.annotation_state.sex = None if value == "Leave Empty" else value

    def _on_life_stage_changed(self, _event=None):
        value = self.life_stage_combo.get()
        self.annotation_state.life_stage = None if value == "Leave Empty" else value

    def _on_call_type_changed(self, _event=None):
        value = self.call_type_combo.get()
        self.annotation_state.call_type = None if value == "Leave Empty" else value

    def _on_score_changed(self, _event=None):
        value = self.score_combo.get()
        self.annotation_state.score = None if value == "Leave Empty" else float(value)

    def _set_metadata_controls(self):
        self.sex_combo.set(self.annotation_state.sex or "Leave Empty")
        self.life_stage_combo.set(self.annotation_state.life_stage or "Leave Empty")
        self.call_type_combo.set(self.annotation_state.call_type or "Leave Empty")
        self.score_combo.set("Leave Empty" if self.annotation_state.score is None else str(self.annotation_state.score))

    def _set_label_value(self):
        self.label_combo["values"] = self.annotation_state.get_visible_classes()
        label = self.annotation_state.current_label
        if not label:
            return
        values = list(self.label_combo.cget("values"))
        if label not in values:
            values.append(label)
            self.label_combo["values"] = values
        self.label_combo.set(label)

    def _sync_visible_class_listbox(self):
        visible = set(self.annotation_state.get_visible_classes())
        self.class_listbox.selection_clear(0, "end")
        for i, cls in enumerate(self.annotation_state.get_all_classes()):
            if cls in visible:
                self.class_listbox.selection_set(i)

    def _on_visible_classes_changed(self, _event=None):
        selected_indices = self.class_listbox.curselection()
        selected = [self.class_listbox.get(i) for i in selected_indices]
        if not selected:
            selected = ["Unknown"] if "Unknown" in self.annotation_state.get_all_classes() else []
            if not selected and self.annotation_state.get_all_classes():
                selected = [self.annotation_state.get_all_classes()[0]]
        self.annotation_state.set_visible_classes(selected)
        self._set_label_value()
        self._sync_visible_class_listbox()

    def _update_summary(self):
        summary = self.session.summary()
        self.summary_var.set(
            f"Finished: {summary['finished_files_in_new_meta']} | "
            f"Done: {summary['done_in_current_session']} | "
            f"Pending: {summary['pending_in_current_session']} | "
            f"Total: {summary['total_files']}"
        )

    def _load_current_sample(self):
        meta_row, _ = self.session.current
        if meta_row is None:
            self.current_file_var.set("Current: all files finished")
            self._update_summary()
            return

        load_current_sample(self.session, self.annotator, self.paths, self.map_widget)
        self.current_file_var.set(f"Current: {meta_row['filename']}")
        self._sync_visible_class_listbox()
        self._set_label_value()
        self._set_metadata_controls()
        self._update_summary()

    def _complete_current(self):
        try:
            self.session.complete(self.annotator.get_boxes())
        except RuntimeError as exc:
            messagebox.showinfo("Session complete", str(exc), parent=self)
        self._load_current_sample()

    def _skip_current(self):
        try:
            self.session.skip_row()
        except RuntimeError as exc:
            messagebox.showinfo("Session complete", str(exc), parent=self)
        self._load_current_sample()

    def _undo_last_file(self):
        self.session.undo_last()
        self._load_current_sample()

    def _play_from_cursor(self):
        self.annotator.play_from_marker()

    def _play_selected_section(self):
        self.annotator.play_selected_section()

    def _stop_audio(self):
        self.annotator.stop_audio()

    def _on_audio_options_changed(self):
        self.annotator.play_selected_on_right_click = self.play_selected_on_rclick_var.get()

    def _on_key_press(self, event):
        key = (event.key or "").lower()
        if key in {" ", "enter"}:
            self._complete_current()
        elif key == "n":
            self._skip_current()
        elif key == "ctrl+z":
            self._undo_last_file()

    def _on_close(self):
        try:
            self.annotator.stop_audio()
            self.annotator.close()
            if hasattr(self.map_widget, "fig"):
                plt.close(self.map_widget.fig)
        finally:
            plt.close("all")
            self.destroy()


class LauncherApp(tk.Tk):
    SETTINGS_PATH = Path.home() / ".anqa" / "launcher-settings.json"

    def __init__(self):
        super().__init__()
        self.title("Anqa Launcher")
        self.geometry("760x310")
        self.resizable(False, False)
        self._build_form()

    def _build_form(self):
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)

        self.source_var = tk.StringVar()
        self.audio_var = tk.StringVar()
        self.reviewed_var = tk.StringVar()
        self.naming_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self.reviewer_var = tk.StringVar()
        self._load_saved_paths()

        self._path_row(frame, "Source dataset folder", self.source_var, folder=True)
        self._path_row(frame, "Audio folder", self.audio_var, folder=True)
        self._path_row(frame, "Reviewed output folder", self.reviewed_var, folder=True)
        self._path_row(frame, "Naming CSV", self.naming_var, folder=False)

        self._entry_row(frame, "Author", self.author_var)
        self._entry_row(frame, "Reviewer", self.reviewer_var)

        ttk.Button(frame, text="Start annotation", command=self._start).pack(anchor="e", pady=(12, 0))

    def _load_saved_paths(self):
        try:
            if not self.SETTINGS_PATH.exists():
                return
            settings = json.loads(self.SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return

        self.source_var.set(settings.get("source_dataset", ""))
        self.audio_var.set(settings.get("audio_folder", ""))
        self.reviewed_var.set(settings.get("reviewed_dataset", ""))
        self.naming_var.set(settings.get("naming_csv", ""))

    def _save_paths(self, source_dataset: Path, audio_folder: Path, reviewed_dataset: Path, naming_csv: Path):
        settings = {
            "source_dataset": str(source_dataset),
            "audio_folder": str(audio_folder),
            "reviewed_dataset": str(reviewed_dataset),
            "naming_csv": str(naming_csv),
        }
        self.SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def _path_row(self, parent, label, var: tk.StringVar, folder: bool):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(0, 8))

        def browse():
            if folder:
                selected = filedialog.askdirectory(parent=self)
            else:
                selected = filedialog.askopenfilename(
                    parent=self,
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                )
            if selected:
                var.set(selected)

        ttk.Button(row, text="Browse", command=browse).pack(side="left")

    def _entry_row(self, parent, label, var: tk.StringVar):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _start(self):
        try:
            source_dataset = Path(self.source_var.get()).expanduser()
            reviewed_dataset = Path(self.reviewed_var.get()).expanduser()
            naming_csv = Path(self.naming_var.get()).expanduser()

            audio_raw = self.audio_var.get().strip()
            audio_folder = Path(audio_raw).expanduser() if audio_raw else source_dataset / "audio"

            if not source_dataset.exists():
                raise FileNotFoundError(f"Source dataset folder not found: {source_dataset}")
            if not audio_folder.exists():
                raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
            if not naming_csv.exists():
                raise FileNotFoundError(f"Naming CSV not found: {naming_csv}")

            reviewed_dataset.mkdir(parents=True, exist_ok=True)
            paths = SessionPaths(
                source_dataset=source_dataset,
                audio_folder=audio_folder,
                reviewed_dataset=reviewed_dataset,
                naming_csv=naming_csv,
            )
            if not paths.original_metadata.exists():
                raise FileNotFoundError(f"Missing metadata file: {paths.original_metadata}")
            if not paths.original_labels.exists():
                raise FileNotFoundError(f"Missing annotations file: {paths.original_labels}")

            self._save_paths(
                source_dataset=source_dataset,
                audio_folder=audio_folder,
                reviewed_dataset=reviewed_dataset,
                naming_csv=naming_csv,
            )

            author = self.author_var.get().strip() or None
            reviewer = self.reviewer_var.get().strip() or None
            AnnotationDesktopWindow(self, paths=paths, author=author, reviewer=reviewer)
        except Exception as exc:
            messagebox.showerror("Unable to start", str(exc), parent=self)


def main():
    app = LauncherApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
