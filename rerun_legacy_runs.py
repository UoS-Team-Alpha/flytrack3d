"""Re-run every run in the output folder with the current pipeline, in place of its legacy results.

For each run_* folder (or the ones named in RUNS / --runs):
    1. The legacy run is moved to <output>/_legacy_backup/<run>/ (deleted at the end if
       KEEP_BACKUP is False and everything succeeded; restored if step 5 fails).
    2. Step 5 again, under the same run name: main_evaluate2d.py for evaluation runs (they have
       ground truth), main_inference2d.py for inference-only runs. Same models, inputs and settings
       as recorded in the run's config.json / metadata.json, except the new confidence logic:
       inference at INFERENCE_CONF (every box saved with its confidence, so AP covers the whole
       curve) and P/R/F1 at OPERATING_CONF.
    3. Every 3D output the run had, again, from the legacy layout as found:
           3d/<label>/factor_graph/tracks_*/   main_tracks3d.py         with that output's params.json
           3d/<label>/factor_graph/guided_*/   main_tracks3d_guided.py  with its params.json and the
                                                                        run's saved annotations (no GUI)
           3d/<label>/points/, triangulated/   main_points3d.py
           3d/<label>/tracks/, a bare factor_graph/   main_tracks3d.py   (older layout, no params:
           3d/<label>/tracks_guided/                   main_tracks3d_guided.py   current settings)
       <label> = the model, or ground_truth (run with --source gt). Each new factor-graph folder is
       renamed to its legacy folder's name, so old and new line up. tracking3d.conf_threshold null
       (every saved box) becomes OPERATING_CONF: the predictions now go down to INFERENCE_CONF.
       Every 3D job draws the fruit (now a hemisphere) and leaves: their caches are not carried
       over, so they are fitted again with the current arena_plotting settings.
    Camera frame offsets: a run's offsets are the ones recorded in its newest params.json that
    has them (they are a property of the recording), else FRAME_OFFSETS[run], else 0 (runs made
    before offsets existed).

Before anything changes, the plan is printed and you are asked to type "yes" (--yes skips it).
configs/legacy/ is deleted first (DELETE_LEGACY_CONFIGS). Every job's output goes to a log in
<output>/_rerun_logs/<datetime>/, with the exact config it ran with. A finished run gets a
.rerun_complete marker and is skipped next time (--redo to run it again); an interrupted one is
picked up from its backup.

    python rerun_legacy_runs.py --dry-run              # print the plan only
    python rerun_legacy_runs.py                        # every run, asks first
    python rerun_legacy_runs.py --runs run_20261008-101724_showoff_1 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ============================================================================ SETTINGS
RUNS = None                    # e.g. ["run_20261008-101724_showoff_1"]; None = every run_* folder
INFERENCE_CONF = 0.001         # models' confidence_threshold: every box above it is saved
OPERATING_CONF = 0.25          # P/R/F1 and tracking, where the legacy setting was null
KEEP_BACKUP = True             # keep <output>/_legacy_backup/<run>/; False = delete it once the run succeeded
DELETE_LEGACY_CONFIGS = True   # delete configs/legacy/ before starting
FRAME_OFFSETS: dict[str, dict[str, int]] = {}   # {run: {camera: offset}} for runs that never recorded any
# =====================================================================================

STEP5_EVAL = ROOT / "pipeline" / "step5_evaluation" / "main_evaluate2d.py"
STEP7 = ROOT / "pipeline" / "step7_points3d" / "main_points3d.py"
STEP8 = ROOT / "pipeline" / "step8_tracking3d" / "main_tracks3d.py"
STEP8_GUIDED = ROOT / "pipeline" / "step8_tracking3d" / "main_tracks3d_guided.py"
BACKUP = "_legacy_backup"
LOGS = "_rerun_logs"
DONE = ".rerun_complete"
NOT_LABELS = {"fruit", "leaves"}          # 3d/ sub-folders that are caches, not outputs
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PERCENT = re.compile(r"(\d{1,3})%\|")


# ----------------------------------------------------------------------------- the plan

@dataclass
class Job:
    run: str
    kind: str                   # step5-eval, step5-infer, points, tracks, guided
    replaces: str               # the legacy output, relative to the run folder
    label: str = ""             # 3d/<label>: a model or ground_truth
    model: str | None = None
    source: str = "pred"
    params: dict | None = None  # the legacy output's params.json
    target: str | None = None   # name for the new factor-graph folder
    notes: list[str] = field(default_factory=list)
    status: str = "pending"
    seconds: float = 0.0
    log: Path | None = None

    @property
    def title(self) -> str:
        what = {"step5-eval": "step 5 evaluation", "step5-infer": "step 6 inference", "points": "step 7 points",
                "tracks": "step 8 tracks", "guided": "step 8 guided"}[self.kind]
        return f"{what}  ->  {self.replaces}"


@dataclass
class RunPlan:
    name: str
    legacy: Path                # where the legacy run is now (the run folder or its backup)
    jobs: list[Job]
    offsets: dict[str, int]
    offsets_from: str
    warnings: list[str]


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _run_models(meta: dict) -> list[str]:
    if meta.get("models"):
        return list(meta["models"])
    return list(dict.fromkeys(m for i in meta.get("inputs", {}).values() for m in i.get("models", {})))


def _offsets(run: str, legacy: Path, cameras: list[str]) -> tuple[dict[str, int], str]:
    recorded = sorted((p for p in legacy.glob("3d/*/factor_graph/*/params.json") if _json(p).get("frame_offsets")),
                      key=lambda p: _json(p).get("created", ""))
    if recorded:
        return {**{c: 0 for c in cameras}, **_json(recorded[-1])["frame_offsets"]}, \
            f"recorded in {recorded[-1].parent.name}"
    if run in FRAME_OFFSETS:
        return {**{c: 0 for c in cameras}, **FRAME_OFFSETS[run]}, "FRAME_OFFSETS setting"
    return {c: 0 for c in cameras}, "none recorded: 0 (made before frame offsets existed)"


def plan_run(name: str, legacy: Path, cameras: list[str]) -> RunPlan:
    meta, config = _json(legacy / "metadata.json"), _json(legacy / "config.json")
    warnings = []
    models = _run_models(meta)
    if not meta or not config:
        return RunPlan(name, legacy, [], {}, "", ["no metadata.json / config.json: not a pipeline run, skipped"])
    kind = "step5-infer" if meta.get("kind") == "inference" else "step5-eval"
    jobs = [Job(name, kind, "the whole run folder (predictions, metrics, summary)")]
    for m in config.get("models", []):
        path = m.get("model_path")
        if m["name"] in models and path and Path(path).is_absolute() and not Path(path).is_file():
            warnings.append(f"weights of {m['name']} not found: {path}")
    for i in meta.get("inputs", {}).values():
        src = i.get("dataset", {}).get("source", {})
        media = src.get("video") or src.get("images")
        if media and not Path(media).exists():
            warnings.append(f"input not found: {media}")

    d3 = legacy / "3d"
    has_annotations = (d3 / "annotations.json").is_file()
    if d3.is_dir():
        label_dirs = [p for p in sorted(d3.iterdir()) if p.is_dir() and p.name not in NOT_LABELS]
        # oldest layout: 3d/points, 3d/triangulated with no label folder -> the run's model
        top = [p for p in label_dirs if p.name in ("points", "triangulated")]
        if top:
            jobs.append(Job(name, "points", " + ".join(f"3d/{p.name}" for p in top), label=models[0], model=models[0]))
        for ld in (p for p in label_dirs if p.name not in ("points", "triangulated")):
            gt = ld.name == "ground_truth"
            model, source = (models[0], "gt") if gt else (ld.name, "pred")
            rel = lambda p: str(p.relative_to(legacy))  # noqa: E731
            pts = [p for p in (ld / "points", ld / "triangulated") if p.is_dir()]
            if pts:
                jobs.append(Job(name, "points", " + ".join(rel(p) for p in pts), ld.name, model, source))
            fg = ld / "factor_graph"
            bare = fg.is_dir() and any(f.is_file() for f in fg.iterdir())
            old_tracks = [p for p in (ld / "tracks",) if p.is_dir()] + ([fg] if bare else [])
            if old_tracks:
                jobs.append(Job(name, "tracks", " + ".join(rel(p) + ("/ (files)" if p == fg else "") for p in old_tracks),
                                ld.name, model, source, target="tracks_legacy",
                                notes=["older layout, no params.json: current tracking3d settings"]))
            if (ld / "tracks_guided").is_dir():
                jobs.append(Job(name, "guided", rel(ld / "tracks_guided"), ld.name, model, source,
                                target="guided_legacy", notes=["older layout, no params.json: current tracking3d settings"]))
            for out in sorted(p for p in fg.glob("*") if p.is_dir() and p.name.startswith(("tracks_", "guided_"))) \
                    if fg.is_dir() else []:
                params = _json(out / "params.json")
                mode = "guided" if out.name.startswith("guided_") or params.get("mode") == "guided" else "tracks"
                notes = [] if params else ["no params.json: current tracking3d settings"]
                t3d = params.get("tracking3d", {})
                if t3d.get("n_tracks") is not None:
                    notes.append(f"n_tracks={t3d['n_tracks']}")
                if params and t3d.get("conf_threshold", None) is None:
                    notes.append(f"conf_threshold null -> {OPERATING_CONF:g}")
                jobs.append(Job(name, mode, rel(out), ld.name, model, source, params or None, target=out.name,
                                notes=notes))
    for j in jobs:
        if j.kind == "guided" and not has_annotations:
            j.status = "skip"
            j.notes.append("no 3d/annotations.json in the run: skipped")
    offsets, offsets_from = _offsets(name, legacy, cameras)
    return RunPlan(name, legacy, jobs, offsets, offsets_from, warnings)


# ----------------------------------------------------------------------------- configs per job

def step5_spec(plan: RunPlan, out_root: Path, cfg_dir: Path) -> Path:
    """The worker spec for a run's step 5: its own config with the new confidence logic."""
    config = _json(plan.legacy / "config.json")
    meta = _json(plan.legacy / "metadata.json")
    config["output_dir"] = str(out_root)
    for m in config.get("models", []):
        m["confidence_threshold"] = INFERENCE_CONF
    ev = config.setdefault("evaluation", {})
    if ev.get("pr_conf_threshold") is None:
        ev["pr_conf_threshold"] = OPERATING_CONF
    legacy_dirs = [p.name for p in plan.legacy.rglob("*") if p.is_dir()]
    spec = {
        "kind": plan.jobs[0].kind, "run_name": plan.name, "config": config, "models": _run_models(meta),
        "items": [{**i["dataset"]["source"], "name": n} for n, i in meta.get("inputs", {}).items()],
        "save_video": "videos" in legacy_dirs, "save_track_plot": "track_plots" in legacy_dirs,
    }
    path = cfg_dir / f"{plan.name}__step5.json"
    path.write_text(json.dumps(spec, indent=2, default=str))
    return path


def tracking_config(job: Job, plan: RunPlan, out_root: Path, cfg_dir: Path) -> Path:
    """configs/tracking.json as it is now (calibration, arena with the hemisphere fruit), with the
    legacy output's tracking3d / evaluation settings, the run's frame offsets and the new conf logic."""
    from common.config import load_config
    config = load_config(ROOT / "configs" / "tracking.json")
    config["output_dir"] = str(out_root)
    p = job.params or {}
    t3d = {**config.get("tracking3d", {}), **{k: v for k, v in p.get("tracking3d", {}).items() if not k.startswith("_")}}
    if t3d.get("conf_threshold") is None:
        t3d["conf_threshold"] = OPERATING_CONF
    config["tracking3d"] = t3d
    ev = {**config.get("evaluation", {}), **p.get("evaluation", {})}
    if ev.get("pr_conf_threshold") is None:
        ev["pr_conf_threshold"] = OPERATING_CONF
    config["evaluation"] = ev
    for cam, view in config.get("cameras", {}).get("views", {}).items():
        view["frame_offset"] = int(plan.offsets.get(cam, 0))
    path = cfg_dir / f"{plan.name}__{job.label}__{job.target or job.kind}.json"
    path.write_text(json.dumps(config, indent=2, default=str))
    return path


def job_command(job: Job, plan: RunPlan, out_root: Path, cfg_dir: Path) -> list[str]:
    py = [sys.executable, "-u"]
    if job.kind.startswith("step5"):
        return py + [str(Path(__file__).resolve()), "--worker", str(step5_spec(plan, out_root, cfg_dir))]
    cfg = tracking_config(job, plan, out_root, cfg_dir)
    script = {"points": STEP7, "tracks": STEP8, "guided": STEP8_GUIDED}[job.kind]
    cmd = py + [str(script), "--config", str(cfg), "--run", plan.name, "--model", job.model, "--source", job.source,
                "--no-show", "--arena"]
    if job.kind == "guided":
        cmd += ["--no-gui", "--annotations", str(out_root / plan.name / "3d" / "annotations.json")]
    return cmd


# ----------------------------------------------------------------------------- the worker (step 5 in a child process)

def worker(spec_path: str) -> None:
    from detection.evaluate_models import run_evaluation, run_inference_only
    spec = json.loads(Path(spec_path).read_text())
    if spec["kind"] == "step5-eval":
        run_evaluation(spec["config"], only_models=spec["models"], save_video=spec["save_video"],
                       save_track_plot=spec["save_track_plot"], run_name=spec["run_name"])
    else:
        run_inference_only(spec["config"], only_models=spec["models"], run_name=spec["run_name"],
                           items=spec["items"])


# ----------------------------------------------------------------------------- running, with progress

class Display:
    """Live progress: every job, the current run's jobs, the current job (from its own tqdm bars)
    and the job's latest output line."""

    def __init__(self, total_jobs: int, total_runs: int):
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn,
                                   TextColumn, TimeElapsedColumn, TimeRemainingColumn)
        from rich.text import Text
        self.Text, self.Panel = Text, Panel
        self.top = Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(bar_width=36),
                            MofNCompleteColumn(), TextColumn("[dim]elapsed"), TimeElapsedColumn(),
                            TextColumn("[dim]left"), TimeRemainingColumn())
        self.job = Progress(TextColumn("   {task.description}"), BarColumn(bar_width=36), TaskProgressColumn(),
                            TimeElapsedColumn())
        self.all = self.top.add_task("All legacy outputs", total=total_jobs)
        self.runs = self.top.add_task(f"Runs (0/{total_runs})", total=total_runs)
        self.run = self.top.add_task("", total=1, visible=False)
        self.cur = self.job.add_task("", total=None, visible=False)
        self.status = Text("", style="dim", overflow="ellipsis", no_wrap=True)
        self.header = Text("")
        self.group = Group(self.header, self.top, self.job, Panel(self.status, title="latest output", title_align="left",
                                                                  border_style="dim"))
        self.live = Live(self.group, refresh_per_second=8)
        self.console = self.live.console

    def start_run(self, i: int, n: int, plan: RunPlan) -> None:
        self.header.plain = ""
        self.header.append(f"Run {i}/{n}: ", style="bold")
        self.header.append(plan.name, style="bold cyan")
        self.header.append(f"   (replacing the legacy run, kept in {BACKUP}/)" if KEEP_BACKUP else "   (replacing the legacy run)",
                           style="dim")
        self.top.update(self.run, description=f"{plan.name}", total=len(plan.jobs), completed=0, visible=True)

    def start_job(self, j: int, job: Job) -> None:
        self.job.update(self.cur, description=f"[{j}] {job.title}", total=None, completed=0, visible=True)
        self.status.plain = ""

    def line(self, text: str) -> None:
        self.status.plain = text[-300:]
        m = PERCENT.findall(text)
        if m:
            self.job.update(self.cur, total=100, completed=int(m[-1]))

    def end_job(self, job: Job) -> None:
        mark = {"ok": "[green]done[/]", "failed": "[red]FAILED[/]", "skip": "[yellow]skipped[/]"}.get(job.status, job.status)
        self.console.print(f"  {mark}  {job.run}  {job.title}  [dim]({job.seconds / 60:.1f} min"
                           f"{f', log {job.log}' if job.log else ''})[/]")
        self.top.advance(self.all)
        self.top.advance(self.run)


def run_child(cmd: list[str], log: Path, display: Display) -> int:
    """Run a job in a child process: everything it prints goes to its log; its latest line (and the
    percentage of its tqdm bars) to the display."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
    with open(log, "wb") as fh:
        fh.write(("$ " + " ".join(cmd) + "\n\n").encode())
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        buf = b""
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            fh.write(chunk)
            buf = (buf + chunk)[-4096:]
            parts = [p for p in re.split(rb"[\r\n]", buf) if p.strip()]
            if parts:
                display.line(ANSI.sub("", parts[-1].decode(errors="replace")).strip())
        return proc.wait()


def _new_dir(parent: Path, before: set[str]) -> Path | None:
    new = [p for p in parent.glob("*") if p.is_dir() and p.name not in before] if parent.is_dir() else []
    return max(new, key=lambda p: p.stat().st_mtime) if new else None


def execute(plan: RunPlan, out_root: Path, log_dir: Path, cfg_dir: Path, display: Display) -> bool:
    run_dir, backup = out_root / plan.name, out_root / BACKUP / plan.name
    if plan.legacy == run_dir:   # first attempt: move the legacy run aside
        backup.parent.mkdir(exist_ok=True)
        shutil.move(str(run_dir), str(backup))
        plan.legacy = backup
    elif run_dir.exists():       # an earlier attempt was interrupted: start again from the backup
        shutil.rmtree(run_dir)
    ok = True
    for j, job in enumerate(plan.jobs, 1):
        display.start_job(j, job)
        if job.status == "skip" or (not ok and not job.kind.startswith("step5")):
            if job.status != "skip":
                job.status, job.notes = "skip", job.notes + ["step 5 failed"]
            display.end_job(job)
            continue
        job.log = log_dir / f"{plan.name}__{j:02d}_{job.kind}_{(job.target or job.label or '').replace('/', '_')}.log"
        fg = run_dir / "3d" / job.label / "factor_graph"
        before = {p.name for p in fg.glob("*")} if fg.is_dir() else set()
        t0 = time.time()
        try:
            code = run_child(job_command(job, plan, out_root, cfg_dir), job.log, display)
        except Exception as e:   # noqa: BLE001 - one job must not stop the others
            job.log.open("a").write(f"\n{type(e).__name__}: {e}\n")
            code = 1
        job.seconds = time.time() - t0
        job.status = "ok" if code == 0 else "failed"
        if job.kind.startswith("step5"):
            if code != 0:
                ok = False
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                shutil.move(str(backup), str(run_dir))   # restore the legacy run
                plan.legacy = run_dir
                job.notes.append("legacy run restored")
            elif (backup / "3d" / "annotations.json").is_file():
                (run_dir / "3d").mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup / "3d" / "annotations.json", run_dir / "3d" / "annotations.json")
        elif code == 0 and job.target and job.kind in ("tracks", "guided"):
            new = _new_dir(fg, before)
            if new is not None and new.name != job.target:
                new.rename(fg / job.target)
        display.end_job(job)
    if ok and all(j.status in ("ok", "skip") for j in plan.jobs):
        (run_dir / DONE).write_text(datetime.now().isoformat(timespec="seconds") + "\n")
        if not KEEP_BACKUP and backup.exists():
            shutil.rmtree(backup)
    return ok


# ----------------------------------------------------------------------------- main

def print_plan(plans: list[RunPlan], skipped_done: list[str], console) -> None:
    from rich.table import Table
    for p in plans:
        t = Table(title=f"[bold cyan]{p.name}[/]   [dim]legacy: {p.legacy}[/]", title_justify="left",
                  show_lines=False, expand=True)
        t.add_column("#", justify="right", style="dim", width=3)
        t.add_column("new job")
        t.add_column("replaces legacy output")
        t.add_column("notes", style="dim")
        for i, j in enumerate(p.jobs, 1):
            t.add_row(str(i), j.title.split("  ->  ")[0] + (f" [{j.source}]" if j.source == "gt" else ""),
                      j.replaces + (f"  [dim]-> {j.target}[/]" if j.target and j.target not in j.replaces else ""),
                      "; ".join(j.notes))
        console.print(t)
        if p.jobs[1:]:
            console.print(f"   frame offsets: {p.offsets}  [dim]({p.offsets_from})[/]")
        for w in p.warnings:
            console.print(f"   [yellow]WARNING[/] {w}")
        console.print()
    for name in skipped_done:
        console.print(f"[dim]{name}: already re-run ({DONE} present) - skipped; --redo to run it again[/]")


def main():
    parser = argparse.ArgumentParser(description="Re-run every legacy run with the current pipeline.")
    parser.add_argument("--runs", nargs="+", default=RUNS, help="Run folder names (default: every run_* folder).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and stop.")
    parser.add_argument("--yes", action="store_true", help="Do not ask before starting.")
    parser.add_argument("--redo", action="store_true", help=f"Also re-run runs that have {DONE}.")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)

    from rich.console import Console
    from common.config import load_config
    console = Console()
    config = load_config(ROOT / "configs" / "tracking.json")
    out_root = Path(config["output_dir"])
    cameras = list(config.get("cameras", {}).get("views", {}))

    names = args.runs or sorted({p.name for p in out_root.glob("run_*") if p.is_dir()}
                                | {p.name for p in (out_root / BACKUP).glob("run_*") if p.is_dir()})
    plans, done = [], []
    for name in names:
        run_dir, backup = out_root / name, out_root / BACKUP / name
        if (run_dir / DONE).is_file() and not args.redo:
            done.append(name)
            continue
        legacy = backup if backup.is_dir() else run_dir   # a backup is the true legacy run (interrupted / --redo)
        if not legacy.is_dir():
            console.print(f"[yellow]{name}: not found in {out_root} or {out_root / BACKUP}[/]")
            continue
        plans.append(plan_run(name, legacy, cameras))
    plans = [p for p in plans if p.jobs]

    console.rule("[bold]Re-run plan")
    console.print(f"output folder : {out_root}\ninference conf: {INFERENCE_CONF:g} (every box saved)   "
                  f"operating conf: {OPERATING_CONF:g} (P/R/F1, tracking, where legacy was null)\n"
                  f"legacy runs   : {'moved to ' + str(out_root / BACKUP) if KEEP_BACKUP else 'deleted once re-run'}\n"
                  f"legacy configs: {'configs/legacy/ deleted' if DELETE_LEGACY_CONFIGS else 'kept'}\n")
    print_plan(plans, done, console)
    n_jobs = sum(len(p.jobs) for p in plans)
    if not plans:
        console.print("Nothing to do.")
        return
    console.print(f"[bold]{len(plans)} runs, {n_jobs} jobs.[/]")
    if args.dry_run:
        return
    if not args.yes and input("Type 'yes' to replace these legacy runs: ").strip().lower() != "yes":
        console.print("Stopped, nothing changed.")
        return

    if DELETE_LEGACY_CONFIGS and (ROOT / "configs" / "legacy").is_dir():
        shutil.rmtree(ROOT / "configs" / "legacy")
        console.print("Deleted configs/legacy/")
    log_dir = out_root / LOGS / datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_dir = log_dir / "configs"
    cfg_dir.mkdir(parents=True)
    console.print(f"Logs and per-job configs: {log_dir}\n")

    display = Display(n_jobs, len(plans))
    t0 = time.time()
    with display.live:
        for i, plan in enumerate(plans, 1):
            display.start_run(i, len(plans), plan)
            display.console.rule(f"[bold cyan]{plan.name}[/]  ({i}/{len(plans)})")
            execute(plan, out_root, log_dir, cfg_dir, display)
            display.top.update(display.runs, description=f"Runs ({i}/{len(plans)})", advance=1)

    from rich.table import Table
    t = Table(title=f"Summary ({(time.time() - t0) / 60:.1f} min)", title_justify="left", expand=True)
    for col in ("run", "job", "status", "min", "log / notes"):
        t.add_column(col)
    for p in plans:
        for j in p.jobs:
            style = {"ok": "green", "failed": "red", "skip": "yellow"}.get(j.status, "")
            t.add_row(p.name, j.title, f"[{style}]{j.status}[/]", f"{j.seconds / 60:.1f}",
                      str(j.log.name if j.log else "") + ("  " + "; ".join(j.notes) if j.notes else ""))
    console.print(t)
    failed = sum(j.status == "failed" for p in plans for j in p.jobs)
    console.print(f"[{'red' if failed else 'green'}]{failed} failed[/]. Logs: {log_dir}")


if __name__ == "__main__":
    main()
