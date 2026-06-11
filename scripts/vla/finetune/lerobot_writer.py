"""
GR00T-flavored LeRobot v2 dataset writer.

Writes demonstrations in the exact format GR00T fine-tuning consumes (template:
Isaac-GR00T/demo_data/cube_to_bowl_5). Isaac-free (numpy/pandas/imageio only) so
it can be unit-tested without launching the sim.

Layout produced:
    <root>/
      meta/{info.json, tasks.jsonl, episodes.jsonl, modality.json}
      data/chunk-000/episode_000000.parquet ...
      videos/chunk-000/observation.images.ego_view/episode_000000.mp4 ...

NOTE: stats.json + relative_stats.json are NOT written here — generate them after
recording with GR00T's own tool (correct by construction):
    python -m gr00t.data.stats --dataset-path <root> \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path scripts/vla/finetune/g1_tabletop_config.py

Usage:
    rec = LeRobotV2Recorder(root, fps=30, state_names=[...], action_names=[...])
    rec.start_episode("reach the blue cube")
    rec.add_frame(state6, action6, rgb_hwc_uint8)   # per sim step
    rec.end_episode()
    ... more episodes ...
    rec.close()                                     # writes meta/
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

_VIDEO_KEY = "ego_view"
_IMG_COL   = f"observation.images.{_VIDEO_KEY}"


class LeRobotV2Recorder:
    def __init__(self, root, fps: int, state_names: list[str], action_names: list[str],
                 modality: dict, image_hw=(480, 640), robot_type="g1_sim",
                 chunk_size: int = 1000):
        self.root        = Path(root)
        self.fps         = fps
        self.state_names = state_names
        self.action_names = action_names
        self.modality    = modality          # meta/modality.json contents
        self.h, self.w   = image_hw
        self.robot_type  = robot_type
        self.chunk_size  = chunk_size

        (self.root / "meta").mkdir(parents=True, exist_ok=True)

        self._tasks: dict[str, int] = {}     # task string -> task_index
        self._episodes: list[dict]  = []     # {episode_index, tasks, length}
        self._global_index = 0               # running frame index across episodes
        self._ep_idx       = 0

        # current-episode buffers
        self._cur_task = None
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []

    # ------------------------------------------------------------------ episodes
    def start_episode(self, task: str) -> None:
        if self._cur_task is not None:
            raise RuntimeError("end_episode() the previous episode first")
        self._cur_task = task
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        self._states.clear(); self._actions.clear(); self._frames.clear()

    def add_frame(self, state: np.ndarray, action: np.ndarray, rgb: np.ndarray) -> None:
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._frames.append(np.asarray(rgb, dtype=np.uint8))

    def end_episode(self) -> None:
        if self._cur_task is None:
            raise RuntimeError("start_episode() first")
        n = len(self._frames)
        if n == 0:
            raise RuntimeError("episode has no frames")
        chunk = self._ep_idx // self.chunk_size
        task_index = self._tasks[self._cur_task]

        # --- parquet ---
        rows = {
            "observation.state": [s for s in self._states],
            "action":            [a for a in self._actions],
            "timestamp":  np.arange(n, dtype=np.float32) / self.fps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, self._ep_idx, dtype=np.int64),
            "index": np.arange(self._global_index, self._global_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
        }
        pq_path = self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{self._ep_idx:06d}.parquet"
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(pq_path, index=False)

        # --- video ---
        vid_path = (self.root / "videos" / f"chunk-{chunk:03d}" / _IMG_COL
                    / f"episode_{self._ep_idx:06d}.mp4")
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(vid_path, self._frames, fps=self.fps, codec="libx264",
                         pixelformat="yuv420p", macro_block_size=1,
                         output_params=["-crf", "20"])

        self._episodes.append({"episode_index": self._ep_idx,
                               "tasks": [self._cur_task], "length": n})
        self._global_index += n
        self._ep_idx += 1
        self._cur_task = None

    # ------------------------------------------------------------------ finalize
    def close(self) -> None:
        meta = self.root / "meta"
        with open(meta / "tasks.jsonl", "w") as f:
            for task, idx in sorted(self._tasks.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
        with open(meta / "episodes.jsonl", "w") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep) + "\n")
        with open(meta / "modality.json", "w") as f:
            json.dump(self.modality, f, indent=4)
        with open(meta / "info.json", "w") as f:
            json.dump(self._info(), f, indent=4)

    def _info(self) -> dict:
        total_frames = self._global_index
        vinfo = {
            "video.height": self.h, "video.width": self.w, "video.codec": "h264",
            "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
            "video.fps": self.fps, "video.channels": 3, "has_audio": False,
        }
        return {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": self._ep_idx,
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{self._ep_idx}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {"dtype": "float32", "names": self.action_names,
                           "shape": [len(self.action_names)]},
                "observation.state": {"dtype": "float32", "names": self.state_names,
                                      "shape": [len(self.state_names)]},
                _IMG_COL: {"dtype": "video", "shape": [self.h, self.w, 3],
                           "names": ["height", "width", "channels"], "info": vinfo},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
            "total_chunks": 0,
            "total_videos": self._ep_idx,   # one camera per episode
        }


# modality.json contents for our single-arm + gripper G1 embodiment.
def g1_tabletop_modality() -> dict:
    return {
        "state":  {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
        "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
        "video":  {_VIDEO_KEY: {"original_key": _IMG_COL}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
