#!/usr/bin/env python3
"""
Visualize selected G1 body frames (axes + labels) in MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT


def _unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_frame_names(config_path: Path) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = data.get(table_name, {})
        for frame_name in table.keys():
            frames.append(frame_name)
    return _unique(frames)


def _draw_axis(v, pos, mat, label: str, size: float = 0.08):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.005, 0.005, 0.005],
            pos=pos,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if i == 0:
            geom.label = label
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.003,
            from_=pos,
            to=pos + size * mat[:, i],
        )
        v.user_scn.ngeom += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize G1 body frames from IK config.")
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        choices=list(ROBOT_XML_DICT.keys()),
    )
    parser.add_argument(
        "--ik-config",
        default=str(Path(__file__).resolve().parents[1] / "general_motion_retargeting" / "ik_configs" / "solarxr_to_g1.json"),
    )
    parser.add_argument("--size", type=float, default=0.08)
    args = parser.parse_args()

    xml_path = ROBOT_XML_DICT[args.robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    frames = _load_frame_names(Path(args.ik_config))

    viewer = mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    )

    try:
        while True:
            mj.mj_forward(model, data)
            viewer.user_scn.ngeom = 0

            for name in frames:
                try:
                    bid = model.body(name).id
                except KeyError:
                    continue
                pos = data.xpos[bid].copy()
                mat = data.xmat[bid].reshape(3, 3).copy()
                _draw_axis(viewer, pos, mat, name, size=args.size)

            viewer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
