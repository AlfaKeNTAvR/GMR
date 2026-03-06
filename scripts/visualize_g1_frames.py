#!/usr/bin/env python3
"""
Visualize selected G1 body frames (axes + labels) in MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, List

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.params import IK_CONFIG_DICT


_SKYBOX = (
    '<texture type="skybox" builtin="gradient"'
    ' rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>'
)
_FLOOR = (
    '<texture type="2d" name="groundplane" builtin="checker" mark="edge"'
    ' rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
    '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>'
)
_SCENE_WRAP = """\
<mujoco model="scene">
  <include file="{robot_xml}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    {skybox}
    {floor}
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"
          contype="1" conaffinity="1" condim="3"/>
  </worldbody>
</mujoco>
"""


def _load_model(xml_path: Path) -> mj.MjModel:
    """Load model wrapped in a scene with skybox and floor if not already present."""
    xml_text = xml_path.read_text(encoding="utf-8")
    if 'type="skybox"' in xml_text:
        return mj.MjModel.from_xml_path(str(xml_path))
    scene_xml = _SCENE_WRAP.format(
        robot_xml=xml_path.name, skybox=_SKYBOX, floor=_FLOOR
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=str(xml_path.parent),
        delete=False, encoding="utf-8"
    ) as f:
        f.write(scene_xml)
        tmp = f.name
    try:
        return mj.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)


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
    parser = argparse.ArgumentParser(description="Visualize robot body frames from an IK config.")
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        choices=list(ROBOT_XML_DICT.keys()),
    )
    parser.add_argument(
        "--ik-config",
        default=None,
        help="Path to IK config JSON. Defaults to the solarxr config for the selected robot.",
    )
    parser.add_argument("--size", type=float, default=0.08)
    args = parser.parse_args()

    if args.ik_config is None:
        solarxr_configs = IK_CONFIG_DICT.get("solarxr", {})
        if args.robot not in solarxr_configs:
            parser.error(f"No default solarxr IK config for '{args.robot}'. Pass --ik-config explicitly.")
        args.ik_config = str(solarxr_configs[args.robot])

    xml_path = ROBOT_XML_DICT[args.robot]
    model = _load_model(Path(str(xml_path)))
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
