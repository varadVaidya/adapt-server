import numpy as np
import mujoco
import mujoco.renderer


def add_visual_capsule(
    scene: mujoco.MjvScene, point1, point2, radius, rgba: np.ndarray
):

    if scene.ngeom >= scene.geoms.size:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtObj.GEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtObj.GEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


def modify_scene(scene: mujoco.MjvScene, position_trace, ref_position_trace=None):
    """Draw the position traces on the string

    Args:
        scene (mujoco.MjvScene):
        position_trace (deque): deque of positions
        ref_position_trace (deque, optional): deque of position. Defaults to None.
    """

    if len(position_trace) > 1:
        for i in range(len(position_trace) - 1):
            rgba = np.array([0.2, 1.0, 0.2, 0.75])
            radius = 0.0025
            point1 = position_trace[i]
            point2 = position_trace[i + 1]
            add_visual_capsule(scene, point1, point2, radius, rgba)

    if ref_position_trace is not None:
        if len(ref_position_trace) > 1:
            for i in range(len(ref_position_trace) - 1):
                rgba = np.array([1.0, 0.2, 0.2, 0.75])
                radius = 0.0025
                point1 = ref_position_trace[i]
                point2 = ref_position_trace[i + 1]
                add_visual_capsule(scene, point1, point2, radius, rgba)
