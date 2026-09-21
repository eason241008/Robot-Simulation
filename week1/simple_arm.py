import time
import numpy as np

import mujoco
import mujoco.viewer


XML = """
<mujoco model="simple_arm">

    <compiler angle="radian"/>

    <option timestep="0.01"/>

    <worldbody>

        <geom
            type="plane"
            size="2 2 0.1"
            rgba="0.8 0.8 0.8 1"
        />

        <body name="link1" pos="0 0 0.1">

            <joint
                name="joint1"
                type="hinge"
                axis="0 0 1"
                range="-3.14 3.14"
            />

            <geom
                type="capsule"
                fromto="0 0 0 0.5 0 0"
                size="0.04"
            />

            <body name="link2" pos="0.5 0 0">

                <joint
                    name="joint2"
                    type="hinge"
                    axis="0 0 1"
                    range="-3.14 3.14"
                />

                <geom
                    type="capsule"
                    fromto="0 0 0 0.4 0 0"
                    size="0.035"
                />

                <site
                    name="end_effector"
                    pos="0.4 0 0"
                    size="0.04"
                    rgba="1 0 0 1"
                />

            </body>

        </body>

    </worldbody>

    <actuator>

        <position
            joint="joint1"
            kp="30"
            ctrlrange="-3.14 3.14"
        />

        <position
            joint="joint2"
            kp="30"
            ctrlrange="-3.14 3.14"
        />

    </actuator>

</mujoco>
"""


model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running():

        t = time.time() - start

        # Action
        data.ctrl[0] = 0.5 * np.sin(t)
        data.ctrl[1] = -0.8 * np.sin(t * 1.5)

        # Simulation step
        mujoco.mj_step(model, data)

        # Observation
        joint_positions = data.qpos.copy()
        ee_position = data.site_xpos[ee_id].copy()

        print(
            "joint:",
            np.round(joint_positions, 2),
            "end effector:",
            np.round(ee_position, 2),
        )

        viewer.sync()

        time.sleep(model.opt.timestep)