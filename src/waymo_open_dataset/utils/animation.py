
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import plot_maps
from waymo_open_dataset.utils import trajectory_utils

def get_box_vertices(x: float, y: float, z: float,
                     yaw: float, length: float, width: float, height=1.0
                     ) -> np.array:
    """
    return the (unordered) vertices of a box.

    input:
      x, y, z: global coordinate of the box center.
      yaw: counter-clockwise angle from x axis to the box heading (length direction).
      length, width, height: dimension along x, y, z.

    output:
      vertices of the box of shape [8, 3].
    """
    # 4 vertices in local coordinate of the box, shape = [4, 2]
    vertices_local_2d = np.array([[-length/2, -width/2],  # bottom_left
                                  [-length/2, width/2],  # top_left
                                  [length/2, width/2],  # top_right
                                  [length/2, -width/2],  # bottom_right
                                  ])

    # Transform the vertices into global coordinate
    rotation_matrix_2d = np.array([[np.cos(yaw), -np.sin(yaw)],
                                   [np.sin(yaw), np.cos(yaw)]])
    translation_vec_2d = np.array([x, y])
    vertices_global_2d = vertices_local_2d.dot(
        np.transpose(rotation_matrix_2d)) + translation_vec_2d

    # Add z cooridinate as well as the top four vertices.
    return np.concatenate((np.insert(vertices_global_2d, 2, z, axis=1),
                          np.insert(vertices_global_2d, 2, z+height, axis=1)))

def add_one_object(figure: go._figure.Figure,
                   x: float, y: float, z: float, yaw: float,
                   length: float, width: float, height: float,
                   obj_type: scenario_pb2.Track.ObjectType):
    """
    Add one object box to the given figure.
    """
    box_dict = {
        scenario_pb2.Track.TYPE_UNSET: ('ivory', 'solid'),
        scenario_pb2.Track.TYPE_VEHICLE: ('magenta', 'solid'),
        scenario_pb2.Track.TYPE_PEDESTRIAN: ('yellow', 'solid'),
        scenario_pb2.Track.TYPE_CYCLIST: ('red', 'solid'),
        scenario_pb2.Track.TYPE_OTHER: ('gray', 'solid'),
    }

    vertices = get_box_vertices(x=x, y=y, z=z, yaw=yaw,
                                length=length, width=width, height=height)

    figure.add_trace(go.Mesh3d(x=vertices[:, 0],
                               y=vertices[:, 1],
                               z=vertices[:, 2],
                               i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                               j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                               k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                               color=box_dict[obj_type.numpy()][0],
                               opacity=.7,
                               flatshading=True))

def add_traffic_signal_lane_state(figure: go._figure.Figure,
                                  pos: map_pb2.MapPoint,
                                  lane_state: map_pb2.TrafficSignalLaneState.State):
    """
    visualize one lane state.
    """
    light_dict = {
        map_pb2.TrafficSignalLaneState.LANE_STATE_UNKNOWN: ('grey', 'circle', 15),
        map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_STOP: ('red', 'x', 6),
        map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_CAUTION: ('yellow', 'x', 6),
        map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_GO: ('green', 'x', 6),
        map_pb2.TrafficSignalLaneState.LANE_STATE_STOP: ('red', 'circle', 15),
        map_pb2.TrafficSignalLaneState.LANE_STATE_CAUTION: ('yellow', 'circle', 15),
        map_pb2.TrafficSignalLaneState.LANE_STATE_GO: ('green', 'circle', 15),
        map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_STOP: ('red', 'circle-open', 15),
        map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_CAUTION: ('yellow', 'circle-open', 15),
    }
    figure.add_trace(
        go.Scatter3d(
            x=[pos.x],
            y=[pos.y],
            z=[pos.z],
            mode='markers',
            marker_symbol=light_dict[lane_state][1],
            marker=dict(
                size=light_dict[lane_state][2],
                color=light_dict[lane_state][0],
            ),
        )
    )

def plot_one_step(scenario: scenario_pb2.Scenario, time_idx: int):
    """
    Visualize a particular time step of a Scenario.
    Note the static map feature will be rendered first.
    input:
      map_features: the map feature of the scenario
      time_idx: the time step index to visualize

    output:
      figure: the generated figure object
      num_map_features: total number of static map features
    """
    # Plot out the static map features.
    figure = plot_maps.plot_map_features(scenario.map_features)
    num_map_featues = len(figure.data)

    # Plot out the traffic lights
    for lane_state in scenario.dynamic_map_states[time_idx].lane_states:
        add_traffic_signal_lane_state(
            figure, lane_state.stop_point, lane_state.state)

    # Plot out the trajectories of the tracked agents.
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(
        scenario)
    for i in range(logged_trajectories.valid.shape[0]):
        if not logged_trajectories.valid[i, time_idx]:
            continue
        add_one_object(figure,
                       logged_trajectories.x[i, time_idx],
                       logged_trajectories.y[i, time_idx],
                       logged_trajectories.z[i, time_idx],
                       logged_trajectories.heading[i, time_idx],
                       logged_trajectories.length[i, time_idx],
                       logged_trajectories.width[i, time_idx],
                       logged_trajectories.height[i, time_idx],
                       logged_trajectories.object_type[i]
                       )

    return figure, num_map_featues

def animate_scenario(scenario: scenario_pb2.Scenario, time_idxs: List):
    frames = []
    for idx in time_idxs:
        if idx not in range(len(scenario.timestamps_seconds)):
            continue

        fig_one_step, num_map_data = plot_one_step(scenario, idx)
        num_total_data = len(fig_one_step.data)
        # # option 1: update only the dynamic part
        # traces_for_update = list(range(num_map_data, num_total_data))
        # frames.append(go.Frame(data=fig_one_step.data[num_map_data:num_total_data],
        #                        layout=fig_one_step.layout,
        #                        traces=traces_for_update, name=f"step {idx}"))
        # option 2: update everything
        traces_for_update = list(range(0, num_total_data))
        frames.append(go.Frame(data=fig_one_step.data,
                               layout=fig_one_step.layout,
                               traces=traces_for_update, name=f"step {idx}"))

        # fig_one_step.show()
        # time.sleep(3.0)

    # Redo the first frame for visualization
    fig, _ = plot_one_step(scenario, time_idxs[0])
    fig.update(frames=frames)

    return fig
