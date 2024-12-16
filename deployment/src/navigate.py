#!/usr/bin/env python3
import os
import roslib
import rospy
from std_msgs.msg import Float32MultiArray

roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import CarrotPlannerSrv, CarrotPlannerSrvResponse
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import numpy as np
import torch
import yaml
from utils import msg_to_pil, to_numpy, transform_images, load_model
from topic_names import (IMAGE_TOPIC, CARROT_SERVICE_NAME)

# Load configuration
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# Globals
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_queue = []
context_size = None

# Callback for image topic
def callback_obs(msg):
    global context_queue
    obs_img = msg_to_pil(msg)
    transformed_img = transform_images(obs_img, model_params["image_size"], center_crop=False).to(device)

    # Efficiently update the queue
    if len(context_queue) >= context_size+1:
        context_queue.pop(-1)  # Remove the oldest image
    context_queue.insert(0, transformed_img)  # Add the new transformed image to the front

# Load model function
def load_navigation_model(args):
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    ckpt_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
    model = load_model(
        ckpt_path,
        model_params,
        device,
    )
    model = model.to(device).eval()
    return model, model_params

# Service handler
def handle_carrot_planner(req):
    rospy.loginfo("Received request for waypoint computation.")
    carrot = req.carrot.pose.position
    carrot_xy = torch.tensor([carrot.x, carrot.y], dtype=torch.float32).to(device)
    carrot_xy = carrot_xy.unsqueeze(0)  # Add batch dimension  

    if len(context_queue) >= context_size+1:
        # Concatenate context images into a single tensor along the channel dimension
        obs_images = torch.cat(context_queue, dim=1).to(device)

        with torch.no_grad():
            distances, waypoints = model(obs_images, carrot_xy)
            distances = to_numpy(distances)[0]
            waypoints = to_numpy(waypoints)[0]

        if model_params["normalize"]:
            waypoints[:2] *= (MAX_V / RATE)

        # Return response
        response = CarrotPlannerSrvResponse()
        path = Path()
        path.header.frame_id = "base_link"
        path.header.stamp = rospy.Time.now()

        for waypoint in waypoints:
            pose = PoseStamped()
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            path.poses.append(pose)
        response.path = path
        # rospy.loginfo(f"Returning path: {response.path}")
        rospy.loginfo(f"Returning path with endpoint {waypoints[-1]}")
        return response
    else:
        rospy.logwarn("Not enough context images for inference.")
        return CarrotPlannerSrvResponse(path=path)

def main():
    global context_size, model, model_params, args

    # Argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="CarrotPlanner Service Server")
    parser.add_argument("--model", default="vint", type=str, help="Model name")
    parser.add_argument("--waypoint", default=2, type=int, help="Waypoint index")
    args = parser.parse_args()

    rospy.init_node("carrot_planner_service")
    rospy.loginfo("Starting CarrotPlanner service...")

    # Load model
    model, model_params = load_navigation_model(args)
    context_size = model_params["context_size"]

    # Subscriber for image topic
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)

    # Service server
    service = rospy.Service(CARROT_SERVICE_NAME, CarrotPlannerSrv, handle_carrot_planner)
    rospy.loginfo("CarrotPlannerSrv service ready. Waiting for requests...")

    rospy.spin()

if __name__ == "__main__":
    main()
