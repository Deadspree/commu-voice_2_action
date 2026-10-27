## Project Overview
- Step 1: Setup unity Simulation to collect Commu robot pose image <-> robot joint angle data pairs -> Done
- Step 2: Train the model to take input as robot pose image and output robot joint angle -> Done
- Step 3: Create a video of people lip-syncing and put it into Lucy. An image (video) imitating CommU can be obtained.
- Step 4: Estimate the axis angle from the image in 3 and create a dataset.



## Directory Structures

- `simulation/unity/` — Unity CommU simulation
- `dataset/commu_pose_dataset` - Dataset for training robot pose image to robot joint angles model(pose_estimation model)
- `pose_estimation` - Where the pose estimation model is trained
- `runpod` - Sending necessary file to train on a runpod ssh
- `third_party` - Folder to clone external github repo
- `README.md` - For human to read on how to install and use the repo
- `requirement.txt` - Store the required libararies

## Working Rules
- For any new dependencies need to be add during coding, add it to requirements.txt
- When planing for the current step, also put the next step into the thinking context
