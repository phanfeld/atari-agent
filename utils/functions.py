import torch
import numpy as np
import imageio
from skimage.transform import resize


def preprocess(frame):
    """
    Reduces the input frame to grayscale and ensures storing the data in an appropriate data type.
    Args:
        frame: a (210, 160, 3) frame of the Atari environment
    Returns:
        a (210, 160) frame, pixel values between 0 and 255, stored as uint8.
        
    """               
    return frame.mean(2).astype(np.uint8)

def frame_to_tensor(frame):
    """
    Turn the frame into a PyTorch tensor for forwarding it through a network. This is done shortly
    before the forward pass to reduce memory usage.
    Args:
        frame: a (210, 160) frame, pixel values between 0 and 255
        device: a PyTorch device
    Returns:
        a (210,160) tensor on the specified device
    """
    frame = frame / 255.
    tensor = torch.tensor(frame, dtype=torch.float32, requires_grad=False)
    return tensor

def clip_reward(reward):
    """
    This function clips the reward that is returned by the Atari environment to be between -1, 0 and 1. Clipping the
    reward is recommended for training DQN agents.
    Args:
        reward: a integer returned by the Atari environment
    Returns:
        an integer, either -1, 0 or 1
    """
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1


def generate_gif(frame_number, frames_for_gif, reward, path):
    """
    Generate gifs for visual progress evaluation.
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}', frames_for_gif, duration=1/30)