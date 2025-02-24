import gymnasium as gym
import pygame
import numpy as np
import torch
import cv2


# Neural Network Definition (must match training architecture)
class Network(torch.nn.Module):
    """Q-network architecture for LunarLander decision making"""

    def __init__(self, state_size, action_size):
        super().__init__()
        # Define network layers
        self.fc1 = torch.nn.Linear(state_size, 64)  # Input to first hidden layer
        self.fc2 = torch.nn.Linear(64, 64)  # Hidden to hidden layer
        self.fc3 = torch.nn.Linear(64, action_size)  # Hidden to output layer

    def forward(self, x):
        """Forward pass through the network"""
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Second ReLU activation
        return self.fc3(x)  # Output raw action values


# Environment and Model Initialization
env = gym.make('LunarLander-v3', render_mode='rgb_array')  # Create environment
state_size = env.observation_space.shape[0]  # State space dimension (8)
action_size = env.action_space.n  # Action space size (4)

# Load trained model
model = Network(state_size, action_size)
model.load_state_dict(torch.load('lunar_lander.pth'))  # Load trained weights
model.eval()  # Set model to evaluation mode

# Pygame Initialization
pygame.init()
screen_width, screen_height = 1000, 800  # Set display resolution
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Lunar Lander-V3 Demonstration")  # Window title
clock = pygame.time.Clock()  # For controlling frame rate

# Video Recording Configuration
video_filename = "lunar_lander_recording.mp4"
fps = 30  # Frames per second
frame_size = (screen_width, screen_height)  # Video resolution
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec (MPEG-4)
writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

# Font Initialization for HUD display
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)  # Font for on-screen text

# Demonstration Execution
max_episodes = 5  # Number of episodes to record
episode_count = 0  # Episode counter

try:
    while episode_count < max_episodes:
        state, _ = env.reset()  # Reset environment for new episode
        total_reward = 0  # Cumulative reward tracker
        done = False  # Episode completion flag
        landed = False  # Landing success flag

        while not done:
            # Handle window close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt  # Allow graceful exit

            # Model Prediction (no gradient calculation needed)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
                action = model(state_tensor).argmax().item()  # Choose best action

            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Episode end condition
            total_reward += reward  # Accumulate reward

            # Environment Rendering
            frame = env.render()  # Get current frame from environment
            frame = np.transpose(frame, (1, 0, 2))  # Adjust axis order for Pygame

            # Pygame Display Setup
            frame_surface = pygame.surfarray.make_surface(frame)  # Create surface
            frame_surface = pygame.transform.scale(frame_surface,
                                                   (screen_width, screen_height))
            screen.blit(frame_surface, (0, 0))  # Draw frame to screen

            # Display Velocity Information
            xdot, ydot = state[2], state[3]  # Extract velocity from state
            speed_text = font.render(f"Speed: x={xdot:.2f}, y={ydot:.2f}",
                                     True, (255, 255, 255))  # White text
            screen.blit(speed_text, (10, 10))  # Position text in top-left

            # Display Landing Result
            if done:
                landed = reward > 0  # Determine landing success
                result_text = font.render(
                    "Successfully Landed!" if landed else "Landing Failed!",
                    True,
                    (0, 255, 0) if landed else (255, 0, 0)  # Green/Red color
                )
                screen.blit(result_text, (10, 50))  # Position result text

            pygame.display.flip()  # Update full display

            # Video Frame Processing
            frame_cv2 = pygame.surfarray.array3d(frame_surface)  # Get pixel data
            frame_cv2 = np.transpose(frame_cv2, (1, 0, 2))  # Adjust axis for OpenCV
            frame_cv2 = cv2.cvtColor(frame_cv2, cv2.COLOR_RGB2BGR)  # Convert to BGR

            writer.write(frame_cv2)  # Write frame to video file

            state = next_state  # Update to new state
            clock.tick(fps)  # Maintain consistent frame rate

        print(f"Episode {episode_count + 1} completed. Total score: {total_reward:.2f}")
        episode_count += 1

except KeyboardInterrupt:
    print("Recording interrupted by user")
finally:
    # Cleanup Resources
    writer.release()  # Finalize video file
    pygame.quit()  # Close Pygame window
    env.close()  # Properly close environment
    print(f"Video saved to {video_filename}")