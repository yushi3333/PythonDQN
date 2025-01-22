import torch
from DQN_Control.replay_buffer import ReplayBuffer
from DQN_Control.model import DQN
from config import action_map, env_params
from utils import *
from environment import SimEnv
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import time

def run():
    try:
        buffer_size = int(1e4)
        batch_size = 32
        state_dim = (128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = len(action_map)
        in_channels = 1
        episodes = 601

        

        # Initialize replay buffer, model, and optimizer
        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = DQN(num_actions, state_dim, in_channels, device)
        

        optimizer = model.Q_optimizer  # The optimizer is part of the DQN class

        # Load model and optimizer checkpoints if they exist
        checkpoint_path = 'weights/model_ep_600'
        start_episode = 0  # Start episode counter
        if os.path.exists(checkpoint_path + "_Q"):
            model.load(checkpoint_path)
            print(f"Loaded model from {checkpoint_path}")
            start_episode = start_episode + 1  # Set this to the next episode after the checkpoint
            print(f"Starting from episode {start_episode}")

    
        env = SimEnv(visuals=False, **env_params)

        # Prepare CSV file to log results
        csv_file = 'training_logs.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['Episode', 'Cumulative Reward', 'Loss Value'])

        episodes_reward = []
        task_timings = []  # To store timings for Gantt chart

        episodes_reward = []

        csv_file_optimizer = 'optimizer_logs.csv'
        with open(csv_file_optimizer, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Batch', 'Learning Rate', 'Param Name', 'Gradient', 'Update Magnitude'])

        for ep in range(start_episode, episodes):

            episode_start_time = time.time()

            # Task 1: Actor Creation
            t1_start = time.time()
            env.create_actors()
            t1_end = time.time()

             # Task 2: Generate Episode
            t2_start = time.time()
            cumulative_reward = env.generate_episode(model, replay_buffer, ep, action_map, eval=False)
            t2_end = time.time()

            
             # Task 3: Compute Loss
            t3_start = time.time()
            print("Replay buffer length: ", len(replay_buffer))
            if len(replay_buffer) > batch_size:
                loss = model.compute_loss(replay_buffer)
                optimizer.zero_grad()
                loss.backward()

                # Log optimizer values before the step
                param_before = {name: param.data.clone() for name, param in model.Q.named_parameters()}

                # Compute the loss
                loss = model.compute_loss(replay_buffer)
                optimizer.zero_grad()
                loss.backward()

                # Apply the optimizer step
                optimizer.step()
                            
                

                # After optimizer step, log parameter updates
                with open(csv_file_optimizer, mode='a', newline='') as file:
                    writer = csv.writer(file)

                    for name, param in model.Q.named_parameters():
                        if param.grad is not None:
                            grad_value = param.grad.abs().mean().item()  # Average gradient value
                            update_magnitude = (param.data - param_before[name]).abs().mean().item()  # Magnitude of the update
            
                            # Log or print these values as needed
                            print(f"Parameter: {name}, Gradient: {grad_value}, Update Magnitude: {update_magnitude}")

                            # Write to CSV
                            writer.writerow([ep, "N/A", name, grad_value, update_magnitude])

                loss_value = loss.item()

            else:
                loss_value = None
            t3_end = time.time()

            # Task 5: Environment Reset
            t5_start = time.time()
            env.reset()
            t5_end = time.time()

            # Log timings
            task_timings.append({
                "Episode": ep,
                "Task": "Actor Creation",
                "Start": t1_start - episode_start_time,
                "End": t1_end - episode_start_time
            })
            task_timings.append({
                "Episode": ep,
                "Task": "Generate Episode",
                "Start": t2_start - episode_start_time,
                "End": t2_end - episode_start_time
            })
            task_timings.append({
                "Episode": ep,
                "Task": "Compute Loss",
                "Start": t3_start - episode_start_time,
                "End": t3_end - episode_start_time
            })
            task_timings.append({
                "Episode": ep,
                "Task": "Environment Reset",
                "Start": t5_start - episode_start_time,
                "End": t5_end - episode_start_time
            })

            # Log cumulative reward and loss to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([ep, cumulative_reward, loss_value])

            episodes_reward.append(cumulative_reward)
            print(f"Episode {ep}: Cumulative Reward = {cumulative_reward}")

            # Save model checkpoint periodically
            
            #if ep % 200 == 0:
                #model.save(f'weights/model_ep_{ep}.pth')
                #print(f"Checkpoint saved at episode {ep}")

            
        plot_rewards(episodes_reward,start_episode, episodes)
        plot_gantt_chart(task_timings)

    
    finally:

        env.quit()
def plot_rewards(rewards, start_episode, episodes):
    
    plt.figure(figsize=(10, 6))
    save_path = f'images/cumulative_rewards_{start_episode}_to_{episodes}.png'
    plt.plot(rewards, label=f'Cumulative Reward per Episode {start_episode} - {episodes}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Performance')
    plt.legend()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

def plot_gantt_chart(task_timings):
    # Convert task timings to a DataFrame
    df = pd.DataFrame(task_timings)

    # Plot Gantt chart
    plt.figure(figsize=(12, 8))
    for i, (episode, group) in enumerate(df.groupby('Episode')):
        for _, task in group.iterrows():
            plt.barh(
                y=f"Episode {episode}",
                left=task['Start'],
                width=task['End'] - task['Start'],
                label=task['Task'] if i == 0 else "",
                edgecolor='black'
            )

    plt.xlabel('Time (seconds)')
    plt.ylabel('Episodes')
    plt.title('Task Timings (Gantt Chart)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/task_gantt_chart.png')
    print("Gantt chart saved to images/task_gantt_chart.png")
    plt.show()

if __name__ == "__main__":
    run()