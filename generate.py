"""
Generate mouse trajectories using trained Conditional LSTM-VAE
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from config import Config
from model import ConditionalLSTMVAE


def load_model(checkpoint_path, num_users, config):
    """Load trained model from checkpoint"""
    model = ConditionalLSTMVAE(config, num_users).to(config.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Final loss: {checkpoint['loss']:.4f}")

    return model


def create_condition(user_id, time_period, action_type, start_pos, end_pos, config):
    """
    Create condition tensor for generation

    Args:
        user_id: Integer user ID
        time_period: Integer (0: morning, 1: afternoon, 2: evening, 3: night)
        action_type: Integer (0: move, 1: click, 2: drag)
        start_pos: Tuple (x, y) in pixels
        end_pos: Tuple (x, y) in pixels
        config: Config object

    Returns:
        condition: Dictionary of condition tensors
    """
    # Normalize positions
    start_x = start_pos[0] / config.SCREEN_WIDTH
    start_y = start_pos[1] / config.SCREEN_HEIGHT
    end_x = end_pos[0] / config.SCREEN_WIDTH
    end_y = end_pos[1] / config.SCREEN_HEIGHT

    condition = {
        'user_id': torch.LongTensor([[user_id]]).to(config.DEVICE),
        'time_period': torch.LongTensor([[time_period]]).to(config.DEVICE),
        'action_type': torch.LongTensor([[action_type]]).to(config.DEVICE),
        'start_pos': torch.FloatTensor([[start_x, start_y]]).to(config.DEVICE),
        'end_pos': torch.FloatTensor([[end_x, end_y]]).to(config.DEVICE)
    }

    return condition


def decode_trajectory(trajectory, start_pos, config):
    """
    Decode trajectory features back to absolute coordinates

    Args:
        trajectory: Tensor of shape (seq_len, input_dim)
        start_pos: Tuple (x, y) starting position in pixels
        config: Config object

    Returns:
        events: List of mouse events with absolute coordinates
    """
    trajectory = trajectory.cpu().numpy()
    events = []

    current_x = start_pos[0] / config.SCREEN_WIDTH
    current_y = start_pos[1] / config.SCREEN_HEIGHT
    current_time = 0.0

    for i, feature in enumerate(trajectory):
        delta_t, delta_x, delta_y, speed_norm, accel_norm, button_norm, state_norm = feature

        # Denormalize speed and acceleration (not used in reconstruction but kept for debugging)
        speed = speed_norm * 10.0
        accel = accel_norm * 200.0 - 100.0

        # Update position
        current_x += delta_x
        current_y += delta_y
        current_time += abs(delta_t)

        # Clip to screen bounds
        current_x = np.clip(current_x, 0, 1)
        current_y = np.clip(current_y, 0, 1)

        # Convert back to pixels
        x_pixel = int(current_x * config.SCREEN_WIDTH)
        y_pixel = int(current_y * config.SCREEN_HEIGHT)

        # Denormalize and decode button and state
        button_denorm = button_norm * 3.0
        state_denorm = state_norm * 2.0
        button_int = int(np.round(button_denorm))
        state_int = int(np.round(state_denorm))

        button_map_inv = {v: k for k, v in config.BUTTON_MAP.items()}
        state_map_inv = {v: k for k, v in config.STATE_MAP.items()}

        button_str = button_map_inv.get(button_int, 'None')
        state_str = state_map_inv.get(state_int, 'Move')

        events.append({
            'timestamp': current_time,
            'x': x_pixel,
            'y': y_pixel,
            'button': button_str,
            'state': state_str
        })

    return events


def plot_trajectory(events, title, save_path=None):
    """Plot mouse trajectory"""
    x_coords = [e['x'] for e in events]
    y_coords = [e['y'] for e in events]

    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=1)
    plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
    plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='x', label='End', zorder=5)

    # Mark click events
    click_events = [e for e in events if e['state'] in ['Pressed', 'Released']]
    if click_events:
        click_x = [e['x'] for e in click_events]
        click_y = [e['y'] for e in click_events]
        plt.scatter(click_x, click_y, c='orange', s=50, marker='^', label='Click', zorder=4)

    plt.xlim(0, Config.SCREEN_WIDTH)
    plt.ylim(0, Config.SCREEN_HEIGHT)
    plt.gca().invert_yaxis()  # Invert y-axis to match screen coordinates
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def save_trajectory_csv(events, filename):
    """Save trajectory to CSV file"""
    df = pd.DataFrame(events)
    df.to_csv(filename, index=False)
    print(f"Trajectory saved to {filename}")


def generate_trajectories(model, conditions_list, config, num_samples=5, temperature=1.0):
    """
    Generate multiple trajectories for given conditions

    Args:
        model: Trained model
        conditions_list: List of condition dictionaries
        config: Config object
        num_samples: Number of samples per condition
        temperature: Sampling temperature

    Returns:
        all_trajectories: List of generated trajectories
    """
    all_trajectories = []

    with torch.no_grad():
        for cond_idx, condition_spec in enumerate(conditions_list):
            print(f"\nGenerating trajectories for condition {cond_idx + 1}/{len(conditions_list)}:")
            print(f"  User ID: {condition_spec['user_id']}")
            print(f"  Time period: {['morning', 'afternoon', 'evening', 'night'][condition_spec['time_period']]}")
            print(f"  Action type: {['move', 'click', 'drag'][condition_spec['action_type']]}")
            print(f"  Start: {condition_spec['start_pos']}")
            print(f"  End: {condition_spec['end_pos']}")

            for sample_idx in range(num_samples):
                # Create condition tensor
                condition = create_condition(
                    condition_spec['user_id'],
                    condition_spec['time_period'],
                    condition_spec['action_type'],
                    condition_spec['start_pos'],
                    condition_spec['end_pos'],
                    config
                )

                # Generate trajectory
                trajectory = model.generate(
                    condition,
                    config.SEQUENCE_LENGTH,
                    temperature=temperature
                )

                # Decode to events
                events = decode_trajectory(
                    trajectory.squeeze(0),
                    condition_spec['start_pos'],
                    config
                )

                all_trajectories.append({
                    'condition': condition_spec,
                    'events': events,
                    'sample_idx': sample_idx
                })

    return all_trajectories


def main(args):
    config = Config()

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # Load checkpoint to get num_users from saved config
    print("Loading checkpoint to get dataset info...")
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)

    # Try to get num_users from checkpoint
    if 'num_users' in checkpoint:
        num_users = checkpoint['num_users']
        print(f"Found num_users in checkpoint: {num_users}")
    else:
        # Fallback: load from dataset
        print("num_users not in checkpoint, loading from dataset...")
        from dataset import get_dataloaders
        _, _, num_users = get_dataloaders(config)
        print(f"Loaded num_users from dataset: {num_users}")

    model = load_model(args.checkpoint, num_users, config)

    # Define conditions for generation
    # Example: Generate trajectories for different scenarios
    conditions_list = [
        {
            'user_id': 0,
            'time_period': 1,  # afternoon
            'action_type': 0,  # move
            'start_pos': (100, 100),
            'end_pos': (800, 600)
        },
        {
            'user_id': 0,
            'time_period': 2,  # evening
            'action_type': 1,  # click
            'start_pos': (500, 300),
            'end_pos': (520, 320)
        },
        {
            'user_id': 1,
            'time_period': 0,  # morning
            'action_type': 2,  # drag
            'start_pos': (200, 400),
            'end_pos': (700, 500)
        }
    ]

    # Generate trajectories
    print("Generating trajectories...")
    trajectories = generate_trajectories(
        model,
        conditions_list,
        config,
        num_samples=args.num_samples,
        temperature=args.temperature
    )

    # Save and visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, traj_data in enumerate(trajectories):
        cond = traj_data['condition']
        events = traj_data['events']
        sample_idx = traj_data['sample_idx']

        # Create descriptive filename
        time_period_name = ['morning', 'afternoon', 'evening', 'night'][cond['time_period']]
        action_type_name = ['move', 'click', 'drag'][cond['action_type']]

        base_name = f"{timestamp}_user{cond['user_id']}_{time_period_name}_{action_type_name}_sample{sample_idx}"

        # Save CSV
        csv_path = f"outputs/{base_name}.csv"
        save_trajectory_csv(events, csv_path)

        # Plot
        title = f"User {cond['user_id']} | {time_period_name.capitalize()} | {action_type_name.capitalize()}\n"
        title += f"Start: {cond['start_pos']} → End: {cond['end_pos']}"

        plot_path = f"outputs/{base_name}.png"
        plot_trajectory(events, title, plot_path)

    print(f"\nGeneration complete! Generated {len(trajectories)} trajectories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate mouse trajectories')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples per condition')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')

    args = parser.parse_args()
    main(args)
