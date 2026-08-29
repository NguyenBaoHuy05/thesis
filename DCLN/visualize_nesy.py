import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_dcln import DCLN

# ─── Lists of Concept & Class Names ──────────────────────────────────────────

SPATIAL_NAMES = [
    # head
    'head_nod', 'head_shake', 'head_tilt_down', 'head_tilt_up', 'head_rotate', 'head_static',
    # hand
    'hand_grasp', 'hand_release', 'hand_wave', 'hand_point', 'hand_clap', 'hand_manipulate', 'hand_fist', 'hand_open', 'hand_touch', 'hand_lift_to_face', 'hand_circular',
    # arm
    'arm_raise', 'arm_lower', 'arm_extend', 'arm_retract', 'arm_circular', 'arm_swing', 'arm_cross', 'arm_hold', 'arm_alternating', 'arm_overhead', 'arm_lateral',
    # hip
    'hip_bend', 'hip_extend', 'hip_sit', 'hip_squat', 'hip_rotate', 'hip_static_stand', 'hip_static_sit', 'hip_sway',
    # leg
    'leg_step', 'leg_kick', 'leg_jump', 'leg_squat', 'leg_lift', 'leg_static_stand', 'leg_static_sit', 'leg_alternating', 'leg_balance',
    # foot
    'foot_step', 'foot_kick', 'foot_jump', 'foot_pivot', 'foot_static', 'foot_tap', 'foot_slide',
    # interaction
    'interaction_two_person'
]

TEMPORAL_NAMES = [
    # direction
    'temp_dir_upward', 'temp_dir_downward', 'temp_dir_forward', 'temp_dir_backward', 'temp_dir_converging', 'temp_dir_diverging', 'temp_dir_reversible',
    # sequence
    'temp_seq_hands_first', 'temp_seq_legs_first', 'temp_seq_body_first', 'temp_seq_simultaneous', 'temp_seq_alternating', 'temp_seq_cascading',
    # dynamics
    'temp_dyn_slow', 'temp_dyn_fast', 'temp_dyn_accelerating', 'temp_dyn_decelerating', 'temp_dyn_rhythm_regular', 'temp_dyn_rhythm_irregular', 'temp_dyn_duration_brief', 'temp_dyn_duration_sustained'
]

CLASS_NAMES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up",
    "clapping", "reading", "writing", "tear up paper", "put on jacket", "take off jacket", "put on a shoe",
    "take off a shoe", "put on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap",
    "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping", "jump up", "phone call",
    "play with phone/tablet", "type on a keyboard", "point to something", "taking a selfie", "check time",
    "rub two hands", "nod head/bow", "shake head", "wipe face", "salute", "put palms together", "cross hands in front",
    "sneeze/cough", "staggering", "falling down", "headache", "chest pain", "back pain", "neck pain",
    "nausea/vomiting", "fan self", "punch/slap", "kicking", "pushing", "pat on back", "point finger", "hugging",
    "giving object", "touch pocket", "shaking hands", "walking towards", "walking apart"
]

def main():
    parser = argparse.ArgumentParser(description="Visualize Neuro-Symbolic routing weights")
    parser.add_argument('--data_path', default='/kaggle/working/NTU60_CS.npz', help='Path to NTU60 sampled dataset')
    parser.add_argument('--checkpoint', default='./epoch60_99.98.pt', help='Path to trained model checkpoint')
    parser.add_argument('--out_dir', default='./visualizations', help='Output directory for saving plots')
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 9, 21, 23], help='Class indices to visualize')
    parser.add_argument('--class_idx', type=int, default=None, help='Specific class index')
    parser.add_argument('--sample_idx', type=int, default=None, help='Specific sample index')
    parser.add_argument('--frame_idx', type=int, default=None, help='Specific frame index (parsed but not directly used)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.class_idx is not None:
        args.classes = [args.class_idx]
    
    # 1. Initialize and load model
    print(f"Loading model checkpoint from {args.checkpoint}...")
    model = DCLN(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph_ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'}
    )
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load dataset
    print(f"Loading dataset from {args.data_path}...")
    if not os.path.exists(args.data_path):
        # Check local path fallback
        fallback = 'd:/Thesis/NTU60_CS_sampled64.npz'
        if os.path.exists(fallback):
            args.data_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {args.data_path} or {fallback}")
            
    dataset = np.load(args.data_path)
    x_test = dataset['x_test']
    y_test = dataset['y_test']
    
    if x_test.ndim == 3:
        N, T, _ = x_test.shape
        x_test = x_test.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        
    y_test = np.where(y_test > 0)[1] if y_test.ndim > 1 else y_test
    print("Dataset loaded and reshaped successfully.")

    # 3. Process each requested class
    for class_id in args.classes:
        class_name = CLASS_NAMES[class_id]
        print(f"\nProcessing class: A{class_id+1} ({class_name})...")
        
        # Find first sample index matching this class
        match_idx = np.where(y_test == class_id)[0]
        if len(match_idx) == 0:
            print(f"Warning: No samples found for class index {class_id}")
            continue
            
        if args.sample_idx is not None:
            sample_idx = args.sample_idx
            print(f"Using specified sample_idx: {sample_idx}")
        else:
            sample_idx = match_idx[0]
            print(f"Using first matching sample_idx: {sample_idx}")
            
        sample_data = x_test[sample_idx] # shape: (3, 64, 25, 2)
        
        # Format sample for model forward pass -> (1, 3, 64, 25, 2)
        input_tensor = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass (without label to get prediction)
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            
            # Retrieve routing values from self.metadata
            metadata = model.metadata
            routing_spatial = metadata['routing_spatial'][0].cpu().numpy() # (53,)
            routing_temporal = metadata['routing_temporal'][0].cpu().numpy() # (21,)

        print(f"  -> Predicted class A{pred_class+1} ({CLASS_NAMES[pred_class]})")

        # 4. Print top active concepts in console
        top_spatial = sorted(zip(SPATIAL_NAMES, routing_spatial), key=lambda x: x[1], reverse=True)[:5]
        top_temporal = sorted(zip(TEMPORAL_NAMES, routing_temporal), key=lambda x: x[1], reverse=True)[:5]
        
        print("  Top Spatial Concepts:")
        for name, score in top_spatial:
            print(f"    - {name}: {score:.4f}")
            
        print("  Top Temporal Concepts:")
        for name, score in top_temporal:
            print(f"    - {name}: {score:.4f}")

        # 5. Create double bar chart visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"Neuro-Symbolic Routing Analysis: A{class_id+1} - {class_name.upper()}\n(Prediction: {CLASS_NAMES[pred_class].upper()})", fontsize=14, fontweight='bold')
        
        # Plot Spatial
        ax1.bar(SPATIAL_NAMES, routing_spatial, color='royalblue', alpha=0.8, edgecolor='black')
        ax1.set_ylabel("Activation Score", fontsize=11)
        ax1.set_title("Spatial Concept Routing (Joint Groups)", fontsize=12, fontweight='bold')
        ax1.set_xticklabels(SPATIAL_NAMES, rotation=90, fontsize=8)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot Temporal
        ax2.bar(TEMPORAL_NAMES, routing_temporal, color='crimson', alpha=0.8, edgecolor='black')
        ax2.set_ylabel("Activation Score", fontsize=11)
        ax2.set_title("Temporal Concept Routing (Dynamics & Sequences)", fontsize=12, fontweight='bold')
        ax2.set_xticklabels(TEMPORAL_NAMES, rotation=45, ha='right', fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, f"class_{class_id+1}_{class_name.replace('/', '_').replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  -> Saved visualization plot to {plot_path}")

if __name__ == '__main__':
    main()
