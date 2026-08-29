import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model_dcln import DCLN

# ─── Joint Groups Mapping ────────────────────────────────────────────────────

JOINT_GROUPS = {
    'head': [2, 3],
    'hand': [6, 7, 10, 11, 21, 22, 23, 24],
    'arm': [4, 5, 6, 7, 8, 9, 10, 11],
    'hip': [0, 1, 12, 16],
    'leg': [12, 13, 14, 15, 16, 17, 18, 19],
    'foot': [14, 15, 18, 19],
    'interaction_two_person': list(range(25))
}

CONCEPT_MAPPING = [
    'head', 'head', 'head', 'head', 'head', 'head',
    'hand', 'hand', 'hand', 'hand', 'hand', 'hand', 'hand', 'hand', 'hand', 'hand', 'hand',
    'arm', 'arm', 'arm', 'arm', 'arm', 'arm', 'arm', 'arm', 'arm', 'arm', 'arm',
    'hip', 'hip', 'hip', 'hip', 'hip', 'hip', 'hip', 'hip',
    'leg', 'leg', 'leg', 'leg', 'leg', 'leg', 'leg', 'leg', 'leg',
    'foot', 'foot', 'foot', 'foot', 'foot', 'foot', 'foot',
    'interaction_two_person'
]

# Skeleton bones connectivity (1-indexed to 0-indexed)
BONES = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
]
BONES = [(u - 1, v - 1) for (u, v) in BONES]

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
    parser = argparse.ArgumentParser(description="Visualize 3D Skeleton with Concept Highlights")
    parser.add_argument('--data_path', default='../NTU60_CS_sampled64.npz', help='Path to dataset')
    parser.add_argument('--checkpoint', default='./epoch60_99.98.pt', help='Path to checkpoint')
    parser.add_argument('--out_dir', default='./visualizations', help='Output directory')
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 23], help='Classes to visualize')
    parser.add_argument('--frames', type=int, nargs='+', default=[10, 30, 50], help='Key frame indices (0-63)')
    parser.add_argument('--class_idx', type=int, default=None, help='Specific class index')
    parser.add_argument('--sample_idx', type=int, default=None, help='Specific sample index')
    parser.add_argument('--frame_idx', type=int, default=None, help='Specific frame index')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.class_idx is not None:
        args.classes = [args.class_idx]
    if args.frame_idx is not None:
        args.frames = [args.frame_idx]

    # 1. Load model
    print("Loading model and checkpoint...")
    model = DCLN(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph_ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'}
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    # 2. Load data
    print("Loading dataset...")
    dataset = np.load(args.data_path)
    x_test = dataset['x_test']
    y_test = dataset['y_test']
    
    if x_test.ndim == 3:
        N, T, _ = x_test.shape
        x_test = x_test.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        
    y_test = np.where(y_test > 0)[1] if y_test.ndim > 1 else y_test

    # 3. Process each class
    for class_id in args.classes:
        class_name = CLASS_NAMES[class_id]
        print(f"\nVisualizing skeleton for: A{class_id+1} ({class_name})")
        
        match_idx = np.where(y_test == class_id)[0]
        if len(match_idx) == 0:
            continue
            
        if args.sample_idx is not None:
            sample_idx = args.sample_idx
            print(f"Using specified sample_idx: {sample_idx}")
        else:
            sample_idx = match_idx[0]
            print(f"Using first matching sample_idx: {sample_idx}")
            
        sample_data = x_test[sample_idx] # (3, 64, 25, 2)
        
        # Get routing weights from model
        input_tensor = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            metadata = model.metadata
            routing_spatial = metadata['routing_spatial'][0].cpu().numpy() # (53,)

        # Calculate joint weights based on routing scores (exclude generic interaction group)
        joint_weights = np.zeros(25)
        for s in range(52): # Exclude last concept index (interaction) for better contrast
            score = routing_spatial[s]
            joints = JOINT_GROUPS[CONCEPT_MAPPING[s]]
            for j in joints:
                joint_weights[j] += score

        # Normalize weights to range [0, 1] for coloring
        if joint_weights.max() > 0:
            joint_weights = (joint_weights - joint_weights.min()) / (joint_weights.max() - joint_weights.min() + 1e-6)

        # Plot 3 key frames side-by-side
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(f"Skeletal Motion & Concept Activation: A{class_id+1} - {class_name.upper()}\n(Highlighting joints with high routing scores)", fontsize=14, fontweight='bold')
        
        for idx, f_idx in enumerate(args.frames):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            ax.set_title(f"Frame {f_idx}", fontsize=11, fontweight='bold')
            
            # Extract coordinates for person 0
            x_coords = sample_data[0, f_idx, :, 0]
            y_coords = sample_data[1, f_idx, :, 0]
            z_coords = sample_data[2, f_idx, :, 0]
            
            # Map NTU coordinates (X: left-right, Y: up-down, Z: depth) to Matplotlib 3D axes
            # Matplotlib Z is height, so map NTU Y -> Matplotlib Z, and NTU Z -> Matplotlib Y
            x_plt = x_coords
            y_plt = z_coords
            z_plt = y_coords
            
            # Color joints based on weights
            # Use 'coolwarm' or 'Reds' colormap. Red represents high concept activation, blue represents low activation.
            sc = ax.scatter(x_plt, y_plt, z_plt, c=joint_weights, cmap='coolwarm', s=120, edgecolors='black', depthshade=False, vmin=0, vmax=1)
            
            # Draw bones connecting joints
            for u, v in BONES:
                bone_weight = (joint_weights[u] + joint_weights[v]) / 2.0
                # Interpolate color from coolwarm map
                color = plt.cm.coolwarm(bone_weight)
                line_width = 1.5 + 3.0 * bone_weight
                ax.plot([x_plt[u], x_plt[v]], 
                        [y_plt[u], y_plt[v]], 
                        [z_plt[u], z_plt[v]], 
                        color=color, linewidth=line_width)
            
            # Normalize view angles
            ax.view_init(elev=15, azim=-60)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.grid(False)
            
        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.2, 0.02, 0.6])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), cax=cbar_ax)
        cb.set_label('Concept Activation Strength', fontsize=11, fontweight='bold')
        
        plot_path = os.path.join(args.out_dir, f"skeleton_class_{class_id+1}_{class_name.replace('/', '_').replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved skeleton plot to {plot_path}")

if __name__ == '__main__':
    main()
