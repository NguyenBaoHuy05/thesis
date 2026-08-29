import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_dcln import DCLN

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
    parser = argparse.ArgumentParser(description="Visualize Attention Maps & Symbolic Priors")
    parser.add_argument('--data_path', default='../NTU60_CS_sampled64.npz', help='Path to dataset')
    parser.add_argument('--checkpoint', default='./epoch60_99.98.pt', help='Path to checkpoint')
    parser.add_argument('--out_dir', default='./visualizations', help='Output directory')
    parser.add_argument('--classes', type=int, nargs='+', default=[0, 9], help='Classes to visualize')
    parser.add_argument('--class_idx', type=int, default=None, help='Specific class index')
    parser.add_argument('--sample_idx', type=int, default=None, help='Specific sample index')
    parser.add_argument('--frame_idx', type=int, default=None, help='Specific frame index (not directly used in attention maps but parsed)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.class_idx is not None:
        args.classes = [args.class_idx]

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
        print(f"\nVisualizing attention maps for: A{class_id+1} ({class_name})")
        
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
        
        input_tensor = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            metadata = model.metadata
            
            base_attn = metadata['base_attn'][0].cpu().numpy()
            relevant_templates = metadata['template_matrix'][0].cpu().numpy()
            attn_map = metadata['attn_map'][0].cpu().numpy()

        # Plot three heatmaps side-by-side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f"Neuro-Symbolic Attention Intervention: A{class_id+1} - {class_name.upper()}", fontsize=14, fontweight='bold')
        
        # 1. Base Attention (Neural)
        im1 = ax1.imshow(base_attn, cmap='viridis')
        ax1.set_title("1. Pure Data-Driven Attention\n(Neural Self-Attention)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Key Index (3 Segments x 25 Joints)", fontsize=9)
        ax1.set_ylabel("Query Index (3 Segments x 25 Joints)", fontsize=9)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. Symbolic Template Prior
        im2 = ax2.imshow(relevant_templates, cmap='coolwarm')
        ax2.set_title("2. Symbolic Concept Prior Map\n(blended from Frozen Banks)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Joint-Time Index", fontsize=9)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. Hybrid Attention (NeSy Intervened)
        im3 = ax3.imshow(attn_map, cmap='viridis')
        ax3.set_title("3. Hybrid NeSy Attention Map\n(Softmax(Neural + Symbolic))", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Joint-Time Index", fontsize=9)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, f"attention_intervention_class_{class_id+1}_{class_name.replace('/', '_').replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved attention intervention plot to {plot_path}")

if __name__ == '__main__':
    main()
