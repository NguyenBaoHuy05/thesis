import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(r'd:\Thesis\DCLN')
from model_dcln import TemporalGraphPRN

def main():
    # Initialize with dummy parameters just to access the methods
    model = TemporalGraphPRN(dim=64, num_classes=60, num_joints=25, num_segments=120)

    # Get the banks
    spatial_masks = model.get_spatial_concepts(25) # Output: (num_spatial, 25, 25)
    temporal_masks = model.get_temporal_concepts(120, 21) # Output: (num_temporal, 3, 3)

    # We have 7 unique spatial groups and 6 unique temporal patterns
    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    fig.suptitle('Full Concept Bank Visualization', fontsize=20, fontweight='bold')

    # --- Spatial Concepts (Row 0) ---
    spatial_indices = {
        'Head': 0, 'Hand': 6, 'Arm': 17, 'Hip': 28, 
        'Leg': 36, 'Foot': 45, 'Interaction (All)': 52
    }
    
    for i, (title, idx) in enumerate(spatial_indices.items()):
        sns.heatmap(spatial_masks[idx].numpy(), ax=axes[0, i], cmap='Blues', cbar=(i==6))
        axes[0, i].set_title(f'Spatial: {title}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

    # --- Temporal Concepts (Row 1) ---
    # Patterns map based on get_temporal_concepts
    temporal_indices = {
        'Pattern 1 (Uniform)': 10,
        'Pattern 2 (Diagonal)': 7,
        'Pattern 3 (Alternating)': 5,
        'Pattern 4 (Middle Focus)': 4,
        'Pattern 5 (Late Focus)': 0,
        'Pattern 6 (Early Focus)': 1
    }
    
    for i, (title, idx) in enumerate(temporal_indices.items()):
        sns.heatmap(temporal_masks[idx].numpy(), ax=axes[1, i], cmap='Oranges', cbar=(i==5), annot=True)
        axes[1, i].set_title(f'Temporal: {title}')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        
    # Hide the last empty subplot in the second row
    axes[1, 6].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to user's visualizations folder
    output_dir = r'D:\Thesis\DCLN\visualizations'
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'concept_bank_full.png')
    plt.savefig(out_file, dpi=150)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
