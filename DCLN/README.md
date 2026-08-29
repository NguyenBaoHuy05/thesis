!python visualize_skeleton.py --data_path ../NTU60_CS_sampled64.npz --checkpoint epoch60_99.98.pt --out_dir ./visualizations --classes 0 23 --frames 10 30 50

!python visualize_attention.py --data_path ../NTU60_CS_sampled64.npz --checkpoint epoch60_99.98.pt --out_dir ./visualizations --class_idx 1 --sample_idx 10 --frame_idx 55

!python visualize_nesy.py --data_path ../NTU60_CS_sampled64.npz --checkpoint epoch60_99.98.pt --out_dir ./visualizations --class_idx 1 --sample_idx 10 --frame_idx 55

