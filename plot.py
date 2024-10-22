from utils import load_progress, plot
import matplotlib.pyplot as plt

checkpoint_dir = "invp_v5"
progress_data, _ = load_progress("checkpoints/" + checkpoint_dir)
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plot(progress_data, fig, ax, checkpoint_dir, plot_script=True)
