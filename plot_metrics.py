import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(iteration_losses, total_iterations, num_epochs, dataloader_size,
                 jsd_history=None, fred_history=None, amp_history=None,
                 plot_jsd=True, plot_fred=False, plot_amp=False):
    """
    Unified plotting function for all metrics
    
    Args:
        iteration_losses: Dictionary of iteration-level losses
        total_iterations: Total number of training iterations
        num_epochs: Total number of epochs
        dataloader_size: Size of dataloader (batches per epoch)
        jsd_history: List of JSD scores per epoch (optional)
        fred_history: List of FReD scores per epoch (optional)
        amp_history: List of AMP scores per epoch (optional)
        plot_jsd: Whether to plot JSD score
        plot_fred: Whether to plot FReD score
        plot_amp: Whether to plot AMP score
    """
    
    # Count how many metric plots we need
    num_metric_plots = sum([plot_jsd, plot_fred, plot_amp])
    total_subplots = 4 + num_metric_plots  # 4 base plots + metric plots
    
    plt.figure(figsize=(12, 5 * total_subplots))
    current_subplot = 1

    # First subplot - Total losses only per iteration
    plt.subplot(total_subplots, 1, current_subplot)
    iterations = range(total_iterations)
    plt.plot(iterations, iteration_losses['total_d_loss'], label='D Total Loss', color='blue')
    plt.plot(iterations, iteration_losses['total_g_loss'], label='G Total Loss', color='red')
    plt.title('Generator and Discriminator Total Losses per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    current_subplot += 1

    # Second subplot - Total losses only per epoch
    plt.subplot(total_subplots, 1, current_subplot)
    epochs = range(num_epochs)
    d_losses_per_epoch = []
    g_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size

        d_epoch_loss = np.mean(iteration_losses['total_d_loss'][start_idx:end_idx])
        g_epoch_loss = np.mean(iteration_losses['total_g_loss'][start_idx:end_idx])

        d_losses_per_epoch.append(d_epoch_loss)
        g_losses_per_epoch.append(g_epoch_loss)

    plt.plot(epochs, d_losses_per_epoch, label='D Total Loss', color='blue')
    plt.plot(epochs, g_losses_per_epoch, label='G Total Loss', color='red')
    plt.title('Generator and Discriminator Total Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    current_subplot += 1

    # Third subplot - Five losses per iteration (including gradient penalty)
    plt.subplot(total_subplots, 1, current_subplot)
    plt.plot(iterations, iteration_losses['dec_loss'], label='Decoder Loss', color='orange')
    plt.plot(iterations, iteration_losses['pixel_loss'], label='Pixel Loss', color='magenta')
    plt.plot(iterations, iteration_losses['wasserstein_loss'], label='Wasserstein Loss', color='cyan')
    plt.plot(iterations, iteration_losses['g_loss'], label='G Loss', color='green')
    plt.plot(iterations, iteration_losses['gradient_penalty'], label='Gradient Penalty', color='purple', linestyle=':')
    plt.title('Individual Loss Components per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    current_subplot += 1

    # Fourth subplot - Five losses per epoch (including gradient penalty)
    plt.subplot(total_subplots, 1, current_subplot)
    dec_losses_per_epoch = []
    pixel_losses_per_epoch = []
    wasserstein_losses_per_epoch = []
    g_losses_per_epoch = []
    gp_losses_per_epoch = []

    for epoch in epochs:
        start_idx = epoch * dataloader_size
        end_idx = (epoch + 1) * dataloader_size
        
        dec_losses_per_epoch.append(np.mean(iteration_losses['dec_loss'][start_idx:end_idx]))
        pixel_losses_per_epoch.append(np.mean(iteration_losses['pixel_loss'][start_idx:end_idx]))
        wasserstein_losses_per_epoch.append(np.mean(iteration_losses['wasserstein_loss'][start_idx:end_idx]))
        g_losses_per_epoch.append(np.mean(iteration_losses['g_loss'][start_idx:end_idx]))
        gp_losses_per_epoch.append(np.mean(iteration_losses['gradient_penalty'][start_idx:end_idx]))

    plt.plot(epochs, dec_losses_per_epoch, label='Decoder Loss', color='orange')
    plt.plot(epochs, pixel_losses_per_epoch, label='Pixel Loss', color='magenta')
    plt.plot(epochs, wasserstein_losses_per_epoch, label='Wasserstein Loss', color='cyan')
    plt.plot(epochs, g_losses_per_epoch, label='G Loss', color='green')
    plt.plot(epochs, gp_losses_per_epoch, label='Gradient Penalty', color='purple', linestyle=':')
    plt.title('Individual Loss Components per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    current_subplot += 1

    # JSD subplot (if requested)
    if plot_jsd and jsd_history is not None:
        plt.subplot(total_subplots, 1, current_subplot)
        jsd_epochs = range(len(jsd_history))
        plt.plot(jsd_epochs, jsd_history, 'g-', label='JSD Score', linewidth=2)
        plt.title('JSD Score Progress (lower is better)')
        plt.xlabel('Epoch')
        plt.ylabel('JSD Score')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        current_subplot += 1

    # FReD subplot (if requested)
    if plot_fred and fred_history is not None:
        plt.subplot(total_subplots, 1, current_subplot)
        fred_epochs = range(len(fred_history))
        plt.plot(fred_epochs, fred_history, 'm-', label='FReD Score', linewidth=2)
        plt.title('FReD Score Progress')
        plt.xlabel('Epoch')
        plt.ylabel('FReD Score')
        plt.ylim(0, 5)  # Adjust based on your actual FReD values
        plt.legend()
        plt.grid(True)
        current_subplot += 1

    # AMP subplot (if requested)
    if plot_amp and amp_history is not None:
        plt.subplot(total_subplots, 1, current_subplot)
        amp_epochs = range(len(amp_history))
        plt.plot(amp_epochs, amp_history, 'r-', label='AMP Score', linewidth=2)
        plt.title('AMP Score Progress (higher is better)')
        plt.xlabel('Epoch')
        plt.ylabel('AMP Score (%)')
        plt.ylim(0, 100)
        
        # Add reference lines
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        plt.axhline(y=75, color='gray', linestyle=':', alpha=0.5, label='75% threshold')
        
        plt.legend()
        plt.grid(True)
        current_subplot += 1

    plt.tight_layout()
    
    # Save with appropriate filename
    metrics_in_filename = []
    if plot_jsd: metrics_in_filename.append('jsd')
    if plot_fred: metrics_in_filename.append('fred')
    if plot_amp: metrics_in_filename.append('amp')
    
    filename = f"training_progress_{'_'.join(metrics_in_filename)}.png" if metrics_in_filename else "training_progress.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()