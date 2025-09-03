import torch
import torch.nn as nn
import os
import time
import numpy as np
from seq_analysis import sample_and_analyze, save_analysis, analyze_sequences
from JSD import jsd
from CutMix import create_cutmix_mask
from amp_evaluator import evaluate_amp_batch

def format_time(seconds):
    """Format seconds into HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calc_gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp=10):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_data.size(0)
    
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    disc_interpolates, _ = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def discriminator_train(discriminator, real_sequences, fake_sequences, mixed_sequences, mask, 
                       optimizer, device, n_chars, lambda_mix=1, lambda_dec=1, scale=1):
    """Single discriminator training step"""
    # Ensure correct shape
    if real_sequences.shape[1] != n_chars:
        real_sequences = real_sequences.transpose(1, 2)
    if fake_sequences.shape[1] != n_chars:
        fake_sequences = fake_sequences.transpose(1, 2)
    if mixed_sequences.shape[1] != n_chars:
        mixed_sequences = mixed_sequences.transpose(1, 2)

    # Get predictions
    real_global, real_pixel = discriminator(real_sequences)
    fake_global, fake_pixel = discriminator(fake_sequences)
    mixed_global, mixed_pixel = discriminator(mixed_sequences)

    # Wasserstein loss
    wasserstein_loss = -torch.mean(real_global) + torch.mean(fake_global)

    # Prepare mask
    mask = mask.squeeze(-1).unsqueeze(1)

    # Decoder loss
    dec_loss = -torch.mean(
        torch.log(real_pixel + 1e-8) +
        torch.log(1 - fake_pixel + 1e-8) +
        mask * torch.log(mixed_pixel + 1e-8) +
        (1 - mask) * torch.log(1 - mixed_pixel + 1e-8)
    ) / 3.0

    # Gradient penalty
    gradient_penalty = calc_gradient_penalty(discriminator, real_sequences, fake_sequences, device)

    # Scale losses
    wasserstein_loss = wasserstein_loss * scale
    gradient_penalty = gradient_penalty * scale

    # Total loss
    total_d_loss = wasserstein_loss + gradient_penalty + lambda_dec * dec_loss
    
    # Update
    optimizer.zero_grad()
    total_d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'total_d_loss': total_d_loss.item(),
        'wasserstein_loss': wasserstein_loss.item(),
        'gradient_penalty': gradient_penalty.item(),
        'dec_loss': dec_loss.item()
    }

def generator_train(generator, discriminator, batch_size, optimizer, device, lambda_dec=1, scale=1):
    """Single generator training step"""
    # Generate fake sequences
    noise = torch.randn(batch_size, 128, device=device)
    fake_sequences = generator(noise)
    fake_sequences = fake_sequences.transpose(1, 2)

    # Get discriminator predictions
    fake_global, fake_pixel = discriminator(fake_sequences)

    pixel_loss = -torch.mean(torch.log(fake_pixel + 1e-8))
    g_loss = -torch.mean(fake_global) * scale                   
    total_g_loss = g_loss + lambda_dec * pixel_loss 
    
    # Update
    optimizer.zero_grad()
    total_g_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'total_g_loss': total_g_loss.item(),
        'g_loss': g_loss.item(),
        'pixel_loss': pixel_loss.item()
    }
       
def train(generator, discriminator, dataloader, num_epochs, 
          n_chars, device, results_dir, d_optimizer, g_optimizer,
          d_scheduler, g_scheduler, d_step=5, g_step=1,
          lambda_dec=1, lambda_mix=1, scale=1):
    """
    Main training function for WGAN-GP
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: DataLoader for real sequences
        num_epochs: Number of training epochs
        n_chars: Number of characters in vocabulary
        device: torch device
        results_dir: Directory to save results
        d_optimizer: Discriminator optimizer
        g_optimizer: Generator optimizer
        d_scheduler: Discriminator learning rate scheduler
        g_scheduler: Generator learning rate scheduler
        d_step: Number of discriminator steps per generator step
        g_step: Number of generator steps
        lambda_mix: Weight for mixed sequences
        lambda_dec: Weight for pixel loss (both in discriminator and generator)
        scale: Scale factor for losses
        
    Returns:
        tuple: (iteration_losses, total_iterations, jsd_history, amp_history)
    """
    
    # Create model directories
    model_save_dir = os.path.join(results_dir, 'saved_models')
    jsd_models_dir = os.path.join(model_save_dir, 'best_jsd')
    orf_models_dir = os.path.join(model_save_dir, 'best_orf')
    amp_models_dir = os.path.join(model_save_dir, 'best_amp')
    
    os.makedirs(jsd_models_dir, exist_ok=True)
    os.makedirs(orf_models_dir, exist_ok=True)
    os.makedirs(amp_models_dir, exist_ok=True)
    
    # Initialize tracking
    jsd_history = []
    amp_history = []
    best_jsd_models = []
    best_orf_models = []
    best_amp_models = []
    
    iteration_losses = {
        'total_d_loss': [],
        'wasserstein_loss': [],
        'gradient_penalty': [],
        'total_g_loss': [],
        'g_loss': [], 
        'dec_loss': [],
        'pixel_loss': []
    }
    
    def update_best_models(score, epoch, model, best_list, maximize=False, max_keep=5):
        """Helper function to update best models list"""
        model_state = model.state_dict()
        if len(best_list) < max_keep:
            best_list.append((score, epoch, model_state))
            best_list.sort(reverse=maximize)
        else:
            if (maximize and score > best_list[-1][0]) or (not maximize and score < best_list[-1][0]):
                best_list[-1] = (score, epoch, model_state)
                best_list.sort(reverse=maximize)
        return best_list

    total_iterations = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_losses = {k: 0 for k in iteration_losses.keys()}
        num_batches = len(dataloader)

        current_lambda_mix = min(lambda_mix, epoch/10)

        for batch_idx, real_sequences in enumerate(dataloader):
            total_iterations += 1

            # Prepare data
            if real_sequences.shape[1] != n_chars:
                real_sequences = real_sequences.transpose(1, 2)
            real_sequences = real_sequences.to(device)
            batch_size = real_sequences.size(0)

            # Generate fake sequences
            noise = torch.randn(batch_size, 128, device=device)
            fake_sequences = generator(noise).detach()
            if fake_sequences.shape[1] != n_chars:
                fake_sequences = fake_sequences.transpose(1, 2)

            # Create mixed sequences
            mask = create_cutmix_mask(real_sequences, lam=0.8)
            mixed_sequences = mask * real_sequences + (1 - mask) * fake_sequences

            # Train discriminator
            d_losses_sum = {k: 0 for k in ['total_d_loss', 'wasserstein_loss', 'gradient_penalty', 'dec_loss']}
            for _ in range(d_step):
                d_losses = discriminator_train(
                    discriminator, real_sequences, fake_sequences, mixed_sequences, mask,
                    d_optimizer, device, n_chars, current_lambda_mix, lambda_dec, scale
                )
                for key in d_losses:
                    d_losses_sum[key] += d_losses[key]

            d_losses_avg = {k: v / d_step for k, v in d_losses_sum.items()}

            # Train generator
            g_losses_sum = {'total_g_loss': 0, 'g_loss': 0, 'pixel_loss': 0}
            for _ in range(g_step):
                g_losses = generator_train(
                    generator, discriminator, batch_size, g_optimizer,
                    device, lambda_dec=lambda_dec, scale=scale
                )
                for key in g_losses:
                    g_losses_sum[key] += g_losses[key]

            g_losses_avg = {k: v / g_step for k, v in g_losses_sum.items()}

            # Update losses
            for key in d_losses_avg:
                iteration_losses[key].append(d_losses_avg[key])
                running_losses[key] += d_losses_avg[key]

            iteration_losses['total_g_loss'].append(g_losses_avg['total_g_loss'])
            iteration_losses['g_loss'].append(g_losses_avg['g_loss'])
            iteration_losses['pixel_loss'].append(g_losses_avg['pixel_loss'])
            running_losses['total_g_loss'] += g_losses_avg['total_g_loss']
            running_losses['g_loss'] += g_losses_avg['g_loss']
            running_losses['pixel_loss'] += g_losses_avg['pixel_loss']

            # Print batch progress
            if batch_idx % 41 == 0:
                print(f'Batch [{batch_idx+1}/{num_batches}]')
                print(f'D_total_loss: {d_losses_avg["total_d_loss"]:.4f}')
                print(f'Wasserstein Loss: {d_losses_avg["wasserstein_loss"]:.4f}')
                print(f'Gradient Penalty: {d_losses_avg["gradient_penalty"]:.4f}')
                print(f'Decoder Loss: {d_losses_avg["dec_loss"]:.4f}')
                print(f'G_total_loss: {g_losses_avg["total_g_loss"]:.4f}')
                print(f'G_loss: {g_losses_avg["g_loss"]:.4f}')
                print(f'Pixel Loss: {g_losses_avg["pixel_loss"]:.4f}\n')

        # Calculate time for this epoch
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        # Calculate epoch averages
        avg_losses = {k: v / num_batches for k, v in running_losses.items()}
        
        # ===== EVALUATION SECTION =====
        # Generate sequences for all evaluations
        noise = torch.randn(320, 128, device=device)
        with torch.no_grad():
            generated_sequences = generator(noise)

        # Get a batch of real sequences for JSD comparison
        real_batch = next(iter(dataloader)).to(device)
        if real_batch.size(0) < 320:
            # If batch size doesn't match, get multiple batches
            real_sequences = []
            for batch in dataloader:
                real_sequences.append(batch)
                if sum(b.size(0) for b in real_sequences) >= 320:
                    break
            real_batch = torch.cat(real_sequences, dim=0)[:320].to(device)

        # Convert to DNA strings using existing function
        generated_seqs = sample_and_analyze(pre_generated=generated_sequences, epoch=epoch, device=device)
        save_analysis(generated_seqs, epoch, results_dir='Results2')

        # Calculate JSD
        current_jsd = jsd(real_batch, generated_sequences)
        jsd_history.append(current_jsd)

        # Analyze sequences to get ORF count
        seq_properties = analyze_sequences(generated_seqs)
        orf_count = seq_properties['valid_orfs']

        # Convert to DNA strings for AMP evaluation (removing padding)
        generated_dna_seqs = [seq.replace('P', '') for seq in generated_seqs]

        # Evaluate AMP properties
        perfect_amp_percentage = evaluate_amp_batch(generated_dna_seqs, return_details=False)
        amp_history.append(perfect_amp_percentage)

        # Update all best models
        best_jsd_models = update_best_models(current_jsd, epoch, generator, best_jsd_models, maximize=False)
        best_orf_models = update_best_models(orf_count, epoch, generator, best_orf_models, maximize=True)
        best_amp_models = update_best_models(perfect_amp_percentage, epoch, generator, best_amp_models, maximize=True)

        # Print epoch averages
        print(f'Epoch [{epoch+1}/{num_epochs}] - Epoch Time: {epoch_time:.2f}s - Total Time: {format_time(total_time)}')
        print(f'D_total_loss: {avg_losses["total_d_loss"]:.4f}')
        print(f'Wasserstein Loss: {avg_losses["wasserstein_loss"]:.4f}')
        print(f'Gradient Penalty: {avg_losses["gradient_penalty"]:.4f}')
        print(f'Decoder Loss: {avg_losses["dec_loss"]:.4f}')
        print(f'G_total_loss: {avg_losses["total_g_loss"]:.4f}')
        print(f'G_loss: {avg_losses["g_loss"]:.4f}')
        print(f'Pixel Loss: {avg_losses["pixel_loss"]:.4f}\n')
        print(f'Latest JSD Score: {current_jsd:.4f}')
        print(f'AMP Score: {perfect_amp_percentage:.2f}%')
        print(50 * "-")

        # Step the schedulers
        d_scheduler.step()
        g_scheduler.step()

    # Save best models at the end of training
    for i, (jsd_score, epoch, model_state) in enumerate(best_jsd_models):
        save_path = os.path.join(jsd_models_dir, f'generator_jsd_{i+1}_epoch_{epoch}_score_{jsd_score:.4f}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'jsd_score': jsd_score
        }, save_path)

    for i, (orf_count, epoch, model_state) in enumerate(best_orf_models):
        save_path = os.path.join(orf_models_dir, f'generator_orf_{i+1}_epoch_{epoch}_count_{orf_count}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'orf_count': orf_count
        }, save_path)

    # Save best AMP models
    for i, (amp_score, epoch, model_state) in enumerate(best_amp_models):
        save_path = os.path.join(amp_models_dir, f'generator_amp_{i+1}_epoch_{epoch}_score_{amp_score:.2f}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'amp_score': amp_score
        }, save_path)

    return iteration_losses, total_iterations, jsd_history, amp_history