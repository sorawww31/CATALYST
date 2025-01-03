"""Utilites related to training models."""

import torch

import wandb


def print_and_save_stats(
    kettle,
    epoch,
    stats,
    current_lr,
    train_loss,
    train_acc,
    valid_acc,
    valid_loss,
    target_acc,
    target_loss,
    target_clean_acc,
    target_clean_loss,
    cos_sim=None,
):
    """Print info into console and into the stats object."""
    stats["train_losses"].append(train_loss)
    stats["train_accs"].append(train_acc)
    stats["learning_rates"].append(current_lr)

    if valid_acc is not None:
        stats["valid_accs"].append(valid_acc)
        stats["valid_losses"].append(valid_loss)

        print(
            f"Epoch: {epoch:<3}| lr: {current_lr:.8f} | "
            f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | '
            f'Validation   loss is {stats["valid_losses"][-1]:7.4f}, valid acc: {stats["valid_accs"][-1]:7.2%} | '
        )

        stats["target_accs"].append(target_acc)
        stats["target_losses"].append(target_loss)
        stats["target_accs_clean"].append(target_clean_acc)
        stats["target_losses_clean"].append(target_clean_loss)
        print(
            f"Epoch: {epoch:<3}| lr: {current_lr:.8f} | "
            f"Target adv. loss is {target_loss:7.4f}, fool  acc: {target_acc:7.2%} | "
            f"Target orig. loss is {target_clean_loss:7.4f}, orig. acc: {target_clean_acc:7.2%} | "
        )

    else:
        if "valid_accs" in stats:
            # Repeat previous answers if validation is not recomputed
            stats["valid_accs"].append(stats["valid_accs"][-1])
            stats["valid_losses"].append(stats["valid_losses"][-1])
            stats["target_accs"].append(stats["target_accs"][-1])
            stats["target_losses"].append(stats["target_losses"][-1])
            stats["target_accs_clean"].append(stats["target_accs_clean"][-1])
            stats["target_losses_clean"].append(stats["target_losses_clean"][-1])

        print(
            f"Epoch: {epoch:<3}| lr: {current_lr:.8f} | "
            f'Training    loss is {stats["train_losses"][-1]:7.4f}, train acc: {stats["train_accs"][-1]:7.2%} | '
        )
    if cos_sim is not None:
        stats["cos_sim"].append(cos_sim)

    if kettle.args.wandb:
        wandb.log(
            {
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "target_acc": target_acc,
                "average_cosine_similarity": cos_sim,
            }
        )


def pgd_step(inputs, labels, model, loss_fn, dm, ds, eps=16, tau=0.01):
    """Perform a single projected signed gradient descent step, maximizing the loss on the given labels."""
    inputs.requires_grad = True
    tau = eps / 255 / ds * tau

    loss = loss_fn(model(inputs), labels)
    grads = torch.grad.autograd(
        loss, inputs, retain_graph=True, create_graph=True, only_inputs=True
    )
    inputs.requires_grad = False
    with torch.no_grad():
        # Gradient Step
        outputs = inputs + tau * grads

        # Projection Step
        outputs = torch.max(torch.min(outputs, eps / ds / 255), -eps / ds / 255)
        outputs = torch.max(
            torch.min(outputs, (1 + dm) / ds - inputs), -dm / ds - inputs
        )
    return outputs


def initialize_metrics():
    """Initialize metrics for recording during training."""
    return {"learning_rates": [], "training_losses": [], "adversarial_losses": []}


def record_metrics(metrics, optimizer, loss, adversarial_loss):
    """Record current learning rate, training loss, and adversarial loss."""
    current_lr = optimizer.param_groups[0]["lr"]
    metrics["learning_rates"].append(current_lr)
    metrics["training_losses"].append(loss.item())
    metrics["adversarial_losses"].append(
        adversarial_loss.item() if adversarial_loss is not None else 0
    )
