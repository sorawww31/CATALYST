"""Repeatable code parts concerning optimization and training schedules."""

import copy
import datetime

import torch
import torch.nn.functional as F

import wandb

from ..consts import BENCHMARK, DEBUG_TRAINING, NON_BLOCKING
from ..utils import cw_loss
from .utils import pgd_step, print_and_save_stats

torch.backends.cudnn.benchmark = BENCHMARK


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    if defs.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=defs.lr,
            momentum=0.9,
            weight_decay=defs.weight_decay,
            nesterov=True,
        )
    elif defs.optimizer == "SGD-basic":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=defs.lr,
            momentum=0.0,
            weight_decay=defs.weight_decay,
            nesterov=False,
        )
    elif defs.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay
        )

    if defs.scheduler == "cyclic":
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(
            f"Optimization will run over {effective_batches} effective batches in a 1-cycle policy."
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=defs.lr / 100,
            max_lr=defs.lr,
            step_size_up=effective_batches // 2,
            cycle_momentum=True if defs.optimizer in ["SGD"] else False,
        )
    elif defs.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[defs.epochs // 2.667, defs.epochs // 1.6, defs.epochs // 1.142],
            gamma=0.1,
        )
    elif defs.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10_000, 15_000, 25_000], gamma=1
        )

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler


def renewal_wolfecondition_stepsize(
    kettle, args, model, criterion, alpha_init, targetset, setup
):
    """
    Wolfe 条件を満たすステップサイズ alpha を探索する。
    不要な再計算を減らすために、元の勾配を使い回し、deepcopy を最小限にした。
    """
    if not targetset:
        return alpha_init  # ターゲットがないならスキップ

    c2, c1 = args.wolfe
    device = setup["device"]

    # ----- 1) まずは元の損失と勾配を一度だけ計算してキャッシュ -----
    # モデルは学習モードではなく eval モードで扱う（勾配計算時は no_grad しないので注意）
    model.eval()

    target_images = torch.stack([data[0] for data in targetset]).to(device)
    intended_class = kettle.poison_setup["intended_class"]
    intended_labels = torch.tensor(intended_class, device=device, dtype=torch.long)

    # 元の損失 fx と勾配 nabla_fx
    fx = criterion(model(target_images), intended_labels)
    nabla_fx = torch.autograd.grad(fx, model.parameters(), create_graph=True)

    # ----- 2) Wolfe条件探索用のパラメータ -----
    alpha = alpha_init
    max_iters = 40  # 最大反復回数
    omega = 0.75  # 学習率縮小係数

    # 勾配を一時保存しておき、あとで計算を再利用しやすくする
    # パラメータを一列に flatten する
    nabla_fx_flat = torch.cat([g.view(-1) for g in nabla_fx]).detach()

    # dot(grad, grad) などは再利用すると若干効率が良い
    nabla_fx_dot = nabla_fx_flat.dot(nabla_fx_flat)

    # ----- 3) テンポラリのパラメータ更新用バッファを用意 -----
    # モデル本体をdeepcopyするコストを減らすため、パラメータだけバックアップして
    # alpha 更新後に元に戻すやり方
    original_params = [p.detach().clone() for p in model.parameters()]

    wolfe_satisfied = False

    for _ in range(max_iters):
        # ----- (A) パラメータを alpha だけ進める -----
        with torch.no_grad():
            for p, g in zip(model.parameters(), nabla_fx):
                p -= alpha * g

        # ----- (B) 新たな損失を計算 -----
        fx_new = criterion(model(target_images), intended_labels)

        # ----- (C) 十分減少条件 (Armijo 条件) の確認 -----
        # fx_new <= fx - c1 * alpha * \nabla f(x)^T d
        # d = -\nabla f(x) なので、\nabla f(x)^T d = - ||\nabla f(x)||^2
        # ここでは nabla_fx_dot = ||\nabla f(x)||^2
        lhs = fx_new.item()
        rhs = fx.item() - c1 * alpha * nabla_fx_dot
        sufficient_decrease = lhs <= rhs

        if sufficient_decrease:
            # ----- (D) 曲率条件の確認 (Wolfe第二条件) -----
            # \nabla f(x + alpha d)^T d >= c2 * \nabla f(x)^T d
            # d = -\nabla f(x)
            nabla_fx_new = torch.autograd.grad(
                fx_new, model.parameters(), create_graph=True
            )
            nabla_fx_new_flat = torch.cat([g.view(-1) for g in nabla_fx_new])
            nabla_fx_new_dot = nabla_fx_new_flat.dot(
                nabla_fx_flat
            )  # \nabla f(x+alpha d) dot \nabla f(x)
            # d = - \nabla f(x) なので \nabla f(x+alpha d)^T d = - nabla_fx_new_dot
            # \nabla f(x)^T d = - nabla_fx_dot
            # Wolfe 第二条件: -nabla_fx_new_dot <= - c2 * nabla_fx_dot → nabla_fx_new_dot >= c2 * nabla_fx_dot
            if nabla_fx_new_dot >= c2 * nabla_fx_dot:
                wolfe_satisfied = True
                break

        # ----- (E) 条件を満たさなかったので alpha を縮小して再トライ -----
        # モデルパラメータを元に戻してから alpha を更新
        with torch.no_grad():
            for p, backup_p in zip(model.parameters(), original_params):
                p.copy_(backup_p)
        alpha *= omega

    if not wolfe_satisfied:
        print(
            "Wolfe条件を満たす学習率が見つかりませんでした。alpha={:.6f}を使用します。".format(
                alpha
            )
        )

    # 計算のあと、パラメータを元に戻しておく(最終的に実際に alpha を使うときに更新する)
    with torch.no_grad():
        for p, backup_p in zip(model.parameters(), original_params):
            p.copy_(backup_p)

    return alpha


def check_cosine_similarity(kettle, model, criterion, inputs, labels, step_size):
    device = kettle.setup["device"]
    model.eval()

    target_images = torch.stack([data[0] for data in kettle.targetset]).to(device)
    intended_class = kettle.poison_setup["intended_class"]
    intended_labels = torch.tensor(intended_class, device=device, dtype=torch.long)

    outputs_normal = model(inputs)
    fx = criterion(outputs_normal, labels)

    # (B) grads_normal を取得
    grads_normal = torch.autograd.grad(fx, model.parameters(), retain_graph=True)
    grads_normal_flat = torch.cat([g.view(-1) for g in grads_normal])

    # (C) ターゲットバッチの forward
    outputs_target = model(target_images)
    fx_target = criterion(outputs_target, intended_labels)

    # (D) grads_target を取得
    grads_target = torch.autograd.grad(fx_target, model.parameters())
    grads_target_flat = torch.cat([g.view(-1) for g in grads_target])

    # (E) Cosine Similarity を一回で計算
    cos_sim = F.cosine_similarity(grads_normal_flat, grads_target_flat, dim=0)
    if kettle.args.wandb:
        wandb.log(
            {
                "train_loss": fx.item(),
                "target_loss": fx_target.item(),
                "cosine_similarity": cos_sim.item(),
                "step-size": step_size,
            }
        )
    return cos_sim.item()


def run_step(
    kettle,
    poison_delta,
    loss_fn,
    epoch,
    stats,
    model,
    defs,
    criterion,
    optimizer,
    scheduler,
    ablation=True,
):

    epoch_loss, total_preds, correct_preds, ave_cos = 0, 0, 0, 0  # <- 追加
    cos_sim = None

    if DEBUG_TRAINING:
        data_timer_start = torch.cuda.Event(enable_timing=True)
        data_timer_end = torch.cuda.Event(enable_timing=True)
        forward_timer_start = torch.cuda.Event(enable_timing=True)
        forward_timer_end = torch.cuda.Event(enable_timing=True)
        backward_timer_start = torch.cuda.Event(enable_timing=True)
        backward_timer_end = torch.cuda.Event(enable_timing=True)

        stats["data_time"] = 0
        stats["forward_time"] = 0
        stats["backward_time"] = 0

        data_timer_start.record()

    if kettle.args.ablation < 1.0:
        # run ablation on a subset of the training set
        loader = kettle.partialloader
    else:
        loader = kettle.trainloader
    current_lr = optimizer.param_groups[0]["lr"]
    step = 0
    for batch, (inputs, labels, ids) in enumerate(loader):
        # Prep Mini-Batch

        model.train()
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(
            dtype=torch.long, device=kettle.setup["device"], non_blocking=NON_BLOCKING
        )

        if DEBUG_TRAINING:
            data_timer_end.record()
            forward_timer_start.record()

        # Add adversarial pattern
        if poison_delta is not None:
            poison_slices, batch_positions = [], []
            for batch_id, image_id in enumerate(ids.tolist()):
                lookup = kettle.poison_lookup.get(image_id)
                if lookup is not None:
                    poison_slices.append(lookup)
                    batch_positions.append(batch_id)
            # Python 3.8:
            # twins = [(b, l) for b, i in enumerate(ids.tolist()) if l:= kettle.poison_lookup.get(i)]
            # poison_slices, batch_positions = zip(*twins)

            if batch_positions:
                inputs[batch_positions] += poison_delta[poison_slices].to(
                    **kettle.setup
                )

        # Add data ation
        if (
            defs.augmentations and kettle.args.data_aug != "none"
        ):  # defs.augmentations is actually a string, but it is False if --noaugment
            inputs = kettle.augment(inputs)

        # Does adversarial training help against poisoning?
        for _ in range(defs.adversarial_steps):
            inputs = pgd_step(
                inputs,
                labels,
                model,
                loss_fn,
                kettle.dm,
                kettle.ds,
                eps=kettle.args.eps,
                tau=kettle.args.tau,
            )

        if defs.mixing_method["type"] != "":
            inputs, extra_labels, mixing_lambda = kettle.mixer.forward(
                inputs, labels, epoch
            )
            outputs = model(inputs)
            loss, _ = kettle.mixer.corrected_loss(
                model, outputs, extra_labels, mixing_lambda, loss_fn
            )
        else:
            # Get loss
            outputs = model(inputs)
            # Use corrected loss if data mixing is applied
            loss = loss_fn(model, outputs, labels)

        if DEBUG_TRAINING:
            forward_timer_end.record()
            backward_timer_start.record()

        loss.backward()

        # Enforce batch-wise privacy if necessary
        # This is a defense discussed in Hong et al., 2020
        # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
        # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
        # of noise to the gradient signal
        with torch.no_grad():
            if defs.privacy["clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), defs.privacy["clip"])
            if defs.privacy["noise"] is not None:
                # generator = torch.distributions.laplace.Laplace(torch.as_tensor(0.0).to(**kettle.setup),
                #                                                 kettle.defs.privacy['noise'])
                for param in model.parameters():
                    # param.grad += generator.sample(param.shape)
                    noise_sample = (
                        torch.randn_like(param)
                        * defs.privacy["clip"]
                        * defs.privacy["noise"]
                    )
                    param.grad += noise_sample

        if (
            (epoch > kettle.args.linesearch_epoch)
            and kettle.args.wolfe
            # and poison_delta is not None
        ):
            alpha = renewal_wolfecondition_stepsize(
                kettle,
                kettle.args,
                defs,
                model,
                criterion,
                current_lr * 2,
                optimizer.param_groups[0]["lr"],
                kettle.targetset,
                kettle.setup,
            )
            optimizer.param_groups[0]["lr"] = alpha
            current_lr = alpha

        optimizer.step()

        predictions = torch.argmax(outputs.data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss += loss.item()

        if DEBUG_TRAINING:
            backward_timer_end.record()
            torch.cuda.synchronize()
            stats["data_time"] += data_timer_start.elapsed_time(data_timer_end)
            stats["forward_time"] += forward_timer_start.elapsed_time(forward_timer_end)
            stats["backward_time"] += backward_timer_start.elapsed_time(
                backward_timer_end
            )

            data_timer_start.record()

        if defs.scheduler == "cyclic":
            scheduler.step()
        if kettle.args.wandb:
            ave_cos += check_cosine_similarity(
                kettle, model, criterion, inputs, labels, current_lr
            )
        if kettle.args.dryrun:
            break

    if defs.scheduler == "linear":
        scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss = run_validation(
            model, criterion, kettle.validloader, kettle.setup, kettle.args.dryrun
        )
        target_acc, target_loss, target_clean_acc, target_clean_loss = check_targets(
            model,
            criterion,
            kettle.targetset,
            kettle.poison_setup["intended_class"],
            kettle.poison_setup["target_class"],
            kettle.setup,
        )
    else:
        valid_acc, valid_loss = None, None
        target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]["lr"]

    print_and_save_stats(
        kettle,
        epoch,
        stats,
        current_lr,
        epoch_loss / (batch + 1),
        correct_preds / total_preds,
        valid_acc,
        valid_loss,
        target_acc,
        target_loss,
        target_clean_acc,
        target_clean_loss,
        ave_cos / (batch + 1),
    )

    if DEBUG_TRAINING:
        print(
            f"Data processing: {datetime.timedelta(milliseconds=stats['data_time'])}, "
            f"Forward pass: {datetime.timedelta(milliseconds=stats['forward_time'])}, "
            f"Backward Pass and Gradient Step: {datetime.timedelta(milliseconds=stats['backward_time'])}"
        )
        stats["data_time"] = 0
        stats["forward_time"] = 0
        stats["backward_time"] = 0


def run_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(
                device=setup["device"], dtype=torch.long, non_blocking=NON_BLOCKING
            )
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if dryrun:
                break

    accuracy = correct / total
    loss_avg = loss / (i + 1)
    return accuracy, loss_avg


def check_targets(model, criterion, targetset, intended_class, original_class, setup):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    if len(targetset) > 0:

        target_images = torch.stack([data[0] for data in targetset]).to(**setup)
        intended_labels = torch.tensor(intended_class).to(
            device=setup["device"], dtype=torch.long
        )
        original_labels = torch.stack(
            [
                torch.as_tensor(data[1], device=setup["device"], dtype=torch.long)
                for data in targetset
            ]
        )
        with torch.no_grad():
            outputs = model(target_images)
            predictions = torch.argmax(outputs, dim=1)

            loss_intended = criterion(outputs, intended_labels)
            accuracy_intended = (
                predictions == intended_labels
            ).sum().float() / predictions.size(0)
            loss_clean = criterion(outputs, original_labels)
            predictions_clean = torch.argmax(outputs, dim=1)
            accuracy_clean = (
                predictions == original_labels
            ).sum().float() / predictions.size(0)

            # print(f'Raw softmax output is {torch.softmax(outputs, dim=1)}, intended: {intended_class}')

        return (
            accuracy_intended.item(),
            loss_intended.item(),
            accuracy_clean.item(),
            loss_clean.item(),
        )
    else:
        return 0, 0, 0, 0
