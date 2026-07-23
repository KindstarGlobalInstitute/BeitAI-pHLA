import os
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def train_MIL(model, optimizer, train_data, valid_data, device, criterion, num_epochs,
              gradient_accumulation_steps=4, eval_steps=50, out_name="optimized", test_data=None):
    rank = dist.get_rank() if dist.is_initialized() else 0

    if train_data is not None:
        max_epochs = 20
        patience = 3
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        if is_main_process():
            loss_out_path = f"./output_MIL/loss/loss_{out_name}.txt"
            model_out_path = f"./model_MIL/EL_Classification_{out_name}.ckpt"
            os.makedirs("./output_MIL", exist_ok=True)
            os.makedirs("./model_MIL", exist_ok=True)
            with open(loss_out_path, "w") as f:
                f.write("epoch\ttrain_loss\tval_loss\n")

        model.train()
        scaler = GradScaler()

        if hasattr(train_data.sampler, 'set_epoch'):
            train_data.sampler.set_epoch(0)

        for epoch in range(max_epochs):
            if hasattr(train_data.sampler, 'set_epoch'):
                train_data.sampler.set_epoch(epoch)

            if is_main_process():
                print(f'Epoch [{epoch + 1}/{max_epochs}]')
                pbar = tqdm(total=len(train_data))

            optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_steps = 0

            for i, batch in enumerate(train_data):
                input_data = batch['input_data'].to(device, non_blocking=True)
                input_ids = batch['input_ids']
                target = batch['labels'].to(device, non_blocking=True)

                with autocast():
                    output = model(input_data, input_ids)
                    loss = criterion(output, target)
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    step_loss = loss.item() * gradient_accumulation_steps
                    epoch_loss += step_loss
                    epoch_steps += 1

                    if is_main_process():
                        pbar.set_postfix({'loss': f'{step_loss:.4f}'})

                if is_main_process():
                    pbar.update(1)

            if is_main_process():
                pbar.close()

            avg_loss = epoch_loss / max(epoch_steps, 1)
            if is_main_process():
                print(f"  Train loss: {avg_loss:.4f}")

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in valid_data:
                    v_input = batch['input_data'].to(device, non_blocking=True)
                    v_ids = batch['input_ids']
                    v_target = batch['labels'].to(device, non_blocking=True)
                    with autocast():
                        v_output = model(v_input, v_ids)
                        v_loss = criterion(v_output, v_target)
                    val_loss += v_loss.item()
                    val_steps += 1

            val_loss /= max(val_steps, 1)

            if dist.is_initialized():
                val_tensor = torch.tensor(val_loss, device=device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                val_loss = val_tensor.item() / dist.get_world_size()

            if is_main_process():
                print(f"  Val loss: {val_loss:.4f}")

                with open(loss_out_path, "a") as f:
                    f.write(f"{epoch+1}\t{avg_loss:.4f}\t{val_loss:.4f}\n")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = (
                        model.module.state_dict() if hasattr(model, 'module')
                        else model.state_dict()
                    )
                else:
                    patience_counter += 1
                    print(f"  No improvement for {patience_counter}/{patience} epochs")

            # Broadcast early stop signal to all processes
            early_stop = torch.tensor(0, device=device)
            if is_main_process() and patience_counter >= patience:
                early_stop = torch.tensor(1, device=device)
            if dist.is_initialized():
                dist.broadcast(early_stop, src=0)

            model.train()

            if early_stop.item() == 1:
                if is_main_process():
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save best model
        if is_main_process():
            save_state = (
                best_model_state
                if best_model_state is not None
                else (model.module.state_dict() if hasattr(model, 'module')
                      else model.state_dict())
            )
            torch.save(save_state, model_out_path)
            print(f"Best model saved to {model_out_path}")

    if test_data is not None and is_main_process():
        model.eval()
        with torch.no_grad():
            test_output, test_target = _evaluate(model, test_data, device)
        try:
            test_auprc = average_precision_score(test_target, test_output)
        except ValueError:
            test_auprc = 0.0
        try:
            test_preds = (test_output >= 0.5).astype(np.float32)
            test_f1 = f1_score(test_target, test_preds)
            test_pre = precision_score(test_target, test_preds, zero_division=0)
            test_recall = recall_score(test_target, test_preds, zero_division=0)
        except Exception:
            test_f1 = test_pre = test_recall = 0.0

        test_out_path = f"./output_MIL/Ablation_result/test_{out_name}.txt"
        print(f"\n===== Test Set Evaluation =====")
        print(f"AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}, Precision: {test_pre:.4f}, Recall: {test_recall:.4f}")
        if not os.path.exists(test_out_path):
            with open(test_out_path, "w") as f:
                f.write("auprc\tf1\tpre\trecall\n")
                f.write(f"{test_auprc:.4f}\t{test_f1:.4f}\t{test_pre:.4f}\t{test_recall:.4f}\n")
        else:
            with open(test_out_path, "a") as f:
                f.write(f"{test_auprc:.4f}\t{test_f1:.4f}\t{test_pre:.4f}\t{test_recall:.4f}\n")


def _evaluate(model, valid_data, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(valid_data):
            input_data = batch['input_data'].to(device, non_blocking=True)
            input_ids = batch['input_ids']
            target = batch['labels'].to(device, non_blocking=True)

            with autocast():
                output = model(input_data, input_ids)

            probs = torch.sigmoid(output).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(target.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return all_probs, all_labels


def valid_MIL(model, valid_data, device, epoch, out_name):
    model.eval()
    pred_all = []
    label_all = []
    with torch.no_grad():
        for batch in tqdm(valid_data):
            input_data = batch['input_data'].to(device, non_blocking=True)
            input_ids = batch['input_ids']
            target = batch['labels'].to(device, non_blocking=True)

            with autocast():
                output = model(input_data, input_ids)
                probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()

            pred_all.append(preds.cpu().numpy())
            label_all.append(target.cpu().numpy())

    pred_all = np.concatenate(pred_all)
    label_all = np.concatenate(label_all)

    f1 = f1_score(label_all, pred_all, zero_division=0)
    pre = precision_score(label_all, pred_all, zero_division=0)
    recall = recall_score(label_all, pred_all, zero_division=0)

    with open(out_name, "a") as file:
        file.write(f"{epoch}\t{f1}\t{pre}\t{recall}\n")
    print(f'Valid [{epoch}]\t{f1}\t{pre}\t{recall}')
