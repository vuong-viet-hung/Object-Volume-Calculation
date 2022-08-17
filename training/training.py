from pathlib import Path

import torch


def train(model: torch.nn.Module,
          train_loader,
          valid_loader,
          compute_loss,
          optimizer,
          metrics: dict,
          n_epochs: int,
          patience: int,
          saved_model_path: Path,
          ):

    valid_losses = []

    for epoch in range(1, n_epochs + 1):
        # Training mode
        model.train()
        train_loss = 0.0
        for images, true_masks in train_loader:
            optimizer.zero_grad()
            predicted_masks = model(images)
            loss = compute_loss(predicted_masks, true_masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            for name, metric in metrics.items():
                print(f"{name}: {metric(predicted_masks, true_masks)}")
            print("Accuracy:",
                  torch.sum(torch.round(predicted_masks) == torch.round(true_masks)) / torch.numel(predicted_masks)
                  )
            torch.save(model, str(saved_model_path))

        # Validation mode
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, true_masks in valid_loader:
                predicted_masks = model(images)
                loss = compute_loss(predicted_masks, true_masks)
                valid_loss += loss.item()
                for name, metric in metrics.items():
                    print(f"{name}: {metric(predicted_masks, true_masks)}")
                print(
                    "Accuracy:",
                    torch.sum(torch.round(predicted_masks) == torch.round(true_masks)) / torch.numel(predicted_masks),
                )

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        print(f"Epoch: {epoch}/{n_epochs})")
        print(f"{train_loss = }")
        print(f"{valid_loss = }")

        # for name, metric in metrics.items():
        #     print(f"{name} = {metrics}")

        valid_losses.append(valid_loss)
        min_valid_losses_idx = valid_losses.index(min(valid_losses))
        if len(valid_losses) - min_valid_losses_idx > patience:
            break
