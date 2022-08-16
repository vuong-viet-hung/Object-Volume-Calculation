import torch


def train(model: torch.nn.Module,
          train_loader,
          valid_loader,
          compute_loss,
          optimizer,
          metrics: dict,
          n_epochs: int):
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
            print(f"{train_loss = }")

        # Validation mode
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, true_masks in valid_loader:
                predicted_masks = model(images)
                loss = compute_loss(predicted_masks, true_masks)
                valid_loss += loss.item()
                print(f"{valid_loss = }")

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        print(f"Epoch: {epoch}/{n_epochs})")

        # for name, metric in metrics.items():
        #     print(f"{name} = {metrics}")
