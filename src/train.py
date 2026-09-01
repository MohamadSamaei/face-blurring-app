"""
Training loop for the Face Blurring project.

Supports two regression losses:
- Huber (with dynamic lambda weighting)
- DIoU (with fixed weight from config)

The loop:
- Trains backbone + classification head + regression head jointly.
- Applies regression loss only on human samples with valid bounding boxes.
- Tracks per-epoch accuracy and losses.
"""

import time
from tqdm import tqdm
import torch

from src.loss_function_DIoU import loss_regression_DIoU


def build_optimizer_and_losses(config, models):
    """
    Construct optimizer and loss functions from config.

    Returns:
        optimizer_general
        loss_fn_classification
        loss_fn_regression (or None for DIoU, handled separately)
        loss_choice ("Huber" or "DIoU")
        weight_diou (only used if loss_choice == "DIoU")
        delta_huber (only used if loss_choice == "Huber")
    """
    backbone, classification_head, regression_head = models

    all_parameters = (
        list(backbone.parameters())
        + list(classification_head.parameters())
        + list(regression_head.parameters())
    )

    lr = config["training"]["lr_general"]
    momentum = config["training"]["momentum_general"]
    opt_name = config["training"]["optimizer_general"]

    optimizer_class = getattr(torch.optim, opt_name)
    optimizer_general = optimizer_class(all_parameters, lr=lr, momentum=momentum)

    cls_loss_name = config["training"]["classification_loss"]
    loss_fn_classification = getattr(torch.nn, cls_loss_name)()

    # Ask user which regression loss to use.
    loss_choice = input("What regression loss do you want among Huber/DIoU? ").strip()
    if loss_choice not in ("Huber", "DIoU"):
        raise ValueError(f"Unsupported regression loss choice: {loss_choice}")

    weight_diou = None
    delta_huber = None
    loss_fn_regression = None

    if loss_choice == "Huber":
        huber_name = config["training"]["regression_loss_huber"]
        delta_huber = config["training"]["delta"]
        loss_fn_regression = getattr(torch.nn, huber_name)(delta=delta_huber)
    elif loss_choice == "DIoU":
        weight_diou = config["training"]["weight_loss_regression_DIOU"]

    return optimizer_general, loss_fn_classification, loss_fn_regression, loss_choice, weight_diou, delta_huber


def training_loop(train_loader, backbone_model, classification_model, regression_model, config, device):
    """
    Main training loop.

    - Uses config for epochs, batch size, optimizer, and loss settings.
    - Applies regression loss only on human samples with valid bounding boxes.
    - Prints per-epoch accuracy and losses.
    """

    models = (backbone_model, classification_model, regression_model)
    optimizer_general, loss_fn_classification, loss_fn_regression, loss_choice, weight_diou, delta_huber = (
        build_optimizer_and_losses(config, models)
    )

    backbone_model.train()
    classification_model.train()
    regression_model.train()

    no_epoch = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]

    accuracy_among_epochs = []
    loss_regression_among_epochs = []
    loss_classification_among_epochs = []
    total_loss_among_epochs = []

    for epoch in range(no_epoch):
        t_initial_epoch = time.perf_counter()
        print(f"Epoch being run: {epoch + 1} | Out of {no_epoch} total epochs")

        sum_total_loss = 0.0
        sum_loss_classification = 0.0
        sum_loss_regression = 0.0
        num_of_correct_predictions = 0
        total_samples = 0
        total_human_samples = 0

        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training phase")):
            images, target_coordinates, target_categories, image_id = data

            images = images.to(device, non_blocking=True)
            target_coordinates = target_coordinates.to(device, non_blocking=True)
            target_categories = target_categories.to(device, non_blocking=True)

            optimizer_general.zero_grad(set_to_none=True)

            output_backbone = backbone_model(images)
            predictions_categories = classification_model(output_backbone)
            predictions_coordinates = regression_model(output_backbone)

            loss_classification = loss_fn_classification(predictions_categories, target_categories)

            _, predicted_labels = torch.max(predictions_categories, dim=1)
            num_of_correct_predictions += (predicted_labels == target_categories).sum().item()
            total_samples += target_categories.size(0)

            # Boolean masking: regression only on humans with valid boxes.
            valid_bb_mask = target_coordinates[:, 2] > 0
            human_mask = (target_categories == 0) & valid_bb_mask

            if human_mask.sum() > 0:
                masked_pred_coords = predictions_coordinates[human_mask]
                masked_target_coords = target_coordinates[human_mask]
                num_humans_in_batch = human_mask.sum().item()

                if loss_choice == "DIoU":
                    loss_regression = loss_regression_DIoU(
                        target_coordinates=masked_target_coords,
                        predicted_coordinates=masked_pred_coords,
                    )
                elif loss_choice == "Huber":
                    loss_regression = loss_fn_regression(masked_pred_coords, masked_target_coords)

                sum_loss_regression += loss_regression.item() * num_humans_in_batch
                total_human_samples += num_humans_in_batch
            else:
                loss_regression = torch.tensor(0.0, device=device, requires_grad=True)

            # Loss weighting
            if loss_choice == "Huber":
                dynamic_lambda = loss_classification.item() / (loss_regression.item() + 1e-6)
                dynamic_lambda = min(max(dynamic_lambda, 5), 100)
                total_loss = loss_classification + dynamic_lambda * loss_regression
            elif loss_choice == "DIoU":
                total_loss = loss_classification + weight_diou * loss_regression

            total_loss.backward(retain_graph=False)
            optimizer_general.step()

            sum_total_loss += total_loss.item() * batch_size
            sum_loss_classification += loss_classification.item() * batch_size

        t_end_epoch = time.perf_counter()
        print(f"Epoch {epoch + 1} took {t_end_epoch - t_initial_epoch:.2f} seconds")

        average_loss_regression = sum_loss_regression / total_human_samples if total_human_samples > 0 else 0.0

        loss_regression_among_epochs.append(average_loss_regression)
        loss_classification_among_epochs.append(sum_loss_classification / total_samples)
        total_loss_among_epochs.append(sum_total_loss / total_samples)
        accuracy_among_epochs.append((num_of_correct_predictions / total_samples) * 100)

        print(f"In the epoch {epoch + 1}, the accuracy is equal to: {(num_of_correct_predictions/total_samples) * 100}")
        print(f"In the epoch {epoch + 1}, loss of classification wrt to total loss is: {sum_loss_classification/total_samples}")
        print(f"In the epoch {epoch + 1}, loss of regression wrt to total loss is: {average_loss_regression}")
        print(f"In the epoch {epoch + 1}, average total loss is: {sum_total_loss/total_samples}")
        print("*" * 30)

    print(f"loss_classification_among_epochs: {loss_classification_among_epochs}")
    print(f"loss_regression_among_epochs: {loss_regression_among_epochs}")
    print(f"total_loss_among_epochs: {total_loss_among_epochs}")
    print(f"accuracy_among_epochs: {accuracy_among_epochs}")