"""
Validation loop for the Face Blurring project.

- Evaluates classification accuracy and regression loss (Huber) on the validation set.
- Writes predictions and ground-truth boxes to a CSV file for later visualization.
- Applies regression loss only on human samples.
"""

import csv
import torch


def validation_loop(validation_loader, backbone_model, classification_model, regression_model, config, device):
    """
    Main validation loop.

    - Uses config for batch size, losses, and output CSV path.
    - Computes classification accuracy and average losses.
    - Saves per-sample predictions to CSV.
    """

    batch_size = config["training"]["batch_size"]
    width_target = config["model"]["width_target"]
    height_target = config["model"]["height_target"]
    output_csv_path = config["data"]["output_csv"]

    # Classification loss (always CrossEntropy or whatever is in config)
    cls_loss_name = config["training"]["classification_loss"]
    loss_fn_classification = getattr(torch.nn, cls_loss_name)()

    # Regression loss for validation metrics (fixed to Huber in your current setup)
    huber_name = config["training"]["regression_loss_huber"]
    delta_huber = config["training"]["delta"]
    loss_fn_regression = getattr(torch.nn, huber_name)(delta=delta_huber)

    backbone_model.eval()
    classification_model.eval()
    regression_model.eval()

    correctly_classified_samples = 0
    total_samples = 0
    sum_loss_classification = 0.0
    sum_loss_regression = 0.0
    sum_total_loss = 0.0
    total_human_samples = 0
    rows = []

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            images, target_coordinates, target_categories, image_ids = data

            images = images.to(device)
            target_coordinates = target_coordinates.to(device)
            target_categories = target_categories.to(device)

            backbone_output = backbone_model(images)
            predictions_classification = classification_model(backbone_output)
            predictions_regression = regression_model(backbone_output)

            _, predictions_labels = torch.max(predictions_classification, dim=1)

            loss_classification = loss_fn_classification(predictions_classification, target_categories)

            # Boolean masking: regression metrics only on humans.
            human_mask = target_categories == 0

            if human_mask.sum() > 0:
                target_coords_human = target_coordinates[human_mask]
                pred_coords_human = predictions_regression[human_mask]
                num_humans_in_batch = human_mask.sum().item()

                loss_regression = loss_fn_regression(pred_coords_human, target_coords_human)
                sum_loss_regression += loss_regression.item() * num_humans_in_batch
                total_human_samples += num_humans_in_batch
            else:
                loss_regression = torch.tensor(0.0, device=device)

            total_loss = loss_classification.item() + loss_regression.item()
            sum_total_loss += total_loss * batch_size

            correctly_classified_samples += (predictions_labels == target_categories).sum().item()
            total_samples += predictions_labels.shape[0]
            sum_loss_classification += loss_classification.item() * batch_size

            # Build CSV rows
            for i in range(images.size(0)):
                is_human = predictions_labels[i].item() == 0

                if is_human:
                    x_1_pred = int(predictions_regression[i][0].item() * width_target)
                    y_1_pred = int(predictions_regression[i][1].item() * height_target)
                    width_pred = int(predictions_regression[i][2].item() * width_target)
                    height_pred = int(predictions_regression[i][3].item() * height_target)
                else:
                    x_1_pred = 0
                    y_1_pred = 0
                    width_pred = 0
                    height_pred = 0

                rows.append({
                    "image_id": image_ids[i],
                    "x_1": float(target_coordinates[i][0].item() * width_target),
                    "y_1": float(target_coordinates[i][1].item() * height_target),
                    "width": float(target_coordinates[i][2].item() * width_target),
                    "height": float(target_coordinates[i][3].item() * height_target),
                    "category": int(target_categories[i].item()),
                    "x_1_pred": x_1_pred,
                    "y_1_pred": y_1_pred,
                    "width_pred": width_pred,
                    "height_pred": height_pred,
                    "predicted_category": int(predictions_labels[i].item()),
                })

    # Write CSV
    with open(output_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "image_id", "x_1", "y_1", "width", "height", "category",
            "x_1_pred", "y_1_pred", "width_pred", "height_pred", "predicted_category"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Metrics
    average_classification_loss = sum_loss_classification / total_samples if total_samples > 0 else 0.0
    average_regression_loss = sum_loss_regression / total_human_samples if total_human_samples > 0 else 0.0
    overall_accuracy = (correctly_classified_samples / total_samples) * 100 if total_samples > 0 else 0.0

    print(f"The overall accuracy of the model: {overall_accuracy:.2f}%")
    print(f"Average classification loss: {average_classification_loss:.6f}")
    print(f"Average regression loss: {average_regression_loss:.6f}")
    print(f"Average total loss: {sum_total_loss / total_samples:.6f}")