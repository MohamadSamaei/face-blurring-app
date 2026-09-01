def loss_regression_DIoU(target_coordinates, predicted_coordinates):
    import torch

    for i in range(target_coordinates.shape[0]):
        # Ground Truth
        x_1_norm_gt, y_1_norm_gt, width_box_norm_gt, height_box_norm_gt = target_coordinates[i, 0], target_coordinates[i, 1], target_coordinates[i, 2], target_coordinates[i, 3] # target_coordinates = x_1_norm, y_1_norm, width_box_norm, height_box_norm
        x_bottem_left_gt = x_1_norm_gt
        y_bottom_left_gt = y_1_norm_gt + height_box_norm_gt
        x_upper_right_gt = x_1_norm_gt + width_box_norm_gt
        y_upper_right_gt = y_1_norm_gt
        x_center_gt = (x_bottem_left_gt + x_upper_right_gt)/2
        y_center_gt = (y_bottom_left_gt + y_upper_right_gt)/2
        area_gt = width_box_norm_gt * height_box_norm_gt
        
        x_predicted, y_predicted, width_predicted, height_predicted = predicted_coordinates[i, 0], predicted_coordinates[i, 1], predicted_coordinates[i, 2], predicted_coordinates[i, 3]
        x_bottem_left_predicted = x_predicted
        y_bottom_left_predicted = y_predicted + height_predicted
        x_upper_right_predicted = x_predicted + width_predicted
        y_upper_right_predicted = y_predicted
        x_center_predicted = (x_bottem_left_predicted + x_upper_right_predicted)/2
        y_center_predicted = (y_bottom_left_predicted + y_upper_right_predicted)/2
        area_predicted = height_predicted * width_predicted

        # intersection
        x_bottom_left_intersection = torch.max(x_bottem_left_gt, x_bottem_left_predicted)
        y_bottom_left_intersection = torch.min(y_bottom_left_gt, y_bottom_left_predicted)
        x_upper_right_intersection = torch.min(x_upper_right_gt, x_upper_right_predicted)
        y_upper_right_intersection = torch.max(y_upper_right_gt, y_upper_right_predicted)

        if x_bottom_left_intersection < x_upper_right_intersection and y_upper_right_intersection < y_bottom_left_intersection:
            intersection = (x_upper_right_intersection - x_bottom_left_intersection) * (y_bottom_left_intersection - y_upper_right_intersection)
        else:
            intersection = 0

        union = area_gt + area_predicted - intersection        
        
        
        d2 = (x_center_gt - x_center_predicted) ** 2 + (y_center_gt - y_center_predicted) ** 2 # d is the distance between the centers
        
        x_1 = torch.min(x_bottem_left_gt, x_bottem_left_predicted)
        y_1 = torch.max(y_bottom_left_gt, y_bottom_left_predicted)
        x_2 = torch.max(x_upper_right_gt, x_upper_right_predicted)
        y_2 = torch.min(y_upper_right_gt, x_upper_right_predicted)

        c2 = (x_1 - x_2) ** 2 + (y_1 - y_2) **2

        iou = intersection/union
        diou = iou - (d2 / c2)
        loss = 1 - diou
    
    
        return loss.mean()




    