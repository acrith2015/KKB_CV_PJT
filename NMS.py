import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

def NMS(lists, thre):
    # if the list of bboxes and scores is empty, return;
    if len(lists) == 0:
        return {}

    # lists is a list. lists[0:4]: x1, x2, y1, y2; lists[4]: score
    x1_arr, x2_arr, y1_arr, y2_arr, score_arr = [lists[:, i] for i in range(5)]

    # calculate the each bbox area
    area_arr = (x2_arr - x1_arr) * (y2_arr - y1_arr)

    # sort the score in descending order
    sorted_idx = score_arr.argsort()[::-1]

    # the array of max value bboxes
    max_bb_arr = []

    lists = np.array(lists)

    while len(sorted_idx) > 0:
        # add the index of new class object
        max_bb_arr.append(sorted_idx[0])

        # calculate the all other bbox's IOU
        min_x_arr = np.maximum(x1_arr[sorted_idx[0]], x1_arr[sorted_idx[1:]])
        max_x_arr = np.minimum(x2_arr[sorted_idx[0]], x2_arr[sorted_idx[1:]])
        min_y_arr = np.maximum(y1_arr[sorted_idx[0]], y1_arr[sorted_idx[1:]])
        max_y_arr = np.minimum(y2_arr[sorted_idx[0]], y2_arr[sorted_idx[1:]])

        # 将没有交集的bbox 置零，保留下来
        width = np.maximum(0, max_x_arr - min_x_arr + 1)
        height = np.maximum(0, max_y_arr - min_y_arr + 1)

        # calculate the inner area
        inner_area = width * height

        iou_arr = inner_area / (area_arr[sorted_idx[0]] + area_arr[sorted_idx[1:]] - inner_area)

        # delete the bboxes of the same class
        higher_iou_bbox_idx_arr = np.argwhere(iou_arr > thre) + 1
        higher_iou_bbox_idx_arr = np.append(higher_iou_bbox_idx_arr, np.array([0]))
        sorted_idx = np.delete(sorted_idx, higher_iou_bbox_idx_arr)

    return lists[max_bb_arr]

if __name__== '__main__':
    boxes = np.array([[1, 20, 1, 25], [1, 19, 2, 26], [4, 21, 6, 24], [30, 40, 22, 32], [28, 38, 24, 32]], dtype=np.float32)
    scores = np.array([[0.8], [0.67], [0.72], [0.35], [0.95]], dtype=np.float32)
    thre = 0.5
    lists = np.hstack((boxes, scores))

    # call the NMS function
    result_boxes_scores = NMS(lists, thre)
    print(result_boxes_scores)

    # visualization
    fig, ax = plt.subplots()
    x1_arr, x2_arr, y1_arr, y2_arr, score_arr = [lists[:, i] for i in range(5)]
    w_arr = x2_arr - x1_arr + 1
    h_arr = y2_arr - y1_arr + 1

    for i in range(len(x1_arr)):
        rect = mpathes.Rectangle((x1_arr[i], y1_arr[i]), w_arr[i], h_arr[i], edgecolor='b', fill=False, linewidth=2)
        ax.add_patch(rect)

    x1_arr, x2_arr, y1_arr, y2_arr, score_arr = [result_boxes_scores[:, i] for i in range(5)]
    w_arr = x2_arr - x1_arr + 1
    h_arr = y2_arr - y1_arr + 1

    for i in range(len(x1_arr)):
        rect = mpathes.Rectangle((x1_arr[i], y1_arr[i]), w_arr[i], h_arr[i], edgecolor='g', fill=False, linewidth=2)
        ax.add_patch(rect)

    ax.set_title('NMS(The Green Rects Are The Max Value Bboxes)', fontsize=12, color='r')
    plt.axis('equal')
    plt.show()



