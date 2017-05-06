#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import tools._init_paths
import cPickle
from datasets.factory import get_imdb


class EvalConfig(object):
    iou_thresh = 0.5
    min_width = 20
    min_height = 20

    # a list of type IDs
    eval_type = None

    # (x1, y1, x2, y2) -> (x, y, w, h) ?
    transform_gt = False
    transform_det = False


def read_bboxes(fin, cls, transform=False):
    im_name = fin.readline().rstrip()
    bbs = []
    if im_name:
        num_bbox = int(fin.readline().rstrip())
        if num_bbox > 0:
            for _ in xrange(num_bbox):
                line = fin.readline().rstrip().split()
                line = map(float, line)
                bbs.append(line)
    else:
        num_bbox = -1



    key = 0
    bboxes = []

    for i, b in enumerate(bbs):
        part = []
        clsId = 0
        for j, num in enumerate(b):
            if j % 5 != 0:
                part.append(num)
                if len(part) == 4:
                    clsId += 1
                    part.append(clsId)
                    if key>=0 and min(part)>=0 and clsId==cls:
                        bboxes.append(part)
                    part = []
            else:
                key = num

    num_bbox = len(bboxes)
    bboxes = np.array(bboxes)


    if num_bbox > 0:
        if transform:
            bboxes[:, 2:4] += (bboxes[:, :2] - 1)


    return {'im_name': im_name,
            'num_bbox': num_bbox,
            'bboxes': bboxes}


def compute_ap(recall, precision):
    """
    Compute VOC AP given precision and recall.
    """
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_points(x, y, target_x):
    '''x should be sorted in ascending order'''
    x1 = np.asarray(x)
    y1 = np.asarray(y)
    y0 = []
    x2 = []
    for x0 in target_x:
        idx = np.where(x1 == x0)[0]
        if idx.shape[0] > 0:
            y0.append(y1[idx].mean())
            x2.append(x0)
        else:
            idx = np.where(x1 > x0)[0]
            if idx.shape[0] > 0:
                w1 = x1[idx[0]] - x0
                w2 = x0 - x1[idx[0] - 1]
                w = w1 + w2
                y0.append((y1[idx[0]] * w2 + y1[idx[0] - 1] * w1) / w)
                x2.append(x0)
            else:
                y0.append(y1[-1])
                x2.append(x1[-1])
    return x2, y0


def eval_roc_pr(config, gt_file, det_file, cls):

    cache_file = os.path.join(det_file)

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            detAll = np.array(cPickle.load(fid))


    default_iou_thresh = config.iou_thresh
    min_width = config.min_width
    min_height = config.min_height
    eval_type = config.eval_type

    assert os.path.exists(gt_file)

    det_conf = []
    det_tp = []
    det_fp = []
    pos_count = 0
    bbox_count = 0
    im_count = 0

    detId = 0
    with open(gt_file, 'r') as fgt:

        for detId in xrange(detAll.shape[1]):
            det = detAll[cls,detId]

            detId = detId + 1
            gt = read_bboxes(fgt, cls, transform=config.transform_gt)

            num_gt = gt['num_bbox']
            iou_thresh = [default_iou_thresh] * num_gt
            gt_hit_mask = [False] * num_gt
            if num_gt > 0:
                if eval_type:
                    mask_pos = [x in eval_type for x in gt['bboxes'][:, 4]]
                else:
                    mask_pos = [True] * num_gt
                num_pos = len(np.where(mask_pos & (gt['bboxes'][:, 2] - gt['bboxes'][:, 0] >= min_width) &
                                       (gt['bboxes'][:, 3] - gt['bboxes'][:, 1] >= min_height))[0])
            else:
                num_pos = 0
            pos_count += num_pos
            bbox_count += num_gt


            num_det = det.shape[0]
            if num_det > 0:
                det_conf.append(det[:, 4])
                det_tp.append(np.zeros(num_det))
                det_fp.append(np.zeros(num_det))

            fp_box = []
            fp_box_iou = []

            for i in xrange(num_det):
                max_iou = -np.inf
                max_idx = -1
                det_bbox = det[i, :4]
                iou = 0

                for j in xrange(num_gt):
                    if gt_hit_mask[j] or not mask_pos[j]:
                        continue
                    gt_bbox = gt['bboxes'][j, :4]

                    x1 = max(det_bbox[0], gt_bbox[0])
                    y1 = max(det_bbox[1], gt_bbox[1])
                    x2 = min(det_bbox[2], gt_bbox[2])
                    y2 = min(det_bbox[3], gt_bbox[3])
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1

                    if w > 0 and h > 0:
                        s1 = (det_bbox[2] - det_bbox[0] + 1) * (det_bbox[3] - det_bbox[1] + 1)
                        s2 = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
                        s3 = w * h
                        iou = s3 / (s1 + s2 - s3)

                        if iou > iou_thresh[j] and iou > max_iou:
                            max_iou = iou
                            max_idx = j

                if max_idx >= 0:
                    det_tp[im_count][i] = 1
                    gt_hit_mask[max_idx] = True
                else:
                    det_fp[im_count][i] = 1
                    fp_box.append(det[i, :])
                    fp_box_iou.append(iou)


            if num_det>0:
                im_count = im_count + 1


    det_conf = np.hstack(det_conf)
    det_tp = np.hstack(det_tp)
    det_fp = np.hstack(det_fp)

    sort_idx = np.argsort(det_conf)[::-1]
    det_tp = np.cumsum(det_tp[sort_idx])
    det_fp = np.cumsum(det_fp[sort_idx])

    keep_idx = np.where((det_tp > 0) | (det_fp > 0))[0]
    det_tp = det_tp[keep_idx]
    det_fp = det_fp[keep_idx]


    recall = det_tp / pos_count
    precision = det_tp / (det_tp + det_fp)
    fppi = det_fp / im_count

    ap = compute_ap(recall, precision)
    fppi_pts, recall_pts = get_points(fppi, recall, [0.1])

    stat_str = 'AP = {:f}\n'.format(ap)
    for i, p in enumerate(fppi_pts):
        stat_str += 'Recall = {:f}, Miss Rate = {:f} @ FPPI = {:s}'.format(recall_pts[i], 1 - recall_pts[i], str(p))
    print stat_str

    return {'fppi': fppi, 'recall': recall, 'precision': precision, 'ap': ap,
            'recall_pts': recall_pts, 'fppi_pts': fppi_pts}




def plot_roc(data, id):
    plt.figure()
    for i, r in enumerate(data):
        plt.plot(r['fppi'], r['recall'], label=id[i], linewidth=2.0)
    plt.draw()
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xscale('log')
    plt.xlabel('FPPI', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.legend(loc='upper left', fontsize=10)
    plt.title('ROC Curve')
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.grid(b=True, which='minor', color='b', linestyle=':')


def plot_pr(data, id):
    plt.figure()
    for i, r in enumerate(data):
        plt.plot(r['recall'], r['precision'], label=id[i], linewidth=2.0)
    plt.draw()
    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(loc='lower left', fontsize=10)
    plt.title('PR Curve')
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.grid(b=True, which='minor', color='b', linestyle=':')
    plt.minorticks_on()


def plot_curves(eval_result, curve_id, i):
    plot_roc(eval_result, curve_id)
#    plot_pr(eval_result, curve_id)
#    plt.show()
    plt.savefig('psdbRoc2Full'+str(i));


if __name__ == '__main__':
    config = EvalConfig()
    config.iou_thresh = 0.5
    config.min_width = 0
    config.min_height = 0
    config.eval_type = [0, 1, 2, 3, 4, 5]
    config.transform_gt = True
    config.transform_det = False

    gt_file = 'data/psdb/phsb_rect_byimage_test.txt'


    for i in range(13, 21):
        print("Iteration", i)
        cacheFilename = \
               'output/pvanet_full/psdb_test/pvanet_frcnn_iter_' + str(i) + \
                '0000/detections.pkl'
        if (os.path.exists(cacheFilename)):
            eval_result = []
            for cls in range(1, 5):
                result = eval_roc_pr(config, gt_file, cacheFilename, cls)
                eval_result.append(result)

            det_id = ['pedestrian', 'head', 'upper body', 'lower body']
            plot_curves(eval_result, det_id, i)



