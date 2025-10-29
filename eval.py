import json
import numpy as np
from util import make_ts_image_for_testing
from sklearn.metrics import precision_score, recall_score, f1_score
from fastprogress import progress_bar
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


def statistical_thr(scores, k=1.0):
    scores = np.asarray(scores)    
    mu = np.mean(scores)
    sigma = np.std(scores)
    threshold = mu + k * sigma
    pred_labels = (scores > threshold).astype(int)
    return threshold, pred_labels


def get_scores(gt_labels, pred_labels):
    if np.count_nonzero(gt_labels) == 0 and np.count_nonzero(pred_labels) == 0:
        f1 = precision = recall = 1
    elif np.count_nonzero(gt_labels) == 0 or np.count_nonzero(pred_labels) == 0:
        f1 = precision = recall = 0
    else:
        f1 = f1_score(gt_labels, pred_labels)
        precision = precision_score(gt_labels, pred_labels)
        recall = recall_score(gt_labels, pred_labels)
    return f1, precision, recall


def gaussian_eval(gt_labels, pred_labels):
    best_precision = best_recall = best_f1 = 0
    best_predicted = pred_labels

    for sigma in range(0, 10):
        if sigma != 0:
            g_predicted = gaussian_filter1d(pred_labels, sigma=sigma)
            _, final_predicted = statistical_thr(g_predicted)
        else:
            final_predicted = pred_labels

        # evaluation
        f1, precision, recall = get_scores(gt_labels, final_predicted)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall 
            best_predicted = final_predicted

    return best_f1, best_precision, best_recall, best_predicted


def evaluation(cfg, value_list, label_list, json_file, data_length):
    print('evaluating...')
    predicted = []
    precision_list = []
    recall_list = []
    f1score_list = []

    with open(json_file, 'r') as file:
        data = json.load(file)
        for num in progress_bar(range(data_length), total=data_length):
            predicted_interval_list = data[num]
            values = value_list[num]
            label = label_list[num]
            length = len(label)

            # parsing
            parsed = json.loads(predicted_interval_list)
            result = [(item["start"], item["end"]) for item in parsed]
            
            # make predicted arr
            predicted = [0 for _ in range(length)]
            for answer in result:
                start, end = answer[0], answer[1] # interval
                predicted[start:end] = [1 for _ in range(start, end)]

            # evaluation
            f1score, precision, recall, predicted = gaussian_eval(label, predicted)

            precision_list.append(precision)
            recall_list.append(recall)
            f1score_list.append(f1score)

            # visualization
            if cfg.visualization:
                image_path = f'./experiment/{cfg.dataset_name}/{cfg.llm_model}/{cfg.class_name}/{cfg.class_name}_predicted_{num}.jpg'
                make_ts_image_for_testing(cfg, image_path, length, values, predicted, label, precision, recall, f1score)

    # scoring
    print('* average precision:', round(np.mean(np.array(precision_list)),3))
    print('* average recall:', round(np.mean(np.array(recall_list)),3))
    print('* average f1score:', round(np.mean(np.array(f1score_list)),3))
    

# ============================================================================================
# ============================================================================================


# macro F1 score
def fsm_evaluation(cfg, label_list, json_file, data_length):
    print('evaluating...')
    all_precision_list = []
    all_recall_list = []
    all_f1score_list = []

    with open(json_file, 'r') as file:
        data = json.load(file)
        for num in progress_bar(range(data_length), total=data_length):
            current_data = data[num]
            label = label_list[num]
            length = len(label)

            # parsing
            intevrval_arr = json.loads(current_data['interval'])

            all_predicted = []
            for k in range(cfg.top_k+1):
                if len(intevrval_arr[k]) > 0:
                    result = [(item["start"], item["end"]) for item in intevrval_arr[k]]
                else:
                    result = []

                # make predicted arr
                predicted = [0 for _ in range(length)]
                for answer in result:
                    start, end = int(answer[0]), int(answer[1]) # interval
                    predicted[start:end] = [1 for _ in range(start, end)]
                all_predicted.append(predicted)
            
            # evaluation (0~1)
            precision_list = []
            recall_list = []
            f1score_list = []
            ffp_predicted = np.mean(np.array(all_predicted[:cfg.top_k]), axis=0)
            org_predicted = np.array(all_predicted[-1])
            
            for a in range(0, 11):
                a = a * 0.1
                final_predicted = a * ffp_predicted + (1-a) * org_predicted
                _, pred_label = statistical_thr(final_predicted)
                f1, precision, recall, _ = gaussian_eval(label, pred_label)

                precision_list.append(precision)
                recall_list.append(recall)
                f1score_list.append(f1)

            all_precision_list.append(precision_list)
            all_recall_list.append(recall_list)
            all_f1score_list.append(f1score_list)

    # scoring
    best_a, best_f1, best_pcs, best_rcl = 0, 0, 0, 0
    for a in range(0, 11):
        precision_list = [row[a] for row in all_precision_list]
        recall_list = [row[a] for row in all_recall_list]
        f1score_list = [row[a] for row in all_f1score_list]

        mean_precision = round(np.mean(np.array(precision_list)),3)
        mean_recall = round(np.mean(np.array(recall_list)),3)
        mean_f1 = round(np.mean(np.array(f1score_list)),3)

        if mean_f1 > best_f1:
            best_a = a
            best_f1, best_pcs, best_rcl = mean_f1, mean_precision, mean_recall

        if a == 0:
            print(f'* original precision:', mean_precision)
            print(f'* original recall:', mean_recall)
            print(f'* original f1score:', mean_f1)
            print('----------------------------------------------------------------------\n')

    print(f'* best precision (a={best_a}):', best_pcs)
    print(f'* best recall (a={best_a}):', best_rcl)
    print(f'* best f1score (a={best_a}):', best_f1)
    return best_a


# macro version
def fsm_visualization(cfg, value_list, label_list, json_file, data_length, best_a):
    print('visualizing...')

    with open(json_file, 'r') as file:
        data = json.load(file)
        for num in progress_bar(range(data_length), total=data_length):
            current_data = data[num]
            values = value_list[num]
            label = label_list[num]
            length = len(label)

            # parsing
            intevrval_arr = json.loads(current_data['interval'])

            all_predicted = []
            for k in range(cfg.top_k+1):
                if len(intevrval_arr[k]) > 0:
                    result = [(item["start"], item["end"]) for item in intevrval_arr[k]]
                else:
                    result = []

                # make predicted arr
                predicted = [0 for _ in range(length)]
                for answer in result:
                    start, end = int(answer[0]), int(answer[1]) # interval
                    predicted[start:end] = [1 for _ in range(start, end)]
                all_predicted.append(predicted)

            # evaluation
            ffp_predicted = np.mean(np.array(all_predicted[:cfg.top_k]), axis=0)
            org_predicted = np.array(all_predicted[-1])

            best_predicted = best_a * ffp_predicted + (1-best_a) * org_predicted
            _, best_pred_label = statistical_thr(best_predicted)
            best_f1, best_precision, best_recall, best_pred_label = gaussian_eval(label, best_pred_label)

            org_predicted = np.array(all_predicted[-1])
            org_pred_label = org_predicted
            org_f1, org_precision, org_recall = get_scores(label, org_pred_label)

            # visualization - best method
            image_path = f'./experiment/{cfg.dataset_name}/{cfg.llm_model}/{cfg.class_name}/{cfg.class_name}_predicted_best_{num}.jpg'
            make_ts_image_for_testing(cfg, image_path, length, values, best_pred_label, label, best_precision, best_recall, best_f1)

            # visualization - original method
            image_path = f'./experiment/{cfg.dataset_name}/{cfg.llm_model}/{cfg.class_name}/{cfg.class_name}_predicted_org_{num}.jpg'
            make_ts_image_for_testing(cfg, image_path, length, values, org_pred_label, label, org_precision, org_recall, org_f1)
