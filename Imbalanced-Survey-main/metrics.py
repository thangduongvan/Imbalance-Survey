from sklearn.metrics import confusion_matrix
import numpy as np
# from imblearn.metrics import sensitivity_score, specificity_score
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
# from imblearn.metrics import sensitivity_score, specificity_score


def get_cm(y_true, y_pred):
    """Return confusion matrix with binary class (only recieve 0 or 1)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn, fp, fn, tp


def get_binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = get_cm(y_true, y_pred)

    try:
        recall = tp / (tp+fn)
    except ZeroDivisionError:
        recall = 0 #float('nan')
    try:
        precision = tp / (tp+fp)
    except ZeroDivisionError:
        precision = 0 #float('nan')
    try:
        specificity = tn / (tn+fp)
    except ZeroDivisionError:
        specificity = 0 #float('nan')
    
    sensitivity = recall
    fscore = (2*precision*recall) / (precision+recall)
    gmean = np.sqrt(sensitivity*specificity)
    balanced_acc = (sensitivity+specificity)*0.5
    accuracy = (tp+tn)/(tn + fp + fn + tp)
    # compute mathew còef
    numerator = tp*tn-fp*fn
    denominator = np.sqrt(
        (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    )
    try:
        mcc = numerator/denominator
    except ZeroDivisionError:
        mcc = 0 #float('nan')

    return {
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'gmean': gmean,
        'balanced_acc': balanced_acc,
        'mcc': mcc,
        'accuracy': accuracy
    }


# def get_metrics(self, y_true, y_pred):
#     """Implement metrics"""
#     precision = precision_score(y_true, y_pred, average='binary')
#     recall = recall_score(y_pred, y_pred, average='binary')
#     f1 = f1_score(y_true, y_pred, average='binary')
#     sensitivity = sensitivity_score(y_true, y_pred, average='binary')
#     specificity = specific_score(y_true, y_pred, average='binary')
#     gmean = np.sqrt(sensitivity*specificity)
#     mathew = matthews_corrcoef(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)

#     return {
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'sensitivity': sensitivity,
#         'specificity': specificity,
#         'gmean': gmean,
#         'mathew': mathew,
#         'accuracy': accuracy
#     }
