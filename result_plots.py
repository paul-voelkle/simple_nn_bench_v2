from plot_utils import plot_2d, plot_settings
from utilities import TestResults, config, confirm
import numpy as np

def plot_result(names:list[str]): 
    auc_curves_tpr = []
    auc_curves_fpr = []
    auc_score = []
    inv_auc_score = []
    linestyle = []

    # plot_settings.font_size_text = 12
    # plot_settings.set_scale()
    # plot_settings.set_font()
    
    save_path='.'
    
    if confirm(f"Change save path? Default path: {save_path}"):
        path = input()

    for name in names:
        path = f"{config.path_results}/{name}"
        exec(f"res_{name} = TestResults().load(path)")
        exec(f"auc_curves_tpr.append(res_{name}.tpr)")
        exec(f"auc_curves_fpr.append(res_{name}.fpr)")
        exec(f"auc_score.append(np.round(res_{name}.auc_score,3))")
        exec(f"inv_auc_score.append(np.round(res_{name}.bck_rej,3))")        
        linestyle.append('solid')

    auc_curves_tpr.append(TestResults().rnd_class)
    auc_curves_fpr.append(TestResults().rnd_class)
    linestyle.append('--')

    auc_labels = []
    inv_auc_labels = []

    for i in range(len(auc_score)):
        auc_labels.append(f"{names[i]}: AUC = {auc_score[i]}")
        inv_auc_labels.append(f"{names[i]}: 1/FPR(0,5) = {inv_auc_score[i]}")

    auc_labels.append('Rand classifier')   
    inv_auc_labels.append('Rand classifier')   

    plot_2d(x=auc_curves_fpr,
            y=auc_curves_tpr, 
            labels=auc_labels, 
            X_label='FPR', 
            Y_label='TPR', 
            #xticks = TICKS,
            #yticks = TICKS,
            path=save_path,
            fname='roc_curve_combined.png',
            linestyle=linestyle)
    
    plot_2d(x=auc_curves_tpr,
        y=[1/curve for curve in auc_curves_fpr], 
        labels=inv_auc_labels, 
        X_label='TPR', 
        Y_label='log(1/FPR)',
        Y_scale='log',
        path=save_path,
        fname='roc_curve_inv_fpr_combined.png', 
        linestyle=linestyle)

