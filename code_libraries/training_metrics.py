from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

def prob_cutoff_confusion_matrix(model, val_data,true_labels, prob_cutoff = 0.5):

    rounded_probs = np.empty(len(true_labels))

    probs = model.predict_proba(val_data)

    for i in range(len(probs)):

        if probs[i][1] >= prob_cutoff:

            rounded_probs[i] = 1

        else:

            rounded_probs[i] = 0

    return confusion_matrix(rounded_probs , true_labels)



def calib_curve_svc(model , val_data , true_labels , bins , model_name):

    probs = model.predict_proba(val_data)
    probs_1 = probs[: , 1]
    true_pos, pred_pos = calibration_curve(true_labels , probs_1 , n_bins=bins)

    plt.plot(pred_pos,
         true_pos, 
         marker='o', 
         linewidth=1, 
         label=str(model_name))

#Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
    plt.plot([0, 1], 
             [0, 1], 
             linestyle='--', 
             label='Perfectly Calibrated')
    
    
    # Set the title and axis labels for the plot
    #plt.title('Probability Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    
    # Add a legend to the plot
    plt.legend(loc='best')
    
    # Show the plot
    plt.show()


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

def calib_curve_nn(model, val_data, true_labels, bins=10, model_name='NN'):

    
    # Predict probabilities
    probs = model.predict(val_data).flatten()  # 1D array for binary classifier
    
    # Compute calibration curve
    true_pos, pred_pos = calibration_curve(true_labels, probs, n_bins=bins)
    
    # Plot calibration curve
    plt.figure(figsize=(6,6))
    plt.plot(pred_pos, true_pos, marker='o', linewidth=1, label=str(model_name))
    
    # Perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Probability Calibration Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()

    