import numpy as np
from onlinelearning.ftrl_adp import FTRL_ADP

def ensemble(n,X_input,Z_input,Y_label,decay_choice,contribute_error_rate):
    errors=[]
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss=0
    z_loss=0
    lamda=0.5
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]
        p_x, decay_x,loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)

        z = Z_input[row]
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

        p=sigmoid(lamda*np.dot(w_x,x)+(1.0-lamda)*np.dot(w_z,z))

        x_loss+=loss_x
        z_loss+=loss_z
        lamda=np.exp(-eta*x_loss)/(np.exp(-eta*x_loss)+np.exp(-eta*z_loss))

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error

def logistic_loss(p,y):
    return (1/np.log(2.0))*(-y*np.log(p)-(1-y)*np.log(1-p))

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))