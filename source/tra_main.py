import warnings
warnings.filterwarnings("ignore")
from evaluation.helpers import *
from onlinelearning.online_learning import *
from em.trapezoidal_expectation_maximization2 import TrapezoidalExpectationMaximization2

if __name__ == "__main__":
    dataset = "ionosphere"
    batch_c = 8
    contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,isshuffle =get_cap_hyperparameter(dataset)

    # adjusting and generating trapezoidal data stream
    all_cont_indices = np.array([False]*11+[True]*12 +[False]*11)
    all_ord_indices = np.array([True]*11 +[False]*12 + [True]*11)
    # all_cont_indices = get_cont_indices(x)
    # all_ord_indices = ~all_cont_indices
    # get_tra_zero_datastream(dataset)

    file1 = open("../dataset/MaskData/" + dataset + "/X_trapezoid.txt", 'r')
    X_zero = pd.read_csv("../dataset/MaskData/" + dataset +"/X_trapezoid_zeros.txt", sep=" ", header=None)
    Y_label = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep=' ', header=None)

    Y_label = Y_label.values
    Y_label = Y_label.flatten()

    X_masked= file1.readlines()
    X_zero=np.array((X_zero))
    n=len(X_masked)
    X_masked=chack_Nan(X_masked,n)

    #getting the hyperparameter
    BATCH_SIZE = math.ceil(n / batch_size_denominator)
    WINDOW_SIZE = math.ceil(n / window_size_denominator)
    WINDOW_WIDTH = len(X_masked[BATCH_SIZE])
    cont_indices = all_cont_indices[:WINDOW_WIDTH]
    ord_indices = all_ord_indices[:WINDOW_WIDTH]

    #starting trapezoidale imputation
    tra = TrapezoidalExpectationMaximization2(cont_indices, ord_indices, window_size=WINDOW_SIZE,window_width=WINDOW_WIDTH)
    j = 0
    X_imp = []
    Z_imp = []
    start = 0
    end = BATCH_SIZE
    WINDOW_WIDTH = len(X_masked[0])

    while end <= n:
        X_batch = X_masked[start:end]
        if decay_coef_change == 1:
            this_decay_coef = batch_c / (j + batch_c)
        else:
            this_decay_coef = 0.5
        if len(X_batch[-1]) > WINDOW_WIDTH:
            WINDOW_WIDTH = len(X_batch[-1])
            cont_indices = all_cont_indices[:WINDOW_WIDTH]
            ord_indices = all_ord_indices[:WINDOW_WIDTH]

        for i, row in enumerate(X_batch):
            now_width = len(row)
            if now_width < WINDOW_WIDTH:
                row = row + [np.nan for i in range(WINDOW_WIDTH - now_width)]
                X_batch[i] = row
        X_batch = np.array(X_batch)
        Z_imp_batch, X_imp_batch = tra.partial_fit_and_predict(X_batch, cont_indices, ord_indices,
                                                                max_workers=1, decay_coef=0.5)
        Z_imp.append(Z_imp_batch[0].tolist())
        X_imp.append(X_imp_batch[0].tolist())
        start = start + 1
        end = start + BATCH_SIZE
    for i in range(1, BATCH_SIZE):
        Z_imp.append(Z_imp_batch[i].tolist())
        X_imp.append(X_imp_batch[i].tolist())

    #getting the CER
    X_input1 = Z_imp
    X_input2 = X_zero
    temp = np.ones((n, 1))
    X_input2 = np.hstack((temp, X_input2))

    if isshuffle == True:
        perm=np.random.seed(1)
        np.random.shuffle(perm)
        Y_label = Y_label[perm]

    error_arr_Z = generate_tra(n, X_input1, Y_label, decay_choice, contribute_error_rate)
    error_arr_X = generate_tra(n, X_input2, Y_label, decay_choice, contribute_error_rate)
    draw_tra_error_picture(error_arr_Z,error_arr_X)