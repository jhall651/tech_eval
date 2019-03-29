
# Function for making segments of time-series
import numpy as np
def splitscalewindow(df, t_steps=5, split=.7):

    # Split features and targets
    X = np.array(df.iloc[:,:-1].copy())
    y = np.array(df.iloc[:,-1].copy())

    train_n = int(len(df) * split)

    # Scale data
    import sys
    eps = sys.float_info.epsilon
    mu = X[:train_n,:].mean(axis=0)
    st = X[:train_n,:].std(axis=0)

    X -= mu
    X = (X + eps) / (st + eps)

    X_slices = []
    y_slices = []
    for i in range(t_steps, len(X)):
        X_slices.append(X[i-t_steps:i,:])
        y_slices.append(y[i-t_steps:i])
    X_slices = np.array(X_slices)
    y_slices = np.array(y_slices)

    # Split into train and test
    X_train = X_slices[:train_n,:,:]
    y_train = y_slices[:train_n,-1]
    X_test = X_slices[train_n:,:,:]
    y_test = y_slices[train_n:,-1]

    # One-hot-encode
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    return(X_train, X_test, y_train, y_test)

import itertools
import pandas as pd
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Function to create triple barrier labels (stop loss, profit target, time stop)
def applyPtSl(close,ptSlTs):
    # Close prices
    # Earliest Time Stop
    out = pd.DataFrame(close.index + pd.Timedelta(days=ptSlTs[2]),index=close.index,columns=['t1'])
    # Standard deviation of returns after 20 period exponential moving average (for Stops and Profit Targets)
    vol = close.pct_change().ewm(20).std()
    pt = ptSlTs[0]*vol
    sl = -ptSlTs[1]*vol
    # Loop through each observation and find which barrier was hit first (stop loss, profit target, time stop)
    for loc,t1 in out['t1'].fillna(close.index[-1]).iteritems():
        # Path prices
        df0=close.loc[loc:t1]
        # Path returns
        df0=df0/df0.loc[loc]-1
        # Earliest stop
        out.loc[loc,'sl']=df0[df0<sl[loc]].dropna(axis=0,how='all').index.min()
        # Earliest profit taking
        out.loc[loc,'pt']=df0[df0>pt[loc]].dropna(axis=0,how='all').index.min()
        # Flatten one-hot-encoding

    return(out.idxmin(axis=1))

def build_LSTM(input_shape,output_shape):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras import regularizers
    # Initialising the RNN
    model = Sequential()

    # Adding first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 38,
                       return_sequences = True,
                       activation='relu',
                       kernel_regularizer=regularizers.l1(.0001),
                       input_shape = (input_shape[0], input_shape[1])
                       ))
    model.add(Dropout(0.2))

    # Adding additional LSTM layers with Dropout regularisation

    model.add(LSTM(units = 38,
                    return_sequences = True,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(.0001)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 24,
                    return_sequences = False,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(.0001)))
    model.add(Dropout(0.2))

    # Final Layer
    model.add(Dense(units = output_shape, activation='softmax'))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    return(model)


def class_report(X_test,y_test,model):
    from sklearn.metrics import classification_report
    y_preds = model.predict(X_test)
    y_preds_flat = pd.DataFrame(y_preds, columns=['pt','sl','t1']).idxmax(1)
    y_test_flat = pd.DataFrame(y_test, columns=['pt','sl','t1']).idxmax(1)
    rep = classification_report(y_test_flat, y_preds_flat)
    return(rep)

# These functions are for fractionally differentiating
def getWeights_FFD(d,thres):
    w,k = [1.0],1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_)
        k+=1
    w = np.array(w[::-1]).reshape(-1,1)
    return(w)

def fracDiff_FFD(series,d,thres=1e-5):
    '''
    Constant width window
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive frational, not necessarily bounded [0,1]
    '''
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1=seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue
            df_[loc1] = np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name] =df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return(df)

def plotMinFFD(df,colname):
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    import numpy as np

    df0 = df.copy()
    out = pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    for d in np.linspace(0,1,11):
        df1 = np.log(df0[[colname]]).resample('1D').last()
        df2=fracDiff_FFD(df1,d,thres=.01)
        corr = np.corrcoef(df1.loc[df2.index,colname],df2[colname])[0,1]
        df2=adfuller(df2[colname], maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr]
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    return

# Function to generate class weight balancing dictionary
def getbalclassweights(y_train):
    class_weights_bal = dict(
                    (1/(y_train.sum()/len(y_train))).reset_index(drop=True))
    return(class_weights_bal)

def build_CNNLSTM(input_shape,output_shape):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Conv1D
    from keras.layers import Dropout
    from keras import regularizers
    # Initialising the RNN
    model = Sequential()

    # Adding first LSTM layer and some Dropout regularisation
    model.add(Conv1D(filters=96,kernel_size=(3,3), padding='same',
                       #activation='relu',
                       #kernel_regularizer=regularizers.l1(.0001),
                       input_shape = (input_shape[0], input_shape[1])
                       ))
    model.add(Dropout(0.4))

    # Adding additional LSTM layers with Dropout regularisation
    model.add(LSTM(units = 38,
                    return_sequences = True,
                    activation='relu',
                    kernel_regularizer=regularizers.l1(.0001)))
    model.add(Dropout(0.4))


    model.add(LSTM(units = 9,
                    return_sequences = False,
                    activation='relu',
                    kernel_regularizer=regularizers.l1(.0001)))
    model.add(Dropout(0.4))

    # Final Layer
    model.add(Dense(units = output_shape, activation='softmax'))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    return(model)
