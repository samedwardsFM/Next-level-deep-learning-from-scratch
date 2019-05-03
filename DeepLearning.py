# Imports
# Check if CuPy is installed to enable GPU processing otherwise use Numpy on CPU
try:
    import cupy as np
    print('Training network on GPU...')
    print()
except ImportError:
    import numpy as np
    print("Couldn't load CuPy, training network on CPU...")
    print()
import numpy
import sys
import copy


def progress(count, total, status='', fill=u'\u2588'):
    """
    Print progress bar to show progress of training
    :param count: is the current count
    :param total: is the maximum count
    :param status: is a text to be printed along the progress bar
    :param fill: is the character used as fill for progress bar
    :return:
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = fill * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r [%s] %s%s %s ' % (bar, percents, '%', status))
    sys.stdout.flush()


class Utils:
    """
    Contains to_categorical()
    """

    def to_categorical(array):
        """
        converts each element of an array to one hot encoded vector
        :return: array of one hot encoded vectors
        """
        x = numpy.array(array)
        unique = numpy.unique(x)
        n_categories = len(unique)
        result = numpy.zeros((x.shape[0], n_categories))
        result[range(len(x)), x] = 1
        return result


class Activation:
    """
    This class contains all activation functions
    usage:
    Activation('name')

    where name is the name of activation function

    Returns f and f_deriv methods that corresponds to the function and it's
    derivative respectively

    Currently included functions are:
    tanh, sigmoid, relu, leaky relu, elu, softmax and linear
    """

    def __tanh(self, x):
        """
        tanh function
        :param x: is a variable
        :return: tanh(x)
        """
        return np.tanh(x)

    def __tanh_deriv(self, x):
        """
        tanh deriviative
        :param x: is a variable
        :return: derivative of tanh(x)
        """
        return 1.0 - np.tanh(x)** 2

    def __sigmoid(self, x):
        """
        sigmoid function
        :param x: is a variable
        :return: sigmoid(x)
        """
        return 1.0 / (1.0 + np.exp(-x))

    def __sigmoid_deriv(self, x):
        """
        sigmoid deriviative
        :param x: is a variable
        :return: derivative of sigmoid(x)
        """
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def __relu(self, x):
        """
        rectified linear unit function
        :param x: is a variable
        :return: relu(x)
        """
        return x * (x > 0)

    def __relu_deriv(self, x):
        """
        derivative of a rectified linear unit function
        :param x: is a variable
        :return: derivative relu(x)
        """
        return 1 * (x > 0)

    def __lrelu(self, x):
        """
        leaky rectified linear unit function
        :param x: is a variable
        :return: lrelu(x)
        """
        return x * (x > 0) + 0.01 * x * (x <= 0)

    def __lrelu_deriv(self, x):
        """
        derivative of a leaky rectified linear unit function
        :param x: is a variable
        :return: derivative lrelu(x)
        """
        return 1 * (x > 0) + 0.01 * (x <= 0)

    def __elu(self, x):
        """
        exponential linear unit function
        :param x: is a variable
        :return: elu(x)
        """
        return x * (x > 0) + 0.01 * (np.exp(x) - 1) * (x <= 0)

    def __elu_deriv(self, x):
        """
        derivative of exponential linear unit function
        :param x: is a variable
        :return: derivative of elu(x)
        """
        return 1 * (x > 0) + 0.01 * np.exp(x) * (x <= 0)

    def __softmax(self, x):
        """
        softmax function
        :param x: is a variable
        :return: softmax(x)
        """
        # stable softmax implementation by shifting max(x) to avoid large exp(x)
        exps = np.atleast_2d(np.exp(x - np.max(x)))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def __softmax_deriv(self, x):
        # Not used
        s = self.__softmax(x)
        SM = s.reshape((-1, 1))
        return np.diagflat(s) - np.dot(SM, SM.T)

    def __linear(self, x):
        """
        linear function
        :param x: is a variable
        :return: x
        """
        return np.array(x)

    def __linear_deriv(self, x):
        """
        derivative of a linear function
        :param x: is a variable
        :return: 1
        """
        x = np.asarray(x)
        return np.ones(x.shape)

    def __init__(self, activation='tanh'):
        """
        :param activation: name of activation fuction
        """
        # softmax derivetaive is not used here, but rather included inside cross-entropy loss
        act = {'sigmoid': [self.__sigmoid, self.__sigmoid_deriv],
               'tanh': [self.__tanh, self.__tanh_deriv],
               'lrelu': [self.__lrelu, self.__lrelu_deriv],
               'relu': [self.__relu, self.__relu_deriv],
               'elu': [self.__elu, self.__elu_deriv],
               'softmax': [self.__softmax, self.__linear_deriv],
               'linear': [self.__linear, self.__linear_deriv]
               }
        self.f = act[activation][0]
        self.f_deriv = act[activation][1]


class Layers:
    """
    Creates neural network layers
    usage:
    Layers()

    Currently included layer type(s):
    Dense
    """
    previous_activation_deriv = None

    class Dense:
        """
        Fully connected neural network layer
        usage:
        Dense(depth, activation='linear', dropout_rate=0, batch_norm=None, **kwargs)
        where:
            :param depth: is the number of neurons in this layer
            :param activation: is the the activation function used on the output neurons of this layer
            :param dropout_rate: is the dropout_rate used on the output neurons of this layer
            :param batch_norm: is the number of neurons in this layer

        """
        input_dim = []
        output = None
        dropout_vec = None

        def __init__(self, depth, activation='linear', dropout_rate=0, batch_norm=None, **kwargs):

            def initialize_weight(size):
                """
                initialize a weight matrix
                :param size: is size of weight matrix
                :return: intialized weight matrix
                """
                # Xavier uniform weight initialization
                return np.random.uniform(low=-np.sqrt(6. / size[0] + size[1]),
                                         high=np.sqrt(6. / size[0] + size[1]), size=size)

            def initialize_bias(size):
                """
                initialize bias vectors by setting their element values to zero
                :param size: is size of bias vector
                :return: zero intialized bias vector
                """
                return np.zeros(size)

            allowed_kwargs = ['input_dim']

            for kwarg in kwargs:
                if kwarg not in allowed_kwargs:
                    raise TypeError('Invalid keyword argument:',
                                    kwarg, ', Allowed keywords are:',
                                    *allowed_kwargs)

            self.activation = Activation(activation).f

            if len(self.input_dim) >= 1:
                self.W = initialize_weight((self.input_dim[-1], depth))
                if activation == 'sigmoid':
                    self.W *= 4
                self.B = initialize_bias((depth,))
                self.depth = depth

                # initialize variables
                self.dW = np.zeros(self.W.shape)
                self.dW_2 = np.zeros(self.W.shape)
                self.dB = np.zeros(self.B.shape)
                self.dB_2 = np.zeros(self.B.shape)
                self.xdW = np.zeros(self.W.shape)
                self.xdB = np.zeros(self.B.shape)

            if 'input_dim' in kwargs:
                self.input_dim.clear()
                self.input_dim.append(kwargs['input_dim'])

            # initialize variables
            self.input_dim.append(depth)
            self.depth = depth
            self.activation_deriv = Activation(activation).f_deriv
            self.input = None
            self.output_x = None
            self.mean = None
            self.var = None
            self.norm = None
            self.moving_mean = np.zeros((1, depth))
            self.moving_var = np.zeros((1, depth))
            self.dropout_rate = dropout_rate
            self.bn_gamma = None
            self.bn_beta = None
            self.bn_dgamma = None
            self.bn_dbeta = None
            self.bn_dgamma_2 = 0
            self.bn_dbeta_2 = 0
            self.bn_epsilon = None
            if batch_norm:
                assert len(batch_norm) == 3, 'batch normalization must contain 3 values (gamma, beta, epsilon)'
                self.bn_gamma, self.bn_beta, self.bn_epsilon = batch_norm

        def batch_norm_forward(self, output_y, training=True):
            """
            Batch normalization forward pass
            :param output_y: layer output (before activation)
            :param training: a flag to check if this forward pass is for training or prediction
            :return: batch normalized output
            """
            if training:
                if self.bn_gamma is None or self.bn_beta is None or self.bn_epsilon is None:
                    return output_y
                self.mean = np.mean(output_y, axis=0)
                self.var = np.var(output_y, axis=0)
                self.norm = (output_y - self.mean)/np.sqrt(self.var + self.bn_epsilon)
                self.moving_mean = 0.9 * self.moving_mean + 0.1 * self.mean
                self.moving_var = 0.9 * self.var + 0.1 * self.moving_var
            else:
                # estimate normalized output at test time using moving average and moving variance
                self.norm = (output_y - self.moving_mean) / np.sqrt(self.moving_var + self.bn_epsilon)
            return self.bn_gamma * self.norm + self.bn_beta

        def forward(self, training=True):
            """
            forward pass
            :param training: a flag to check if this forward pass is for training or prediction
            :return: output from forward pass
            """

            self.z = self.input.dot(self.W) + self.B
            self.z = self.batch_norm_forward(self.z)
            self.output = self.activation(self.z)

            # applying dropout in training phase only
            if training and self.dropout_rate > 0:
                self.dropout_vec = np.random.binomial(1, (1 - self.dropout_rate), self.output.shape[1])
                self.output *= self.dropout_vec / (1 - self.dropout_rate)

            return self.output

        def batch_norm_backward(self, dout):
            """
            batch normalization backward pass
            :param dout: layer's output gradient
            :return:
            """
            if self.bn_gamma is None or self.bn_beta is None or self.bn_epsilon is None:
                return dout

            m, d = self.z.shape

            x_mu = self.z - self.mean
            std_inv = 1.0 / np.sqrt(self.var + 1e-8)

            dx_norm = dout * self.bn_gamma
            dvar = np.sum(dx_norm * x_mu, axis=0) * -.5 * std_inv ** 3
            dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0)

            dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / m) + (dmu / m)
            self.bn_dgamma = np.sum(dout * self.norm, axis=0)
            self.bn_dbeta = np.sum(dout, axis=0)

            # initialize bn_dgamma_2 and bn_dbeta_2 for RMSProp
            if self.bn_dgamma_2 is None:
                self.bn_dgamma_2 = np.zeros(self.bn_dgamma)
                self.bn_dbeta_2 = np.zeros(self.bn_dbeta)

            return dx

        def backward(self, delta):
            """
            backward pass
            :param delta: dLdY
            :return: dLdX
            """

            # dLdZ = dLdY * dYdZ
            dLdY_dYdZ = np.array(delta) * self.activation_deriv(self.z)

            # Batch norm backward pass
            dLdY_dYdZ = self.batch_norm_backward(dLdY_dYdZ)

            # dLdW = dLdY * dYdZ * dZdW
            self.dW = self.input.T.dot(dLdY_dYdZ)

            # dLdB = dLdY * dYdZ * dZdB
            self.dB = np.sum(dLdY_dYdZ, axis=0)

            # dLdX = dLdY * dYdZ * dZdX
            dLdY_dYdZ_dZdX = dLdY_dYdZ.dot(self.W.T)
            return dLdY_dYdZ_dZdX


class Losses:
    """
    Contains objective functions
    Implemented loss functions:
    MSE and softmax_cross_entropy
    """

    def MSE(y_true, y_pred):
        """
        mean square error
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: loss, dLdY
        """
        # MSE
        error = y_pred - y_true
        loss = np.sum(error ** 2 / 2) / y_true.shape[0]
        # calculate the delta of the output layer
        delta = error / y_true.shape[0]
        # return loss and delta
        return loss, delta

    def softmax_cross_entropy(y_true, y_pred):
        """
        softmax with cross entropy loss
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: loss, dLdY
        """
        epsilon = 10e-8
        # clip log values to avoid problems
        y_pred = np.clip(y_pred, a_min=epsilon, a_max=1-epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        delta = (y_pred - y_true) / y_true.shape[0]
        return loss, delta


class Metrics:
    """
    Metrics
    Implemented metrics:
    accuracy
    """
    def accuracy(y_true, y_pred):
        """
        compute accuracy
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: accuracy
        """
        n = y_pred.shape[0]
        return float(np.sum(1 * (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))/n)


class Regularizers:
    """
    Regularizers
    implemented regularizers:
    L1 & L2
    """

    def l1(layer, reg_lambda=0.01):
        """
        l1 regularizer
        :param layer: layer to regularize
        :param reg_lambda: regulariztion parameter
        :return: regularization gradient
        """
        dl1dw = reg_lambda * np.sign(layer.W)
        return dl1dw

    def l2(layer, reg_lambda=0.01):
        """
        l2 regularizer
        :param layer: layer to regularize
        :param reg_lambda: regulariztion parameter
        :return: regularization gradient
        """
        dl2dw = reg_lambda * layer.W
        return dl2dw


class Optimizers:
    """
    Optimizers
    implemented optimizers:
    gradient descend (SGD), AdaGrad and RMSProp
    """

    def SGD(lr, layers, regularizer=None, reg_lambda=0.01, gamma=0, decay=0):
        """
        gradient descend
        :param lr: learning rate
        :param layers: layers to optimize
        :param regularizer: regularization
        :param reg_lambda: regularization parameter
        :param gamma: momentum parameter
        :param decay: not used in SGD
        """
        for layer in layers[1:]:

            # weights gradient
            dW = layer.dW

            # add regularization gradient
            if regularizer:
                dW += regularizer(layer, reg_lambda)

            # update batch norm parameters
            if layer.bn_gamma is not None:
                layer.bn_gamma -= lr * layer.bn_dgamma
                layer.bn_beta -= lr * layer.bn_dbeta

            # update gradients and add momentum
            dW += lr * dW + gamma * layer.xdW
            dB = lr * layer.dB + gamma * layer.xdB

            # record update for next step's momentum
            if gamma > 0:
                layer.xdW = dW
                layer.xdB = dB

            # update gradients in the layer
            layer.dW -= dW
            layer.dB -= dB

    def AdaGrad(lr, layers, regularizer=None, reg_lambda=0.01, gamma=0, decay=0.0):
        """
        AdaGrad
        :param lr: learning rate
        :param layers: layers to optimize
        :param regularizer: regularization
        :param reg_lambda: regularization parameter
        :param gamma: not used for AdaGrad
        :param decay: learning rate decay
        """
        for layer in layers[1:]:

            # weights gradient
            dW = layer.dW

            # add regularization gradient
            if regularizer:
                dW += regularizer(layer, reg_lambda)

            # update batch norm parameters
            if layer.bn_gamma is not None:
                layer.bn_dgamma_2 += layer.bn_dgamma * layer.bn_dgamma
                layer.bn_gamma -= lr * layer.bn_dgamma / (np.sqrt(layer.bn_dgamma_2) + 1e-7)
                layer.bn_dbeta_2 += layer.bn_dbeta * layer.bn_dbeta
                layer.bn_beta -= lr * layer.bn_dbeta / (np.sqrt(layer.bn_dbeta_2) + 1e-7)

            # update weights
            layer.dW_2 += dW * dW
            layer.W -= lr * dW / (np.sqrt(layer.dW_2) + 1e-7)

            # update biases
            layer.dB_2 = decay * layer.dB_2 + (1 - decay) * layer.dB * layer.dB
            layer.B -= lr * layer.dB / (np.sqrt(layer.dB_2) + 1e-7)

    def RMSProp(lr, layers, regularizer=None, reg_lambda=0.01, gamma=0, decay=0.9):
        """
        RMSProp
        :param lr: learning rate
        :param layers: layers to optimize
        :param regularizer: regularization
        :param reg_lambda: regularization parameter
        :param gamma: not used for RMSProp
        :param decay: learning rate decay
        """
        for layer in layers[1:]:

            # weights gradient
            dW = layer.dW

            # add regularization gradient
            if regularizer:
                dW += regularizer(layer, reg_lambda)

            # update batch norm parameters
            if layer.bn_gamma is not None:
                layer.bn_dgamma_2 = decay * layer.bn_dgamma_2 + (1 - decay) * layer.bn_dgamma * layer.bn_dgamma
                layer.bn_gamma -= lr * layer.bn_dgamma / (np.sqrt(layer.bn_dgamma_2) + 1e-7)
                layer.bn_dbeta_2 = decay * layer.bn_dbeta_2 + (1 - decay) * layer.bn_dbeta * layer.bn_dbeta
                layer.bn_beta -= lr * layer.bn_dbeta / (np.sqrt(layer.bn_dbeta_2) + 1e-7)

            # update weights
            layer.dW_2 = decay * layer.dW_2 + (1 - decay) * dW * dW
            layer.W -= lr * dW / (np.sqrt(layer.dW_2) + 1e-7)

            # update biases
            layer.dB_2 = decay * layer.dB_2 + (1 - decay) * layer.dB * layer.dB
            layer.B -= lr * layer.dB / (np.sqrt(layer.dB_2) + 1e-7)


class NN:
    """
    neural network class
    """

    def __init__(self, loss, optimizer, regularizer=None, reg_lambda=0.01):
        self.layers = []
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.reg_lambda = reg_lambda

    def add(self, layer):
        """
        adds a layer to the network
        :param layer:
        :return:
        """
        self.layers.append(layer)

    def forward(self, input_x, training=True):
        """
        netwok forward pass
        :param input_x: input vector
        :param training: flag to check if training or testing
        :return: network output
        """
        self.layers[0].input = input_x
        dropout_rate = self.layers[0].dropout_rate
        x = np.array(input_x)

        # apply dropout on network's input vector only
        if training and dropout_rate > 0:
            x *= np.random.binomial(1, (1 - dropout_rate), input_x.shape[1]) / (1 - dropout_rate)

        # forward pass through whole network
        for layer in self.layers[1:]:
            layer.input = x
            output = layer.forward(training)
            x = output
        return output

    def backward(self, delta):
        """
        back-propagation
        :param delta: network's loss gradient
        """

        # backward pass through whole network
        dLdY = delta
        for layer in reversed(self.layers[1:]):
            dLdY = layer.backward(dLdY)

    def predict(self, x):
        """
        predict x using current network state
        :param x: input variable
        :return: predicted output
        """
        x_ = np.asarray(x)
        return self.forward(x_, training=False)

    def fit(self,
            x,
            y,
            learning_rate=0.1,
            gamma=0,
            decay=0,
            epochs=100,
            batch_size=1,
            print_freq=1,
            validation=None):
        """
        train neural network
        :param x: input variable
        :param y: true labels
        :param learning_rate: learning rate
        :param gamma: momentum for SGD
        :param decay: learning rate decay for AdaGrad and RMSProp
        :param epochs: number of epochs
        :param batch_size: batch size
        :param print_freq: print frequency of output
        :param validation: validation set to evaluate on each epoch
        :return: dictionary of results
        results contain:
        training loss, training accuracy [, validation loss, validation accuracy, best model object and best epoch]
        """

        x_in = np.array(x)
        y_in = np.array(y)
        if validation:
            assert len(validation) == 2, 'validation must contain 2 arrays'
            val_x, val_y = validation

        losses = []
        accuracy = []
        val_losses = []
        val_accuracy = []
        max_val = 0
        best_model = copy.deepcopy(self)
        best_epoch = 0
        bar_counter = 0

        for k in range(epochs):

            # shuffle training data each epoch
            indices = np.array(range(x_in.shape[0]))
            indices = np.random.choice(indices, size=indices.shape)
            x_ = x_in[indices]
            y_ = y_in[indices]
            loss = []

            # set progress bar size
            bar_max = x_in.shape[0] * print_freq

            for iter_ in range(0, x_.shape[0], batch_size):

                # slice input for one mini-batch
                x_batch = x_[iter_: iter_ + batch_size]
                y_batch = y_[iter_: iter_ + batch_size]
                if y_batch.ndim == 1:
                    y_batch = y_batch.reshape((y_batch.shape[0], 1))

                # forward pass
                y_hat = self.forward(x_batch)

                # calculate loss
                loss_, delta = self.loss(y_batch,
                                         y_hat)

                # construct loss vector
                loss.append(loss_)

                # backward pass
                self.backward(delta)

                # update weights and biases
                self.optimizer(learning_rate,
                               self.layers,
                               self.regularizer,
                               reg_lambda=self.reg_lambda,
                               gamma=gamma,
                               decay=decay)

                # update progress bar
                bar_counter += batch_size
                progress(min(bar_counter, bar_max), bar_max)

            # calculate training loss and accuracy
            y_pred = self.predict(x_)
            accuracy.append(Metrics.accuracy(y_, y_pred))
            losses.append(sum(loss) / len(loss))

            # calculate validation loss and accuracy
            if validation:
                y_pred_val = self.predict(val_x)
                val_accuracy.append(Metrics.accuracy(val_y, y_pred_val))
                if val_accuracy[-1] > max_val:
                    best_model = copy.deepcopy(self)
                    best_epoch = len(val_accuracy)
                    max_val = val_accuracy[-1]
                val_loss, _ = self.loss(val_y, y_pred_val)
                val_losses.append(float(val_loss))

            # show progress bar
            if k % print_freq == 0 and (k != 0 or print_freq == 1):
                bar_counter = 0
                print()
                header = "Epoch " + str(k + 1) + '/' + str(epochs)
                header += ' training loss = ' + str(losses[k])[:7] +\
                          ', training accuracy = ' + str(accuracy[k]*100)[:8] + '%   '
                if validation:
                    header += 'validation loss = ' + str(val_losses[-1])[:7] +\
                              ', validation accuracy = ' + str(val_accuracy[-1]*100)[:8] + '%   '
                print(header)
                if k < epochs - 1:
                    progress(min(bar_counter, bar_max), bar_max)

        print()
        print()

        # output results in a dictionary
        result = {'Loss': losses, 'Accuracy': accuracy}
        if validation:
            result['Val Loss'] = val_losses
            result['Val Accuracy'] = val_accuracy
            result['Best model'] = best_model
            result['Best epoch'] = best_epoch

        return result

    def fit_lr_schedule(self,
                        x,
                        y,
                        schedule=[0.1],
                        gamma=0,
                        decay=0,
                        epochs=[100],
                        batch_size=1,
                        print_freq=1,
                        validation=None):
        """
        train neural network using learning rate schedule
        :param x: input variable
        :param y: true labels
        :param schedule: learning rate schedule
        :param gamma: momentum for SGD
        :param decay: learning rate decay for AdaGrad and RMSProp
        :param epochs: epochs schedule
        :param batch_size: batch size
        :param print_freq: print frequency of output
        :param validation: validation set to evaluate on each epoch
        :return: dictionary of results
        results contain:
        training loss, training accuracy [, validation loss, validation accuracy, best model object and best epoch]
        """

        loss = []
        acc = []
        val_loss = []
        val_acc = []
        return_result = {}
        model = self
        for lr, epoch in zip(schedule, epochs):
            result = model.fit(x,
                               y,
                               lr,
                               gamma=gamma,
                               decay=decay,
                               epochs=epoch,
                               batch_size=batch_size,
                               print_freq=print_freq,
                               validation=validation)

            loss_r, acc_r = result['Loss'], result['Accuracy']
            last_epoch = result['Best epoch']
            model = result['Best model']
            print('Best epoch ', last_epoch)
            loss.extend(loss_r[:last_epoch - 1])
            acc.extend(acc_r[:last_epoch - 1])
            if validation:
                val_loss_r, val_acc_r = result['Val Loss'], result['Val Accuracy']
                val_loss.extend(val_loss_r[:last_epoch - 1])
                val_acc.extend(val_acc_r[:last_epoch - 1])
        return_result['Loss'] = loss
        return_result['Accuracy'] = acc
        if validation:
            return_result['Val Loss'] = val_loss
            return_result['Val Accuracy'] = val_acc
        return_result['Best model'] = model
        return return_result

