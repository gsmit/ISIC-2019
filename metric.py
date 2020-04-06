import tensorflow.keras.backend as K


def balanced_accuracy(num_classes):
    """
    Calculates the mean of the per-class accuracies.
    Same as sklearn.metrics.balanced_accuracy_score and sklearn.metrics.recall_score with macro average

    Thanks to: https://github.com/wanghsinwei/isic-2019
    """

    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_pred = K.argmax(y_pred, axis=-1)
        class_acc_total = 0
        seen_classes = 0

        for c in range(num_classes):
            accuracy_mask = K.cast(K.equal(class_id_true, c), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_pred), 'int32') * accuracy_mask
            accuracy_mask_sum = K.sum(accuracy_mask)
            class_acc = K.cast(K.sum(class_acc_tensor) / K.maximum(accuracy_mask_sum, 1), K.floatx())
            class_acc_total += class_acc

            condition = K.equal(accuracy_mask_sum, 0)
            seen_classes = K.switch(condition, seen_classes, seen_classes + 1)

        return class_acc_total / K.cast(seen_classes, K.floatx())

    fn.__name__ = 'balanced_accuracy'
    return fn


def categorical_focal_loss(gamma=2., alpha=.25):
    """ https://github.com/umbertogriffo/focal-loss-keras """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed
