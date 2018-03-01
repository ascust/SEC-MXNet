import mxnet as mx
import numpy as np

class SEC_expand_loss(mx.metric.EvalMetric):
    def __init__(self):
        super(SEC_expand_loss, self).__init__("SEC_expand_loss")

    def update(self, labels, preds):
        self.num_inst += 1
        self.sum_metric += preds[2].asnumpy()[0]

class SEC_seed_loss(mx.metric.EvalMetric):
    def __init__(self):
        super(SEC_seed_loss, self).__init__("SEC_seed_loss")
    def update(self, labels, preds):
        self.num_inst += 1
        self.sum_metric += preds[0].asnumpy()[0]

class SEC_constrain_loss(mx.metric.EvalMetric):
    def __init__(self):
        super(SEC_constrain_loss, self).__init__("SEC_constrain_loss")
    def update(self, labels, preds):
        self.num_inst += 1
        self.sum_metric += preds[1].asnumpy()[0]

class L2Loss(mx.metric.EvalMetric):
    def __init__(self):
        super(L2Loss, self).__init__('L2Loss')

    def update(self, labels, preds):
        labels = labels[0].asnumpy()
        preds = preds[0].asnumpy()
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        self.num_inst += labels.shape[0]
        res = np.sum((labels - preds) * (labels - preds))
        self.sum_metric += res

class MultiLogisticLoss(mx.metric.EvalMetric):
    def __init__(self, l_index=0, p_index=0):
        self.epsilon = 1e-20
        self.l_index = l_index
        self.p_index = p_index
        super(MultiLogisticLoss, self).__init__('MultiLogisticLoss')

    def update(self, labels, preds):
        labels = labels[self.l_index].asnumpy()
        preds = preds[self.p_index].asnumpy()
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        self.num_inst += labels.shape[0]
        res = 0
        pred_l1 = preds[labels == 1]
        pred_l1[pred_l1 <= self.epsilon] = self.epsilon
        pred_l2 = 1 - preds[labels == 0]
        pred_l2[pred_l2 <= self.epsilon] = self.epsilon

        res += -np.log(pred_l1).sum()
        res += -np.log(pred_l2).sum()
        self.sum_metric += res

class Loss(mx.metric.EvalMetric):
    """Calculate loss"""

    def __init__(self):
        super(Loss, self).__init__('loss')

    def update(self, labels, preds):
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()


        pred = pred.reshape(pred.shape[0],pred.shape[1], -1)
        label = label.astype(np.int32)
        valid_index = label != 255
        prob = np.swapaxes(pred, 0, 1)
        prob = prob[:, valid_index]
        label = label[valid_index]

        loss = np.sum(-np.log(prob[label, np.arange(len(label))]))
        self.sum_metric += loss
        self.num_inst += valid_index.sum()


class Accuracy(mx.metric.EvalMetric):
    """Calculate accuracy"""

    def __init__(self):
        super(Accuracy, self).__init__('accuracy')

    def update(self, labels, preds):
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        pred = pred.argmax(1)

        pred = pred.astype(np.int32).reshape(pred.shape[0], -1)
        label = label.astype(np.int32)
        valid_index = label != 255
        self.sum_metric += (label[valid_index] == pred[valid_index]).sum()
        self.num_inst += valid_index.sum()


class IOU(object):
    def __init__(self, class_num, class_names, ignored_label=255):
        self.ignored_label = ignored_label
        self.class_num = class_num
        self.class_names = class_names
        assert len(class_names) == class_num
        self.conf_mat = None
        self.reset()

    def reset(self):
        self.conf_mat = np.zeros((self.class_num, self.class_num), dtype=np.ulonglong)

    def update(self, label, pred_label):
        label = label.reshape(1, -1)
        pred_label = pred_label.reshape(1, -1)
        self.__eval_pair(pred_label, label)

    def __eval_pair(self, pred_label, label):
        valid_index = label.flat != self.ignored_label
        gt = np.extract(valid_index, label.flat)
        p = np.extract(valid_index, pred_label.flat)
        temp = np.ravel_multi_index(np.array([gt, p]), (self.conf_mat.shape))
        temp_mat = np.bincount(temp, minlength=np.prod(self.conf_mat.shape)).reshape(self.conf_mat.shape)
        self.conf_mat[:]=self.conf_mat+temp_mat

    def get(self):
        return "iou", np.mean(self.get_scores())

    def get_scores(self):
        scores = []
        for i in range(self.class_num):
            tp = np.longlong(self.conf_mat[i, i])
            gti = np.longlong(self.conf_mat[i, :].sum())
            resi = np.longlong(self.conf_mat[:, i].sum())
            denom = gti+resi-tp
            try:
                res = float(tp)/denom
            except ZeroDivisionError:
                res = 0
            scores.append(res)
        return scores

    def get_class_values(self):
        return zip(self.class_names, self.get_scores())
