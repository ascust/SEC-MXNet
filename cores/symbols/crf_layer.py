import mxnet as mx
import numpy as np
from cores.utils.CRF import CRF
class CRFLayer(mx.operator.CustomOp):
    def __init__(self, pos_xy_std, pos_w,
		bi_xy_std, bi_rgb_std, bi_w,
        maxiter, scale_factor, min_prob):
        self.min_prob = float(min_prob)
        self.CRF = CRF(pos_xy_std=int(pos_xy_std),
		        pos_w=int(pos_w), bi_xy_std=int(bi_xy_std), bi_rgb_std=int(bi_rgb_std),
                bi_w=int(bi_w), maxiter=int(maxiter), scale_factor=float(scale_factor))

    def forward(self, is_train, req, in_data, out_data, aux):
        unary = in_data[0].asnumpy()
        small_ims = in_data[1].asnumpy()
        N = unary.shape[0]
        self.result = np.zeros(unary.shape, dtype=np.float32)

        for i in range(N):
            self.result[i] = self.CRF.inference(small_ims[i], unary[i])
            # print np.min(unary[i]), np.max(unary[i]), np.mean(np.abs(unary[i]))
            # print np.any(np.isnan(unary[i])), np.any(np.isnan(self.result[i]))

        self.result[self.result<self.min_prob] = self.min_prob
        self.result = self.result/np.sum(self.result, axis=1, keepdims=True)
        self.assign(out_data[0], req[0], mx.nd.array(np.log(self.result)))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        grad = out_grad[0].asnumpy()
        # grad = (1-self.result)*grad
        grad[:] = 0.0
        self.assign(in_grad[0], req[0], mx.nd.array(grad))

@mx.operator.register("crf_layer")
class CRFLayerProp(mx.operator.CustomOpProp):
    def __init__(self, pos_xy_std=3, pos_w=3,
		bi_xy_std=80, bi_rgb_std=13, bi_w=10,
        maxiter=10, scale_factor=12.0, min_prob=0.0001):
        self.pos_xy_std = pos_xy_std
        self.pos_w = pos_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.bi_w = bi_w
        self.maxiter = maxiter
        self.scale_factor = scale_factor
        self.min_prob = min_prob
        super(CRFLayerProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data', 'small_ims']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        small_im_shape = [data_shape[0], data_shape[2], data_shape[3], 3]
        output_shape = data_shape
        return [data_shape, small_im_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return CRFLayer(pos_xy_std=self.pos_xy_std,
                        pos_w=self.pos_w,
                        bi_rgb_std=self.bi_rgb_std,
                        bi_xy_std = self.bi_xy_std,
                        bi_w=self.bi_w,
                        maxiter=self.maxiter,
                        scale_factor=self.scale_factor,
                        min_prob=self.min_prob)