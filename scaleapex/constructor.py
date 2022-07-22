import types
from apex.fp16_utils import master_params_to_model_params
from apex.multi_tensor_apply import multi_tensor_applier
import torch


class AmpOptimizerState(object):
    def __init__(self):
        pass

def _master_params_to_model_params(self):
    stash = self._amp_stash
    if multi_tensor_applier.available:
        if len(stash.all_fp16_params) > 0:
            multi_tensor_applier(
                stash.multi_tensor_scale,
                stash.dummy_overflow_buf,
                [stash.all_fp32_from_fp16_params, stash.all_fp16_params],
                1.0)
    else:
        for fp16_group, fp32_from_fp16_group in zip(stash.fp16_groups, stash.fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)

def lazy_init_with_master_weights(self):
    stash = self._amp_stash
    stash.fp16_groups = []
    stash.fp32_from_fp16_groups = []
    stash.fp32_from_fp32_groups = []
    for i, param_group in enumerate(self.param_groups):
        # maybe_print("FP16_Optimizer processing param group {}:".format(i))
        fp16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_fp16_params_this_group = []
        for i, param in enumerate(param_group['params']):
            if param.requires_grad:
                if param.type() == 'torch.cuda.HalfTensor':
                    # maybe_print("FP16_Optimizer received torch.cuda.HalfTensor with {}"
                    #             .format(param.size()))
                    fp16_params_this_group.append(param)
                    master_param = param.detach().clone().float()
                    master_param.requires_grad = True
                    param_group['params'][i] = master_param
                    fp32_from_fp16_params_this_group.append(master_param)
                    # Reset existing state dict key to the new master param.
                    # We still need to recast per-param state tensors, if any, to FP32.
                    if param in self.state:
                       self.state[master_param] = self.state.pop(param)
                elif param.type() == 'torch.cuda.FloatTensor':
                    # maybe_print("FP16_Optimizer received torch.cuda.FloatTensor with {}"
                    #             .format(param.size()))
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param
                else:
                    raise TypeError("Optimizer's parameters must be either "
                                    "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                    "Received {}".format(param.type()))

        stash.fp16_groups.append(fp16_params_this_group)
        stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
        stash.fp32_from_fp32_groups.append(fp32_params_this_group)

    stash.all_fp16_params = []
    for group in stash.fp16_groups:
        stash.all_fp16_params += group

    stash.all_fp32_from_fp16_params = []
    for group in stash.fp32_from_fp16_groups:
        stash.all_fp32_from_fp16_params += group

    stash.all_fp32_from_fp32_params = []
    for group in stash.fp32_from_fp32_groups:
        stash.all_fp32_from_fp32_params += group

    # stash.all_fp32_from_fp16_grad_stash = [None for _ in stash.all_fp32_from_fp16_params]
    stash.all_fp32_from_fp32_grad_stash = [None for _ in stash.all_fp32_from_fp32_params]

    for param in stash.all_fp32_from_fp16_params:
        param.grad = None

    #for param in stash.all_fp32_from_fp32_params:
        #param.grad = None

    # Leverage state_dict() and load_state_dict() to recast preexisting per-param state tensors
    self.load_state_dict(self.state_dict())

def post_backward_models_are_masters(scaler, params, stashed_grads, scale_override=None):
    grads_have_scale, stashed_have_scale, out_scale = scaler.loss_scale(), 1.0, 1.0

    if scale_override is not None:
        grads_have_scale, stashed_have_scale, out_scale = scale_override

    # This is a lot of python overhead...
    grads_needing_unscale = []
    grads_needing_unscale_with_stash = []
    stashed = []
    for param, stashed_grad in zip(params, stashed_grads):
        if param.grad is None and stashed_grad is not None:
            param.grad = stashed_grad
        elif param.grad is not None and stashed_grad is None:
            grads_needing_unscale.append(param.grad)
        elif param.grad is not None and stashed_grad is not None:
            grads_needing_unscale_with_stash.append(param.grad)
            stashed.append(stashed_grad)
        else: # param.grad is None and stashed_grad is None
            continue

    # unscale() implements grads*(1/scale), so "scale" should be grads_have_scale/out_scale.
    if len(grads_needing_unscale) > 0:
        scaler.unscale(
            grads_needing_unscale,
            grads_needing_unscale,
            None, # unused_scale, currently present to avoid API breakage elsewhere
            models_are_masters=True,
            scale_override=grads_have_scale/out_scale)

    if len(grads_needing_unscale_with_stash) > 0:
        scaler.unscale_with_stashed(
            grads_needing_unscale_with_stash,
            stashed,
            grads_needing_unscale_with_stash,
            scale_override=(grads_have_scale, stashed_have_scale, out_scale))

    # Clear the stash.
    for i in range(len(stashed_grads)):
        stashed_grads[i] = None

def has_overflow(scaler):
    if scaler.has_fused_kernel:
        return scaler._overflow_buf.item()
    else:
        return scaler._has_overflow

def sync_has_overflow(scaler):
    if scaler.has_fused_kernel:
        # by defualt, it is sum
        torch.distributed.all_reduce(scaler._overflow_buf)
    else:
        overflow = torch.cuda.IntTensor([scaler._has_overflow])
        torch.distributed.all_reduce(overflow)
        scaler._has_overflow = overflow.item()

def model_params_to_master_params_with_overflow_check(self, scaler):
    stash = self._amp_stash

    _amp_lazy_init(self)

    # This is a lot of python overhead...
    fp16_grads_needing_unscale = []
    new_fp32_grads = []
    fp16_grads_needing_unscale_with_stash = []
    preexisting_fp32_grads = []
    for fp16_param, fp32_param in zip(stash.all_fp16_params,
                                      stash.all_fp32_from_fp16_params):
        if fp16_param.grad is None and fp32_param.grad is not None:
            continue
        elif fp16_param.grad is not None and fp32_param.grad is None:
            fp32_param.grad = torch.empty_like(fp32_param)
            fp16_grads_needing_unscale.append(fp16_param.grad)
            new_fp32_grads.append(fp32_param.grad)
        elif fp16_param.grad is not None and fp32_param.grad is not None:
            fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
            preexisting_fp32_grads.append(fp32_param.grad)
        else: # fp16_param.grad is None and fp32_param.grad is None:
            continue

    if len(fp16_grads_needing_unscale) > 0:
        scaler.unscale(
            fp16_grads_needing_unscale,
            new_fp32_grads,
            scaler.loss_scale(),
            models_are_masters=False)

    if len(fp16_grads_needing_unscale_with_stash) > 0:
        scaler.unscale_with_stashed(
            fp16_grads_needing_unscale_with_stash,
            preexisting_fp32_grads,
            preexisting_fp32_grads)

    # fp32 params can be treated as they would be in the "no_master_weights" case.
    post_backward_models_are_masters(
        scaler,
        stash.all_fp32_from_fp32_params,
        stash.all_fp32_from_fp32_grad_stash)

def _amp_lazy_init(self):
    stash = self._amp_stash

    if not stash.lazy_init_called:
        lazy_init_with_master_weights(self)
        stash.lazy_init_called = True

def zero_fp32_from_fp16_grad(self):
    stash = self._amp_stash
    _amp_lazy_init(self)
    for param in stash.all_fp32_from_fp16_params:
        param.grad = None

def convert2fp16_optimizer(optimizer):
    if hasattr(optimizer, "_amp_stash"):
        raise RuntimeError("A given optimizer should only be passed through amp.initialize once.")
    else:
        optimizer._amp_stash = AmpOptimizerState()

    optimizer._amp_stash.lazy_init_called = False

    # TODO:  Centralize exposure and import error checking for the C backend.
    if multi_tensor_applier.available:
        import amp_C
        optimizer._amp_stash.multi_tensor_scale = amp_C.multi_tensor_scale
        optimizer._amp_stash.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
        optimizer._amp_stash.dummy_overflow_buf = torch.cuda.IntTensor([0]);

    old_step = optimizer.step
    def new_step(self, closure=None, scaler=None):
        if closure is not None:
            raise RuntimeError("Currently, Amp does not support closure use with optimizers.")
        assert scaler is not None
        scaler.clear_overflow_state()
        model_params_to_master_params_with_overflow_check(self, scaler)
        sync_has_overflow(scaler)
        retval = None
        overflow = has_overflow(scaler)
        if not overflow:
            retval = old_step()
            _master_params_to_model_params(self)
        # clear the grad to save memory; no need to call this function at the
        # beginning for the first iteration as it will be initialized as None
        # for .grad field.
        zero_fp32_from_fp16_grad(self)
        return retval, overflow

    optimizer.step = types.MethodType(new_step, optimizer)

    return optimizer

def list_dict_copy(x):
    if isinstance(x, list):
        return [list_dict_copy(y) for y in x]
    elif isinstance(x, dict):
        return dict((k, list_dict_copy(v)) for k, v in x.items())
    else:
        return x

def optim_constructor(optim, parameters, **kwargs):
    parameters = list_dict_copy(parameters)
    optimizer = optim(parameters, **kwargs)
    return convert2fp16_optimizer(optimizer)

def dynamic_scaler():
    from apex.amp.scaler import LossScaler
    ret = LossScaler(loss_scale='dynamic', min_loss_scale=1.)
    def state_dict():
        return {
            '_unskipped': ret._unskipped,
            '_loss_scale': ret._loss_scale
        }
    def load_state_dict(state_dict):
        ret._unskipped = state_dict['_unskipped']
        ret._loss_scale = state_dict['_loss_scale']
        ret.state_dict = state_dict
        ret.load_state_dict = load_state_dict
        return ret
    ret.state_dict = state_dict
    ret.load_state_dict = load_state_dict
    return ret

