import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
from beit3_tools import utils

class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

class RerankHandler(object):
    def __init__(self,) -> None:
        super().__init__()
        self.metric_logger = None

    def train_batch(self, model, query_points, query_image, positive_image, negative_images):
        
        negative_image0 = negative_images[:, 0:1]  #[B, 1, C, H, W]
        negative_image1 = negative_images[:, 1:2]  #[B, 1, C, H, W]
        negative_image2 = negative_images[:, 2:3]  #[B, 1, C, H, W]

        # calculate cross entropy loss
        loss = model(query_points, query_image, positive_image, negative_image0, negative_image1, negative_image2)

        return {
            "loss": loss,
        }
    
class RerankHandler3t1(object):
    def __init__(self,) -> None:
        super().__init__()
        self.metric_logger = None

    def train_batch(self, model, query_points0, query_points1, query_points2, query_image0, query_image1, query_image2, positive_image, negative_images):
        
        negative_image0 = negative_images[:, 0:1]  #[B, 1, C, H, W]
        negative_image1 = negative_images[:, 1:2]  #[B, 1, C, H, W]
        negative_image2 = negative_images[:, 2:3]  #[B, 1, C, H, W]

        # calculate cross entropy loss
        loss = model(query_points0, query_points1, query_points2, query_image0, query_image1, query_image2, positive_image, negative_image0, negative_image1, negative_image2)

        return {
            "loss": loss,
        }

class ClassifierHandler(object):
    def __init__(self,) -> None:
        super().__init__()
        self.metric_logger = None

    def train_batch(self, model, features, labels):
        
        # calculate cross entropy loss
        loss = model(features, labels)

        return {
            "loss": loss,
        }

def get_handler(args):
    if args.task == "rerank":
        return RerankHandler()
    if args.task == "rerank3t1":
        return RerankHandler3t1()
    if args.task == "classifier":
        return ClassifierHandler()
    
def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: TaskHandler, epoch: int, start_steps: int,
        lr_schedule_values: list, loss_scaler, max_norm: float = 0,
        update_freq: int = 1, model_ema: Optional[ModelEma] = None,
        log_writer: Optional[utils.TensorboardLogger] = None,
        task=None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            if not isinstance(data[tensor_key], list):
                # data[tensor_key] = [t.to(device, non_blocking=True) for t in data[tensor_key]]
                data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and 'image' in tensor_key:
                data[tensor_key] = data[tensor_key].half()

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.amp.autocast("cuda"):
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value,
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value,
                "lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay_value,
                "grad_norm": grad_norm,
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
