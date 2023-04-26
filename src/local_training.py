import torch, logging, tqdm
import numpy as np
from itertools import chain
from typing import Any, Dict, List
import wandb

from .utils import trainer_utils as utils
from .utils import distributed_utils


logger = logging.getLogger(__name__)


def _all_gather_list_sync(args, logging_outputs: List[Dict[str, Any]], ignore=False):
    """
    Sync logging outputs across workers. all_gather_list_sync is
    suifeature when logging outputs are complex types.
    """
    if ignore:
        logging_outputs = []
    results = list(
        zip(
            *distributed_utils.all_gather_list(
                [logging_outputs],
                max_size=getattr(args, "all_gather_list_size", 1048576),
            )
        )
    )
    logging_outputs = results[0]
    logging_outputs = list(chain.from_iterable(logging_outputs))
    return logging_outputs

def train_naive(
    args,
    local_model,
    optimizer,
    criterion,
    n_epoch,
    data_loader,
    metric,
    data_name,
    is_master,
    world_size,
):
    if is_master:
        logger.info("[Epoch] {}".format(n_epoch))
    for iter, sample in tqdm.tqdm(enumerate(data_loader)):
        optimizer.zero_grad(set_to_none=True)
        output = local_model(**sample["net_input"])
        target = local_model.module.get_targets(sample)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        preds = torch.sigmoid(output["pred_output"]).view(-1).detach()
        truths = target.view(-1)

        logging_outputs = {
            "loss": float(loss.detach().cpu()),
            "preds": preds,
            "truths": truths,
        }

        if world_size > 1:
            _logging_outputs = _all_gather_list_sync(args, [logging_outputs])

            for key in logging_outputs.keys():
                if key == "loss":
                    logging_outputs[key] = float(
                        sum(log[key] for log in _logging_outputs)
                    )
                elif key in ["preds", "truths"]:
                    logging_outputs[key] = np.concatenate(
                        [log[key].numpy() for log in _logging_outputs]
                    )
                else:
                    raise NotImplementedError("What else?")
            del _logging_outputs
        else:
            logging_outputs["preds"] = logging_outputs["preds"].cpu().numpy()
            logging_outputs["truths"] = logging_outputs["truths"].cpu().numpy()

        metric(**logging_outputs)  # iter_update

    with torch.no_grad():
        train_metric_dict = metric.get_epoch_dict(len(data_loader))

    log_dict = utils.log_from_dict(
        train_metric_dict, "train", data_name, n_epoch, is_master
    )

    if is_master and args.debug == False:
        wandb.log(log_dict, commit=False)

    return log_dict


def inference(
    args,
    model,
    data_loader,
    data_type,
    data_name,
    n_epoch,
    criterion,
    metric,
    is_master,
    world_size,
):
    model.eval()
    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(data_loader)):

            output = model(**sample["net_input"])
            target = model.module.get_targets(sample)
            loss = criterion(output, target)

            preds = torch.sigmoid(output["pred_output"]).view(-1).detach()
            truths = target.view(-1)

            logging_outputs = {
                "loss": float(loss.detach().cpu()),
                "preds": preds,
                "truths": truths,
            }
            if world_size > 1:
                _logging_outputs = _all_gather_list_sync(args, [logging_outputs])

                for key in logging_outputs.keys():
                    if key == "loss":
                        logging_outputs[key] = float(
                            sum(log[key] for log in _logging_outputs)
                        )
                    elif key in ["preds", "truths"]:
                        logging_outputs[key] = np.concatenate(
                            [log[key].numpy() for log in _logging_outputs]
                        )
                    else:
                        raise NotImplementedError("What else?")
                del _logging_outputs
            else:
                logging_outputs["preds"] = logging_outputs["preds"].cpu().numpy()
                logging_outputs["truths"] = logging_outputs["truths"].cpu().numpy()

            metric(**logging_outputs)  # iter_uddate

        metric_dict = metric.get_epoch_dict(len(data_loader))

    log_dict = utils.log_from_dict(
        metric_dict, data_type, data_name, n_epoch, is_master
    )

    if is_master and args.debug == False:
        wandb.log(log_dict, commit=False)

    return metric_dict
