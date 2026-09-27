import torch

from mollift_generator.training.ema import EMAModel


def test_load_state_dict_uses_current_model_device_and_dtype():
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    ema = EMAModel(model)
    checkpoint_state = {
        key: value.detach().cpu().float().clone() if torch.is_floating_point(value) else value.cpu().clone()
        for key, value in ema.state_dict().items()
    }
    ema.load_state_dict(checkpoint_state)
    for key, value in ema.shadow.items():
        assert value.device == model.state_dict()[key].device
        assert value.dtype == model.state_dict()[key].dtype
    ema.update(model)
