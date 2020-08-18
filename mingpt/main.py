import logging

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path

from mingpt.conf import Config
from mingpt.model import GPT
from mingpt.utils import sample, set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: Config) -> None:
    log.info(cfg.pretty())
    set_seed(cfg.seed)

    train_dataset = instantiate(cfg.train_dataset)

    # Hydra 1.0.0rc3 bug, should just return None for None config.
    test_dataset = (
        instantiate(cfg.test_dataset) if cfg.test_dataset is not None else None
    )

    # construct the model
    device = torch.device(cfg.device)
    model: GPT = instantiate(cfg.model, vocab_size=train_dataset.vocab_size)
    model.to(device)
    if cfg.load:
        # Hydra is changing the current working directory in every run to facilitate a transparent
        # scratch space for the job.
        # to_absolute_path converts a path to be relative to the original working directory the app was launched from.
        path = to_absolute_path(cfg.load)

        # TODO: checkpoint loading is not working correctly for some reason.
        log.info(f"Loading model from {path}")
        model.load_state_dict(torch.load(path))
        model.eval()

    if cfg.train:
        # construct a trainer
        trainer = instantiate(
            cfg.trainer,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        trainer.train()

    context = cfg.context
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
        None, ...
    ].to(device)
    y = sample(model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
    completion = "".join([train_dataset.itos[int(i)] for i in y])
    log.info(completion)


if __name__ == "__main__":
    my_app()
