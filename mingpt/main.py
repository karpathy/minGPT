import hydra
import torch
from hydra.utils import instantiate

from mingpt.conf import Config
from mingpt.utils import sample
from mingpt.utils import set_seed


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: Config) -> None:
    print(cfg.pretty())
    set_seed(cfg.seed)

    train_dataset = instantiate(cfg.train_dataset)
    test_dataset = (
        # Hydra 1.0.0rc3 bug, should just return None for None config.
        instantiate(cfg.test_dataset)
        if cfg.test_dataset is not None
        else None
    )
    # construct the model
    model = instantiate(cfg.model, vocab_size=train_dataset.vocab_size)

    # construct a trainer
    trainer = instantiate(
        cfg.trainer, model=model, train_dataset=train_dataset, test_dataset=test_dataset
    )
    trainer.train()
    # (... enjoy the show for a while... )
    #
    # sample from the model (the [None, ...] and [0] are to push/pop a needed dummy batch dimension)

    x = torch.tensor([1, 2, 3], dtype=torch.long)[None, ...]  # context conditioning
    y = sample(model, x, steps=30, temperature=1.0, sample=True, top_k=5)[0]
    # our model filled in the integer sequence with 30 additional likely integers
    print(y)


if __name__ == "__main__":
    my_app()
