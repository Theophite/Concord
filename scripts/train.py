from util.import_util import script_imports

script_imports()

import json

from modules.util import create
from modules.util.args.TrainArgs import TrainArgs
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig


def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    # Concord v2 checkpoint-restart: when scripts/concord_train_restart.py relaunches us after a
    # sample-triggered exit(42), it sets CONCORD_RESUMING so this fresh process resumes from the
    # backup the previous process just wrote -- a clean allocator, so the graph recaptures without
    # the Windows sampling-fragmentation wedge.
    import os
    if os.environ.get("CONCORD_RESUMING"):
        train_config.continue_last_backup = True
        print("[concord-restart] CONCORD_RESUMING set -> resuming from last backup", flush=True)

    try:
        with open("secrets.json" if args.secrets_path is None else args.secrets_path, "r") as f:
            secrets_dict=json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        if args.secrets_path is not None:
            raise

    trainer = create.create_trainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()


if __name__ == '__main__':
    main()
