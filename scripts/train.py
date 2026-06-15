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
    # The GUI / restart wrapper deliver CTRL_BREAK (SIGBREAK) to stop us. Python catches Ctrl+C
    # (SIGINT) by default but NOT Ctrl+Break, so without this the OS hard-terminates the process
    # (STATUS_CONTROL_C_EXIT) and the graceful-save path below never runs. Turn SIGBREAK into a
    # KeyboardInterrupt so trainer.end() (final save / backup_before_save) gets to run.
    import signal as _signal

    if hasattr(_signal, "SIGBREAK"):
        def _on_break(_sig, _frame):
            raise KeyboardInterrupt()
        _signal.signal(_signal.SIGBREAK, _on_break)

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
        # Reuse the existing latent cache on resume. With clear_cache_before_training=True the trainer
        # wipes + re-encodes the WHOLE dataset (minutes) at every restart -- that re-cache, not the
        # graph, is the slow resume. The cache is valid (same dataset), so skip it on resume.
        train_config.clear_cache_before_training = False
        print("[concord-restart] CONCORD_RESUMING set -> resume from backup, reuse latent cache",
              flush=True)

    # GUI command bridge: TrainUI's restart-wrapper path runs us in a separate process, so the GUI
    # can't reach this in-process TrainCommands. It writes a request to CONCORD_GUI_CMD_FILE instead;
    # this watcher maps it to the matching command so "Sample now" / "Backup now" work. A "sample"
    # request carries the GUI's currently-selected sample-definition file on a second line: we point
    # the config at it and clear the launch-time inlined samples (to_pack_dict bakes the list in) so
    # the WHOLE selected queue is re-read fresh -- honoring the sampling-tab selector and any edits --
    # then the wrapper recycles the process exactly as it does for a timed sample.
    import os as _os
    _gui_cmd_file = _os.environ.get("CONCORD_GUI_CMD_FILE")
    if _gui_cmd_file:
        import threading as _threading
        import time as _time

        def _watch_gui_commands():
            while True:
                _time.sleep(1.0)
                try:
                    if not _os.path.exists(_gui_cmd_file):
                        continue
                    with open(_gui_cmd_file, "r", encoding="utf-8") as _f:
                        _lines = _f.read().splitlines()
                    _os.remove(_gui_cmd_file)
                    _req = _lines[0].strip() if _lines else ""
                    _arg = _lines[1].strip() if len(_lines) > 1 else ""
                    if _req == "sample":
                        if _arg:
                            train_config.sample_definition_file_name = _arg
                            train_config.samples = None
                        commands.sample_default()
                    elif _req == "backup":
                        commands.backup()
                except Exception:
                    pass

        _threading.Thread(target=_watch_gui_commands, daemon=True).start()

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
