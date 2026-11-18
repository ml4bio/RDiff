"""RDiff_seq command-line entry point."""

import run_lib
from absl import app, flags
from ml_collections.config_flags import config_flags
import wandb
import os


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Inference configuration.', lock_config=True
)
flags.DEFINE_string('workdir', None, 'Work directory.')
flags.DEFINE_enum('mode', 'inf_dir', ['inf_dir'], 'Running mode.')
flags.DEFINE_string('inf_folder', 'inference', 'The directory or .pdb file path for inference')
flags.DEFINE_boolean('deterministic', True, 'Set random seed for reproducibility.')
flags.DEFINE_boolean('no_wandb', False, 'Disable logging to wandb.')
flags.DEFINE_string('exp_name', 'default', 'Experiment name.')
flags.DEFINE_string('proj_name', 'RNA_Inverse', 'Project name for wandb.')
flags.mark_flags_as_required(['config'])


def main(argv):
    # Set random seed
    if FLAGS.deterministic:
        run_lib.set_random_seed(FLAGS.config)

    if FLAGS.workdir is None:
        # use the current path as workdir
        FLAGS.workdir = os.path.dirname(os.path.abspath(__file__))

    # Create the working directory
    if not os.path.exists(FLAGS.workdir):
        os.makedirs(FLAGS.workdir)
    if not os.path.exists(os.path.join(FLAGS.workdir, 'wandb')):
        os.makedirs(os.path.join(FLAGS.workdir, 'wandb'))

    wandb.init(
        project=FLAGS.proj_name,
        name=FLAGS.exp_name,
        config=dict(FLAGS.config),
        mode='disabled' if FLAGS.no_wandb else 'online',
        dir=FLAGS.workdir,
    )

    run_lib.vpsde_inference_dir(FLAGS.config, FLAGS.workdir, FLAGS.inf_folder)


if __name__ == '__main__':
    app.run(main)
