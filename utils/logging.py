"""
General logging class for PyTorch training loops.
"""
import os
import torch

class Logger(object):
    """
    Abstracted class to expose logging methods to a series of files on the run directory.
    """
    def __init__(self, run_dir):
        """
        Construct logger and keep files open.
        """
        ### specify directories; create directories if they don't exist:
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.ckpt_dir = os.path.join(run_dir, 'ckpts/')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.log_dir = os.path.join(run_dir, 'logs/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        ### create output logging file and keep open:
        self.training_log = os.path.join(self.log_dir, 'training.log')
        self._training_log_f = open(self.training_log, 'w')
        self.messages_log = os.path.join(self.log_dir, 'messages.log')
        self._messages_log_f = open(self.messages_log, 'w')


    def close(self):
        """
        Close all file handles.
        """
        self._training_log_f.close()
        self._messages_log_f.close()


    def log(self, step, t_loss, v_loss, t_avg, v_avg):
        """
        Log a training loss message to the logfile.
        """
        self._training_log_f.write(
            "Step: {0} | Raw Train NCLL: {1:.4f} | Raw Valid NCLL: {2:.4f} | "
            "Seq Train NCLL: {3:.4f} | Seq Valid NCLL: {4:.4f}\n".format(
                step, t_loss, v_loss, t_avg, v_avg))
        self._training_log_f.flush()


    def save(self, timestep, model):
        """Save model to run directory."""
        _model_path = os.path.join(self.ckpt_dir, "crf.t{}.pt".format(timestep))
        torch.save(model.state_dict(), _model_path)
        self._messages_log_f.write("Saved models to: {}\n".format(_model_path))
        self._messages_log_f.flush()
