
import numpy as np

class Reaching:
    def __init__(self, task_params):
        """
        Initialize a Reaching task
        Args:
            task_params: dict with keys:
                'trial_len' : int (total trial length in time steps)
                'onset'     : (min_s, max_s)
                'stim_dur'  : (min_s, max_s)
                'delay_dur' : (min_s, max_s)
                'n_stim'    : number of unique stimuli
                'batch_size': batch size for get_batch
        """

        self.task_params = task_params
        self.phases = np.linspace(0, 2 * np.pi, task_params['n_stim'], endpoint=False)



    def get_batch(self):
        """
        Generate a batch of trials with one-hot encoded stimuli.
        Returns:
            inputs:  (B, T, n_inputs)
            targets: (B, T, n_classes)
            mask:    (B, T, 1)
        """
        task_params = self.task_params
        phases = self.phases
        batch_size = task_params['batch_size']

        # Convert durations to samples
        T = task_params['trial_len']
        n_classes = len(phases)

        # Sample parameters per trial
        phase_idx = np.random.choice(n_classes, size=batch_size, replace=True)
        onset = np.random.randint(task_params['onset'][0], task_params['onset'][1], size=batch_size)
        stim_dur = np.random.randint(task_params['stim_dur'][0], task_params['stim_dur'][1], size=batch_size)
        delay_dur = np.random.randint(task_params['delay_dur'][0], task_params['delay_dur'][1], size=batch_size)
        delay_end = onset + stim_dur + delay_dur

        # One-hot stimulus vectors
        phase_onehot = np.eye(n_classes)[phase_idx]

        # Define channel count (stimulus one-hot + 1 for go cue)
        n_inputs = n_classes + 1

        # Allocate arrays
        inputs  = np.zeros((batch_size, T, n_inputs), dtype=np.float32)
        targets = np.zeros((batch_size, T, n_classes), dtype=np.float32)
        mask    = np.zeros((batch_size, T, 1), dtype=np.float32)

        for b in range(batch_size):
            # Stimulus presentation
            inputs[b, onset[b]:onset[b]+stim_dur[b], :n_classes] = phase_onehot[b]

            # Go cue after delay
            inputs[b, delay_end[b]:, -1] = 1.0

            # Target is the class identity after go cue
            targets[b, delay_end[b]:, :] = phase_onehot[b]

            # Mask active only after stimulus + delay
            mask[b, onset[b]+stim_dur[b]:, 0] = 1.0

        return inputs, targets, mask

            



            

