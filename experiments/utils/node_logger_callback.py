import random
import time
import tensorflow as tf


class NodeEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, node_id, min_delay_s: float = 0.0, max_delay_s: float = 0.0):
        super().__init__()
        self.node_id = node_id
        self.min_delay_s = min_delay_s
        self.max_delay_s = max_delay_s
        self.last_delay = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        if self.max_delay_s > 0:
            delay = random.uniform(self.min_delay_s, self.max_delay_s)
            self.last_delay = delay
            print(f"\n[Nó {self.node_id}] ▶ Iniciando epoch {epoch + 1} (delay: {delay:.2f}s)")
            time.sleep(delay)
        else:
            self.last_delay = 0.0
            print(f"\n[Nó {self.node_id}] ▶ Iniciando epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        acc     = logs.get("accuracy", 0)
        loss    = logs.get("loss", 0)
        acc_fed = logs.get("accuracy_fed", "N/A")
        print(f"[Nó {self.node_id}] ✔ Epoch {epoch + 1} concluída — loss: {loss:.4f} | acc: {acc:.4f} | acc_fed: {acc_fed}")
        if logs is not None:
            logs["epoch_delay_s"] = self.last_delay  