"""PyTorch Lightning module for insider threat detection with custom loss.

Wraps GCNLSTMInsiderThreat with training, validation, and evaluation logic.
Implements the anomaly-aware loss from the paper: suppress true activity class
for malicious examples, push model to flag anomalies via low confidence.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import pytorch_lightning as pl

from certgnn.model import GCNLSTMInsiderThreat


class InsiderThreatLightning(pl.LightningModule):
    """Lightning module for insider threat detection.

    Loss function:
    - For normal samples: standard cross-entropy on true activity
    - For malicious samples: uniform distribution over all BUT the true activity
      (train model to flag true activity as anomalous)

    Evaluation metric: AUC using model's confidence in true activity as anomaly score
    (low confidence = anomalous, high confidence = normal)
    """

    def __init__(
        self,
        num_node_features: int = 54,
        gcn_hidden_dim: int = 16,
        lstm_hidden_dim: int = 32,
        num_activity_classes: int = 192,
        lstm_num_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_type: str = "standard",  # "standard" (from paper) or "anomaly_aware" (custom)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GCNLSTMInsiderThreat(
            num_node_features=num_node_features,
            gcn_hidden_dim=gcn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_activity_classes=num_activity_classes,
            lstm_num_layers=lstm_num_layers,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_activity_classes = num_activity_classes
        self.loss_type = loss_type  # "standard" or "anomaly_aware"

        # For collecting outputs during val/test
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

    def forward(self, batch_data):
        """Forward pass."""
        return self.model(batch_data)

    def _anomaly_aware_loss(self, logits, y_act, y_label):
        """Custom loss that handles malicious examples specially.

        For normal (y_label=0): standard CE on true activity
        For malicious (y_label=1): target is uniform over all activities except true one
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        # Validate that y_act indices are within bounds
        max_activity = y_act.max().item()
        if max_activity >= num_classes:
            raise RuntimeError(
                f"Activity index {max_activity} >= num_classes {num_classes}. "
                f"This likely means metadata has {num_classes} classes but data has indices up to {max_activity}. "
                f"Check: metadata['num_classes'] = {num_classes}, max(y_act) = {max_activity}"
            )

        # Get softmax probabilities
        probs = F.softmax(logits, dim=1)  # [batch, num_classes]

        # Initialize target distribution
        target = probs.clone()  # [batch, num_classes]

        # One-hot encode true activity classes
        true_class_onehot = F.one_hot(y_act, num_classes).bool()  # [batch, num_classes]

        # Malicious mask
        is_malicious = (y_label > 0).unsqueeze(1)  # [batch, 1]

        # For malicious samples: zero out the true class
        target[is_malicious.squeeze(1)] = 0.0

        # Create uniform distribution over remaining classes
        # (for malicious: uniform over num_classes-1; for normal: keep original)
        uniform = torch.ones_like(probs) / (num_classes - 1)
        uniform[true_class_onehot] = 0.0  # don't count the true class in uniform

        # Replace malicious targets with uniform
        target[is_malicious.squeeze(1)] = uniform[is_malicious.squeeze(1)]

        # Normalize if needed
        row_sums = target.sum(dim=1, keepdim=True)
        target = torch.where(row_sums > 0, target / row_sums, target)

        # Cross-entropy: -sum(target * log(probs))
        log_probs = torch.log(probs + 1e-10)
        loss = -(target * log_probs).sum(dim=1).mean()

        return loss

    def _standard_loss(self, logits, y_act):
        """Standard cross-entropy loss from the paper.

        This is the original loss used in the paper:
        L = -∑ log(P(y_act | logits))
        """
        return F.cross_entropy(logits, y_act)

    def training_step(self, batch, batch_idx):
        """Training step on a batch."""
        logits = self.forward(batch)

        if self.loss_type == "standard":
            loss = self._standard_loss(logits, batch.y_act)
        elif self.loss_type == "anomaly_aware":
            loss = self._anomaly_aware_loss(logits, batch.y_act, batch.y_label)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        batch_size = batch.y_act.shape[0]
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step (collect outputs for AUC calculation)."""
        logits = self.forward(batch)
        loss = self._anomaly_aware_loss(logits, batch.y_act, batch.y_label)

        # Get probabilities and scores
        probs = F.softmax(logits, dim=1)  # [batch_size, num_activity_classes]
        scores = probs.gather(1, batch.y_act.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Collect for epoch-end AUC
        self.val_preds.append(scores.detach().cpu())
        self.val_labels.append(batch.y_label.detach().cpu())

        batch_size = batch.y_act.shape[0]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_end(self):
        """Calculate AUC at end of validation epoch."""
        if len(self.val_preds) == 0:
            return

        preds = torch.cat(self.val_preds).numpy()
        labels = torch.cat(self.val_labels).numpy()

        # Anomaly score: LOW confidence in true activity = anomalous
        # So we need to invert: anomaly_score = 1 - probability
        anomaly_scores = 1 - preds

        try:
            auc = roc_auc_score(labels, anomaly_scores)
            self.log("val_auc", auc, on_epoch=True, prog_bar=True, batch_size=len(labels))
        except ValueError:
            # Happens if only one class in batch
            pass

        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        """Test step (same as validation)."""
        logits = self.forward(batch)

        probs = F.softmax(logits, dim=1)
        scores = probs.gather(1, batch.y_act.unsqueeze(1)).squeeze(1)

        self.test_preds.append(scores.detach().cpu())
        self.test_labels.append(batch.y_label.detach().cpu())

    def on_test_epoch_end(self):
        """Calculate final AUC on test set."""
        if len(self.test_preds) == 0:
            return

        preds = torch.cat(self.test_preds).numpy()
        labels = torch.cat(self.test_labels).numpy()

        anomaly_scores = 1 - preds

        try:
            auc = roc_auc_score(labels, anomaly_scores)
            fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)

            # Find optimal threshold at target FPR ~16% (as in paper)
            target_fpr = 0.16
            idx = np.argmin(np.abs(fpr - target_fpr))
            optimal_threshold = thresholds[idx]
            optimal_tpr = tpr[idx]
            optimal_fpr = fpr[idx]

            self.log("test_auc", auc, batch_size=len(labels))
            self.log("test_tpr_at_16fpr", float(optimal_tpr), batch_size=len(labels))
            self.log("test_fpr_at_threshold", float(optimal_fpr), batch_size=len(labels))
        except ValueError:
            pass

        self.test_preds.clear()
        self.test_labels.clear()

    def on_train_epoch_end(self) -> None:
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# Import numpy for threshold calculation
import numpy as np
