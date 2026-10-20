import os
import sys

import numpy as np
import pandas as pd

from hovsg.utils.constants import COCO_STUFF_CLASSES, MATTERPORT_LABELS_160, MATTERPORT_LABELS_40, MATTERPORT_GT_LABELS, OPENVOCAB_MATTERPORT_LABELS
from hovsg.utils.clip_utils import get_text_feats_multiple_templates
from hovsg.utils.encoder_http import is_http


def _http_label_feats(clip_model, classes, file_name):
    """Label features from the encoder service's own text tower.

    The ``.npy`` files shipped in ``hovsg/labels`` are OpenCLIP ViT-H-14 features
    and would silently mislabel every object under any other encoder (even
    another 1024-d one), so they are never read here. The features are computed
    with upstream's templates (``get_text_feats_multiple_templates``) and cached
    per encoder profile, revision and class list (``HttpEncoder.label_feats_path``).
    """
    path = clip_model.label_feats_path(file_name, classes)
    if path is not None and path.is_file():
        text_feats = np.load(path)
        if text_feats.shape == (len(classes), clip_model.dim):
            return text_feats, classes
    text_feats = get_text_feats_multiple_templates(classes, clip_model, clip_model.dim)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}.npy")
            np.save(tmp, text_feats)
            os.replace(tmp, path)
        except OSError as exc:  # a read-only cache only costs a recompute next time
            print(f"label feature cache {path} not written: {exc}")
    return text_feats, classes


def compute_label_feats(clip_model, clip_feat_dim, label_feat_path, classes, pre_computed_feats_path="text_feats.npy"):
    """
    Either load precomputed label features or run clip to obtain those
    """
    if is_http(clip_model):
        return _http_label_feats(clip_model, classes, pre_computed_feats_path)
    # check if the text features are pre-computed
    if os.path.exists(os.path.join(label_feat_path, pre_computed_feats_path)):
        if ".npy" in pre_computed_feats_path:
            text_feats = np.load(os.path.join(label_feat_path, pre_computed_feats_path))
    else:
        # if not, compute, store and return them based on the provided classes and a clip model
        text_feats = get_text_feats_multiple_templates(classes, clip_model, clip_feat_dim)
        np.save(os.path.join(label_feat_path, pre_computed_feats_path), text_feats)
    return text_feats, classes


def get_label_feats(clip_model, clip_feat_dim, obj_labels, label_feat_path=None):
    label_feat_path = os.path.dirname(os.path.abspath(__file__))
    if obj_labels == "COCO_STUFF_CLASSES":
        classes = list(COCO_STUFF_CLASSES.values())
        text_feats, classes = compute_label_feats(clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_COCO_STUFF_CLASSES.npy")
    elif obj_labels == "MATTERPORT_LABELS_160":
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, MATTERPORT_LABELS_160, "text_feats_MATTERPORT_LABELS_160.npy"
        )
    elif obj_labels == "MATTERPORT_LABELS_40":
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, MATTERPORT_LABELS_40, "text_feats_MATTERPORT_LABELS_40.npy"
        )
    elif obj_labels == "MATTERPORT_GT_LABELS":
        classes = list(MATTERPORT_GT_LABELS.values())
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_MATTERPORT_GT_LABELS.npy"
        )
    elif obj_labels == "OPENVOCAB_MATTERPORT_LABELS":
        classes = list()
        for key, val in OPENVOCAB_MATTERPORT_LABELS.items():
            classes.append(key)
            classes.extend(val)
        classes = list(set(classes))
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_OPENVOCAB_MATTERPORT_LABELS.npy"
        )
    elif obj_labels == "HM3DSEM_LABELS":
        # TODO: change this
        label_feat_path = "hovsg/labels"
        classes_matrix = pd.read_csv(os.path.join(label_feat_path, "HM3D_CountsOfObjectTypes.csv"), header=0, sep=";")
        classes = list(classes_matrix[classes_matrix.keys()[0]].values)
        text_feats, classes = compute_label_feats(clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_HM3DSEM_LABELS.npy")
    return text_feats, classes
