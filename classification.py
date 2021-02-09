# Lint as: python3
from absl import app
from absl import flags
import numpy as np
import h5py
from os import path
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "path to the saved features")
flags.DEFINE_enum("feature_type",
                  "3d_pointcaps_net",
                  ["3d_pointcaps_net", "pointnet", "caca"],
                  "type of the model that predicts the features.")
flags.DEFINE_enum("method_type",
                  "svm",
                  ["svm", "equal_kmeans"],
                  "type of method used for classification.")
flags.DEFINE_bool("use_kpts",
                  True,
                  "use keypoints in features if true.")


def load_3d_pointcaps_net_features():
  train_data = h5py.File(path.join(
      FLAGS.data_dir, "latent_caps", "saved_train_wo_part_label.h5"))
  test_data = h5py.File(path.join(
      FLAGS.data_dir, "latent_caps", "saved_test_wo_part_label.h5"))
  train_feat = train_data["data"][:]
  train_gt = train_data["cls_label"][:]
  test_feat = test_data["data"][:]
  test_gt = test_data["cls_label"][:]
  train_feat = train_feat.reshape([train_feat.shape[0], -1])
  test_feat = test_feat.reshape([test_feat.shape[0], -1])
  return train_feat, train_gt, test_feat, test_gt


def normalize(kpts):
  max_bound = kpts.max(axis=1, keepdims=True)
  min_bound = kpts.min(axis=1, keepdims=True)
  center = (max_bound + min_bound) * 0.5
  kpts -= center
  max_bound = kpts.max(axis=(1, 2), keepdims=True)
  min_bound = kpts.min(axis=(1, 2), keepdims=True)
  scale = max_bound - min_bound
  kpts /= np.maximum(scale, 1e-7)
  return kpts


def load_pointnet_features():
  with h5py.File(path.join(FLAGS.data_dir, "feat_valid.h5"), "r") as f:
    test_feat = f["feat"][:]
    test_gt = f["label"][:]
    if FLAGS.feature_type == "caca" and FLAGS.use_kpts:
      test_kpts = np.transpose(f["kps"][:], [0, 2, 1])
      test_kpts = normalize(test_kpts)
  with h5py.File(path.join(FLAGS.data_dir, "feat_train.h5"), "r") as f:
    train_feat = f["feat"][:]
    train_gt = f["label"][:]
    if FLAGS.feature_type == "caca" and FLAGS.use_kpts:
      train_kpts = np.transpose(f["kps"][:], [0, 2, 1])
      train_kpts = normalize(train_kpts)
  train_gt = train_gt.reshape([-1]).astype(np.uint8)
  test_gt = test_gt.reshape([-1]).astype(np.uint8)
  train_feat = train_feat.reshape([train_feat.shape[0], -1])
  test_feat = test_feat.reshape([test_feat.shape[0], -1])
  if FLAGS.feature_type == "caca" and FLAGS.use_kpts:
    train_kpts = train_kpts.reshape([train_kpts.shape[0], -1])
    train_feat = np.concatenate([train_feat, train_kpts], axis=-1)
    test_kpts = test_kpts.reshape([test_kpts.shape[0], -1])
    test_feat = np.concatenate([test_feat, test_kpts], axis=-1)
  return train_feat, train_gt, test_feat, test_gt


def linear_svm_classification(train_feat, train_gt, test_feat, test_gt):
  classifier = LinearSVC(verbose=1, C=0.1)
  classifier.fit(train_feat, train_gt.astype(int))
  return classifier.score(test_feat, test_gt.astype(int))


def construct_cost_matrix(preds, gts, n_pred_cls=13, n_gt_cls=13):
  cost = np.zeros([n_pred_cls, n_gt_cls], dtype=np.int32)
  for pred, gt in zip(preds, gts):
    cost[pred, gt] += 1
  return cost


def reassign_labels(preds, assignment):
  return assignment[preds]


def equal_kmeans_classification(train_feat, train_gt, test_feat, test_gt):
  cluster = KMeans(n_clusters=13, verbose=1)
  cluster.fit(train_feat)
  train_preds = cluster.labels_
  test_preds = cluster.predict(test_feat)
  c = construct_cost_matrix(train_preds, train_gt)
  row_ind, col_ind = linear_sum_assignment(c, maximize=True)
  test_preds = reassign_labels(test_preds, col_ind)
  return (test_preds == test_gt).sum() * 1. / test_preds.shape[0]


def main(unused_args):
  if FLAGS.data_dir is None:
    raise ValueError("data_dir needs to be specified, {} given.".format(
        FLAGS.data_dir
    ))

  if FLAGS.feature_type == "3d_pointcaps_net":
    train_feat, train_gt, test_feat, test_gt = load_3d_pointcaps_net_features()
  elif FLAGS.feature_type == "pointnet" or FLAGS.feature_type == "caca":
    train_feat, train_gt, test_feat, test_gt = load_pointnet_features()

  if FLAGS.method_type == "svm":
    accuracy = linear_svm_classification(
        train_feat, train_gt, test_feat, test_gt)
  elif FLAGS.method_type == "equal_kmeans":
    accuracy = equal_kmeans_classification(
        train_feat, train_gt, test_feat, test_gt)

  print("{} feature on {}: {}".format(
      FLAGS.feature_type, FLAGS.method_type, accuracy))


if __name__ == '__main__':
  app.run(main)
