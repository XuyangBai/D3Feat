# loss.py ---
#
# Filename: loss.py
# Description: Code is the courtesy of Alexander Hermans and is based on the
#              https://github.com/VisualComputingInstitute/triplet-reid paper. If you use it please consider citing.
# Author: Alexander Hermans
# Project: In Defense of the Triplet Loss for Person Re-Identification
# Version: 1.0

# Code:

import numbers
import numpy as np
import tensorflow as tf


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))


cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]


def get_at_indices(tensor, indices):
    """ Like `tensor[np.arange(len(tensor)), indices]` in numpy. """
    counter = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
    return tf.gather_nd(tensor, tf.stack((counter, indices), -1))


def desc_loss(dists, pids, pos_margin=0.1, neg_margin=1.4, batch_precision_at_k=None):
    """Computes the contrastive loss.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        pos_margin, neg_margin.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    with tf.name_scope("desc_loss"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        negative_mask = tf.logical_not(same_identity_mask)
        # positive_mask = tf.logical_xor(same_identity_mask,
        #                              tf.eye(tf.shape(pids)[0], dtype=tf.bool))

        furthest_positive = tf.reduce_max(dists * tf.cast(same_identity_mask, tf.float32), axis=1)
        # closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])), (dists, negative_mask), tf.float32)
        closest_negative = tf.reduce_min(dists + 1e5 * tf.cast(same_identity_mask, tf.float32), axis=1)
        # Another way of achieving the same, though more hacky:
        # closest_negative_col = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)
        # closest_negative_row = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=0)
        # closest_negative = tf.minimum(closest_negative_col, closest_negative_row)

        diff = furthest_positive - closest_negative
        # calculate average diff to monitor the training
        # average_diff = tf.reduce_min(diff)
        average_negative = tf.map_fn(lambda x: tf.reduce_mean(tf.boolean_mask(x[0], x[1])), (dists, negative_mask), tf.float32)
        average_diff = tf.reduce_mean(furthest_positive - average_negative)
        # accuracy = tf.count_nonzero(tf.greater_equal(0., diff), dtype=tf.int64) / tf.shape(diff)[0]
        accuracy = tf.reduce_sum(tf.cast(tf.greater_equal(0., diff), tf.float32)) / tf.cast(tf.shape(diff)[0], tf.float32)

        # contrastive loss
        diff = tf.maximum(furthest_positive - pos_margin, 0.0) + tf.maximum(neg_margin - closest_negative, 0.0)

    if batch_precision_at_k is None:
        # tf.summary.scalar('loss', tf.reduce_mean(diff))
        return tf.reduce_mean(diff), accuracy, average_diff

    # For monitoring, compute the within-batch top-1 accuracy and the
    # within-batch precision-at-k, which is somewhat more expressive.
    with tf.name_scope("monitoring"):
        # This is like argsort along the last axis. Add one to K as we'll
        # drop the diagonal.
        _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k + 1)

        # Drop the diagonal (distance to self is always least).
        indices = indices[:, 1:]

        # Generate the index indexing into the batch dimension.
        # This is simething like [[0,0,0],[1,1,1],...,[B,B,B]]
        batch_index = tf.tile(
            tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
            (1, tf.shape(indices)[1]))

        # Stitch the above together with the argsort indices to get the
        # indices of the top-k of each row.
        topk_indices = tf.stack((batch_index, indices), -1)

        # See if the topk belong to the same person as they should, or not.
        topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)

        # All of the above could be reduced to the simpler following if k==1
        # top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])

        topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
        top1 = tf.reduce_mean(topk_is_same_f32[:, 0])
        prec_at_k = tf.reduce_mean(topk_is_same_f32)

        # Finally, let's get some more info that can help in debugging while
        # we're at it!
        negative_dists = tf.boolean_mask(dists, negative_mask)
        positive_dists = tf.boolean_mask(dists, same_identity_mask)

        return diff, top1, prec_at_k, topk_is_same, negative_dists, positive_dists


def det_loss(dists, score1, score2, pids):
    with tf.name_scope("det_loss"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1), tf.expand_dims(pids, axis=0))

        furthest_positive = tf.reduce_max(dists * tf.cast(same_identity_mask, tf.float32), axis=1)
        closest_negative = tf.reduce_min(dists + 1e5 * tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = tf.expand_dims(furthest_positive - closest_negative, 1)
        score_loss = tf.reduce_mean(diff * (score1 + score2 + 1e-6))
        # score_loss = tf.Print(score_loss, [score_loss], message="score_loss")
        return score_loss


LOSS_CHOICES = {
    'desc_loss': desc_loss,
    'det_loss': det_loss,
}
