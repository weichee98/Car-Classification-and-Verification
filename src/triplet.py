import numpy as np
import tensorflow as tf


class TripletInput:

    def __init__(self, filenames, labels, img_size=(64, 64)):
        self.__IMG_SIZE = img_size
        self.filenames = filenames
        self.labels = labels
        self.label_indices = {
            label: np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        self.choose_labels = [k for k in self.label_indices if len(self.label_indices[k]) > 1]
        self.num_classes = len(self.choose_labels)

    def __parse_function(self, filename):
        image_string = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image_string, channels=3)
        return image

    def __image_rescale_resize(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, self.__IMG_SIZE)
        return image

    def __image_augmentation(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(
            image, 
            size=(
                np.random.randint(0.8 * self.__IMG_SIZE[0], self.__IMG_SIZE[0]), 
                np.random.randint(0.8 * self.__IMG_SIZE[1], self.__IMG_SIZE[1]), 
                3
            )
        )
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_contrast(image, 0.7, 1)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_saturation(image, 0.5, 1.5)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.resize(image, self.__IMG_SIZE)
        return image

    def get_triplets_batch(self, batch_size=30, mode='train'):
        idxs_a, idxs_p, idxs_n = [], [], []
        if batch_size < self.num_classes // 2:
            labels = np.random.choice(self.choose_labels, size=(batch_size, 2), replace=False)
        else:
            labels = list()
            for _ in range(batch_size // (self.num_classes // 2)):
                new_l = np.random.choice(self.choose_labels, size=(self.num_classes // 2, 2), replace=False)
                labels.append(new_l)
            else:
                new_l = np.random.choice(self.choose_labels, size=(batch_size % (self.num_classes // 2), 2), replace=False)
                labels.append(new_l)
            labels = np.concatenate(labels, axis=0)
        for label_l, label_r in labels:
            a, p = np.random.choice(self.label_indices[label_l], size=2, replace=False)
            n, = np.random.choice(self.label_indices[label_r], size=1, replace=False)
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)

        if len(idxs_a) == len(idxs_p) == len(idxs_n) == batch_size:
            images = map(self.__image_rescale_resize, map(self.__parse_function, self.filenames[idxs_a + idxs_p + idxs_n]))
            if mode == 'train':
                images = map(self.__image_augmentation, images)
            images = tf.convert_to_tensor(list(images))
            a, p, n = images[:batch_size], images[batch_size:2 * batch_size], images[2 * batch_size:]
            return a, p, n
        else:
            raise Exception(f'Inconsistent length of idxs_a, idxs_p, idxs_n, batch_size: {len(idxs_a)}, {len(idxs_p)}, {len(idxs_n)}, {batch_size}')


def triplet_loss(model_anchor, model_positive, model_negative, margin):
    model_anchor = tf.math.l2_normalize(model_anchor, axis=1)
    model_positive = tf.math.l2_normalize(model_positive, axis=1)
    model_negative = tf.math.l2_normalize(model_negative, axis=1)
    distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
    distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))
    return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))

def triplet_loss_hard(model_anchor, model_positive, model_negative, margin):
    model_anchor = tf.math.l2_normalize(model_anchor, axis=1)
    model_positive = tf.math.l2_normalize(model_positive, axis=1)
    model_negative = tf.math.l2_normalize(model_negative, axis=1)
    distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
    distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))
    return tf.reduce_mean(tf.reduce_max(distance1) - tf.reduce_min(distance2) + margin)


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.math.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.math.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.math.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.math.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.math.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.math.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.math.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.math.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.math.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False, semi_hard=False, hard=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrixd
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, dtype=tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    hard_loss = tf.reduce_max(tf.reduce_max(triplet_loss, axis=2), axis=1)
    
    semihard = tf.math.logical_and(
        tf.math.greater(anchor_negative_dist, anchor_positive_dist),
        tf.math.greater(anchor_positive_dist + margin, anchor_negative_dist),
    )
    semihard = tf.cast(semihard, dtype=tf.float32)
    semihard_loss = tf.multiply(semihard, triplet_loss)
    semihard_loss = tf.reduce_max(tf.reduce_max(semihard_loss, axis=2), axis=1)
    semihard_loss = tf.where(tf.greater(semihard_loss, 1e-16), semihard_loss, hard_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    valid_triplets = tf.cast(tf.greater(hard_loss, 1e-16), dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    hard_loss = tf.reduce_sum(hard_loss) / (num_positive_triplets + 1e-16)

    valid_triplets = tf.cast(tf.greater(semihard_loss, 1e-16), dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    semihard_loss = tf.reduce_sum(semihard_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, hard_loss, semihard_loss
