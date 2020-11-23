import tensorflow as tf

def dot_product_cross_entropy_loss(labels, embeddings):
    embeddings = tf.math.l2_normalize(embeddings, axis=1)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings)) * 5.0
    dot_product = tf.math.sigmoid(dot_product)

    positive_mask = tf.math.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    negative_mask = tf.math.logical_not(positive_mask)

    p_loss = tf.where(positive_mask, -1.0 * tf.math.log(dot_product), 0.0)
    p_loss = tf.reduce_sum(p_loss) / tf.reduce_sum(tf.cast(positive_mask, dtype=tf.float32))

    n_loss = tf.where(negative_mask, -1.0 * tf.math.log(1.0 - dot_product), 0.0)
    n_loss = tf.reduce_sum(n_loss) / tf.reduce_sum(tf.cast(negative_mask, dtype=tf.float32))

    return p_loss + n_loss