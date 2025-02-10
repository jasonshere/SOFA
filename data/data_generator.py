import itertools
import numpy as np
import tensorflow as tf
from itertools import zip_longest

# Global Variables for IDs
test_id = 1
train_id = 1


def generate_test(session_max_length, test_sessions):
    global test_id
    for session in test_sessions:
        yield from _process_session(session, session_max_length, test_id, is_training=False)
        test_id += 1


def generate_train(session_max_length, train_sessions):
    global train_id
    for session in train_sessions:
        yield from _process_session(session, session_max_length, train_id, is_training=True)
        train_id += 1


def _process_session(session, session_max_length, session_id, is_training):
    sub_sessions, labels, masks, ids, is_last = [], [], [], [], []
    length = len(session)
    for i in range(length - 1):
        sub_sessions.append(session[:i + 1])
        labels.append(session[i + 1])
        masks.append(True)
        is_last.append(i == (length - 2))
        ids.append(session_id)

    sub_sessions_ = sub_sessions.copy()
    sub_sessions_[0] = sub_sessions_[0] + [0] * (session_max_length - 1)

    reversed_training_features = np.flip(np.array(list(zip_longest(*sub_sessions_, fillvalue=0))).T)
    mask = reversed_training_features != 0
    flipped_mask = mask[:, ::-1]
    reversed_training_features[flipped_mask] = reversed_training_features[mask]
    reversed_training_features[~flipped_mask] = 0
    reversed_training_features = reversed_training_features[::-1]
    reversed_training_features_mask = (reversed_training_features > 0).astype(np.float32)
    training_labels = np.array(labels)

    training_adjacency_matrice, training_alias_inputs, training_session_items, training_session_non_zero_items = (
        [], [], [], [])

    for u_input, mask, target in zip(reversed_training_features, reversed_training_features_mask, training_labels):
        node = np.unique(u_input)
        items = node.tolist() + (session_max_length - len(node)) * [0]
        non_zero_items = np.array(items)
        non_zero_items[non_zero_items == 0] = non_zero_items[1]
        non_zero_items = non_zero_items.tolist()

        adjacency_matrix = np.zeros((session_max_length, session_max_length))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adjacency_matrix[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u != v and adjacency_matrix[u][v] == 0:
                adjacency_matrix[u][v] = 1
                adjacency_matrix[v][u] = 1

        training_adjacency_matrice.append(adjacency_matrix)
        training_alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        training_session_items.append(items)
        training_session_non_zero_items.append(non_zero_items)

    yield _pad_and_format(session_max_length, training_adjacency_matrice, training_alias_inputs,
                          training_session_items, training_session_non_zero_items, reversed_training_features_mask,
                          training_labels, ids, masks, is_last)


def _pad_and_format(session_max_length, adjacency_matrices, alias_inputs, session_items, non_zero_items,
                    reversed_features_mask, labels, ids, masks, is_last):
    def pad_to_max_length(data, shape):
        if len(data) == 0:
            return np.zeros((session_max_length, *shape))  # Handle empty data
        padding_zeros = np.zeros((session_max_length - len(data), *shape))
        return np.vstack([data, padding_zeros])

    # Handle empty session_items by replacing with a default padded sequence
    session_items = [seq if len(seq) > 0 else [-1] * session_max_length for seq in session_items]

    sub = np.array([seq + [-1] * (session_max_length - len(seq)) for seq in session_items])
    if sub.shape[0] < session_max_length:
        padding_zeros_for_sub = np.zeros((session_max_length - sub.shape[0], session_max_length))
        sub = np.vstack([sub, padding_zeros_for_sub])

    adjacency_matrices = pad_to_max_length(adjacency_matrices, (session_max_length, session_max_length))
    alias_inputs = pad_to_max_length(alias_inputs, (session_max_length,))
    session_items = pad_to_max_length(session_items, (session_max_length,))
    non_zero_items = pad_to_max_length(non_zero_items, (session_max_length,))
    reversed_features_mask = pad_to_max_length(reversed_features_mask, (session_max_length,))
    labels = np.array(labels.tolist() + [0] * (session_max_length - len(labels)))
    dids = np.array(ids + [-1] * (session_max_length - len(ids)))
    masks = masks + [False] * (session_max_length - len(masks))
    is_last = is_last + [False] * (session_max_length - len(is_last))

    history = [ids[:ind + 1] + [-1] * (session_max_length - len(ids[:ind + 1])) for ind, _ in enumerate(ids)]
    history = np.vstack([history, np.zeros((session_max_length - len(history), session_max_length))])

    return (alias_inputs, adjacency_matrices, session_items, non_zero_items,
            reversed_features_mask, labels, masks, dids, history, is_last, sub)




def create_dataset(generator_fn, session_max_length, sessions, output_types):
    return tf.data.Dataset.from_generator(
        lambda: generator_fn(session_max_length, sessions),
        output_types=output_types
    )


def prepare_datasets(session_max_length, training_sessions, test_sessions):
    training_ds = create_dataset(
        generate_train, session_max_length, training_sessions,
        output_types=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.bool, tf.int64, tf.int64, tf.bool, tf.int64)
    )

    test_ds = create_dataset(
        generate_test, session_max_length, test_sessions,
        output_types=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.bool, tf.int64, tf.int64, tf.bool, tf.int64)
    )

    return training_ds, test_ds
