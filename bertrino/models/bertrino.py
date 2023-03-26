import numpy as np
import tensorflow as tf


def get_sequential_positional_encoding(max_len, d_emb):
    pos_enc = np.array([[
                    pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 \
                    else np.zeros(d_emb) for pos in range(max_len)
                ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def get_continuous_positional_encoding(timestamps, d_emb):
    pos_enc = np.array([
        [timestamp / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] \
        if timestamp != 0 else np.zeros(d_emb) for timestamp in timestamps
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def get_attention_module(config):
    return tf.keras.layers.MultiHeadAttention(
        num_heads=config.num_attention_heads,
        key_dim=config.hidden_size // config.num_attention_heads,
    )


def get_ffn_module(config):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(config.intermediate_size, activation=config.hidden_act),
        tf.keras.layers.Dense(config.hidden_size),
    ])


def get_transformer_encoder_module(config):
    def transformer_encoder_module(inputs):
        query, key, value = inputs
        attention_output = get_attention_module(config)(query, value, key)
        attention_output = tf.keras.layers.Dropout(config.hidden_dropout_prob)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(query + attention_output)
        ffn_output = get_ffn_module(config)(attention_output)
        ffn_output = tf.keras.layers.Dropout(config.hidden_dropout_prob)(ffn_output)
        sequence_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        return sequence_output

    return transformer_encoder_module


def create_bert_model(config):
    inputs = tf.keras.layers.Input((config.max_position_embeddings,), dtype=tf.int64)
    word_embeddings = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size, name="token_embedding")(inputs)
    seq_position_embeddings = tf.keras.layers.Embedding(
        input_dim=config.max_position_embeddings,
        output_dim=config.hidden_size,
        weights=[get_sequential_positional_encoding(config.max_position_embeddings, config.hidden_size)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.max_position_embeddings, delta=1))
    embeddings = word_embeddings + seq_position_embeddings

    encoder_output = embeddings
    for i in range(config.num_hidden_layers):
        transformer_encoder = get_transformer_encoder_module(config)
        encoder_output = transformer_encoder([encoder_output, encoder_output, encoder_output])

    return tf.keras.Model(inputs, encoder_output, name="bert_model")


def create_masked_language_bert_model(config):
    bert_model = create_bert_model(config)
    mlm_inputs = tf.keras.layers.Input((config.max_position_embeddings,), dtype=tf.int64)
    bert_output = bert_model(mlm_inputs)
    mlm_output = tf.keras.layers.Dense(config.vocab_size, name="mlm_cls", activation="softmax")(bert_output)
    mlm_model = tf.keras.Model(mlm_inputs, mlm_output, name="masked_bert_model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LR)
    mlm_model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), weighted_metrics=[])
    return mlm_model
