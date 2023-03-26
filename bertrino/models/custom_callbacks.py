import tensorflow as tf
import numpy as np


class MaskedTextGenerator(tf.keras.callbacks.Callback):
    def __init__(self, sample_tokens, decoder, top_k=5, mask_token_id=4):
        super().__init__()
        self.sample_tokens = self.pad_to_len(sample_tokens)
        self.decoder = decoder
        self.original_len = tf.shape(sample_tokens)[-1]
        self.mask_token_id = mask_token_id
        self.k = top_k

    @staticmethod
    def pad_to_len(arr, pad_to=128):
        current_len = tf.shape(arr)[-1]
        pad_amount = pad_to - current_len
        return tf.pad(arr, [(0, 0), (0, pad_amount)], mode='CONSTANT', constant_values=6)

    def decode(self, tokens):
        return self.decoder(tokens)

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model(self.sample_tokens, training=False).numpy()[:, :self.original_len]
        masked_index = np.where(self.sample_tokens == self.mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]
        top_indices = mask_prediction[0].argsort()[-self.k:][::-1]
        values = mask_prediction[0][top_indices]

        _input_text = self.decode(self.sample_tokens[0].numpy()[:self.original_len])
        print("\n\n--- TEST MODEL PERFORMANCE ---")
        print(f"\tINPUT TEXT --> '{_input_text}'")

        print(f"\n--- TOP {len(top_indices)} PROBABLE INFERENCES ---")
        for i in range(len(top_indices)):
            v, p = values[i], top_indices[i]
            _tokens = np.copy(self.sample_tokens[0, :self.original_len])
            _tokens[masked_index[0]] = p
            print(f"\tOUR {i + 1}th PREDICTION --> '{self.decode(_tokens)}'  ([mask]={self.decode(p)}@{v:.4f})")
        print("\n")


class MaskedTextGeneratorv2(tf.keras.callbacks.Callback):
    def __init__(self, sample_tokens, decoder, top_k=5, mask_token_id=4):
        super().__init__()
        self.sample_tokens = self.pad_to_len(sample_tokens)
        self.decoder = decoder
        self.original_len = tf.shape(sample_tokens)[-1]
        self.mask_token_id = mask_token_id
        self.k = top_k

    @staticmethod
    def pad_to_len(arr, pad_to=128):
        return tf.keras.preprocessing.sequence.pad_sequences(arr, maxlen=pad_to, padding='post', value=6)

    def decode(self, tokens):
        # Make sure to import or define the appropriate function for decoding tokens into text.
        return self.decoder(tokens)

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model(self.sample_tokens, training=False).numpy()[:, :self.original_len]
        masked_index = np.where(self.sample_tokens == self.mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]
        top_indices = mask_prediction[0].argsort()[-self.k:][::-1]
        values = mask_prediction[0][top_indices]

        _input_text = self.decode(self.sample_tokens[0].numpy()[:self.original_len])
        print(f"\n--- TEST MODEL PERFORMANCE ---")
        print(f"INPUT TEXT --> '{_input_text}'")

        print(f"\n--- TOP {len(top_indices)} PROBABLE INFERENCES ---")
        for i in range(len(top_indices)):
            v, p = values[i], top_indices[i]
            _tokens = np.copy(self.sample_tokens[0, :self.original_len])
            _tokens[masked_index[0]] = p
            print(f"OUR {i + 1}th PREDICTION --> '{self.decode(_tokens)}'  ([mask]={self.decode(p)}@{v:.4f})")
        print("\n")


