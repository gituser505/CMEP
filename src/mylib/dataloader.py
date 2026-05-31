from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split

class IsingDataLoader:
    def __init__(self, file_spins, file_betas, L, N, **kwargs):
        self.spins = np.memmap(file_spins, dtype=np.int8, mode="r", shape=(N, L, L, 1))
        self.betas = np.memmap(file_betas, dtype=np.float64, mode="r", shape=(N, 1))
        self.L = L
        self.N = N

    def get_betas(self, indices):
        return np.array(self.betas[indices]).squeeze(-1)

    def get_spins(self, indices):
        return np.array(self.spins[indices]).squeeze(-1)

    @staticmethod
    def normalize_array(x):
        min, max = x.min(), x.max()
        return (x - min) / (max-min)

    @staticmethod
    def cast_to_float32(spins, betas):
        return tf.cast(spins, tf.float32), tf.cast(betas, tf.float32)

    def random_augment(self, spins_batch, betas_batch):
        batch_size = tf.shape(spins_batch)[0]
        
        invert_mask = tf.random.uniform([batch_size, 1, 1, 1], 0, 2, dtype=tf.int32) == 1
        inverted_spins = tf.constant(1, dtype=spins_batch.dtype) - spins_batch
        spins_batch = tf.where(invert_mask, inverted_spins, spins_batch)

        lr_mask = tf.random.uniform([batch_size, 1, 1, 1], 0, 2, dtype=tf.int32) == 1
        spins_batch = tf.where(lr_mask, tf.image.flip_left_right(spins_batch), spins_batch)

        ud_mask = tf.random.uniform([batch_size, 1, 1, 1], 0, 2, dtype=tf.int32) == 1
        spins_batch = tf.where(ud_mask, tf.image.flip_up_down(spins_batch), spins_batch)

        tp_mask = tf.random.uniform([batch_size, 1, 1, 1], 0, 2, dtype=tf.int32) == 1
        transposed_spins = tf.transpose(spins_batch, perm=[0, 2, 1, 3])
        
        spins_batch = tf.where(tp_mask, transposed_spins, spins_batch)

        return spins_batch, betas_batch

    
    def get_training_data(self, split_ratio, batch_size, augment=False):
        #stratified split indices for training/validation
        indices = np.arange(self.N)
        betas_norm = self.normalize_array(self.betas.copy())
        _, beta_category = np.unique(betas_norm, return_inverse=True)
        train_idx, val_idx = train_test_split(indices, train_size=split_ratio, stratify=beta_category)

        train_data = Dataset.from_tensor_slices((self.spins[train_idx], betas_norm[train_idx]))
        val_data = Dataset.from_tensor_slices((self.spins[val_idx], betas_norm[val_idx]))
        
        train_data = train_data.shuffle(len(train_idx)).batch(batch_size)
        val_data = val_data.batch(batch_size)

        if augment is True:
            train_data = train_data.map(self.random_augment, num_parallel_calls=tf.data.AUTOTUNE)

        train_data = train_data.map(self.cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        val_data = val_data.map(self.cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)

        train_data = train_data.prefetch(tf.data.AUTOTUNE)
        val_data = val_data.prefetch(tf.data.AUTOTUNE)

        return train_data, val_data
    
    def get_training_data_homogeneous(self, split_ratio, batch_size, augment=False):
        indices = np.arange(self.N)
        betas_norm = self.normalize_array(self.betas.copy())
        _, beta_category = np.unique(betas_norm.flatten(), return_inverse=True)
        train_idx, val_idx = train_test_split(indices, train_size=split_ratio, stratify=beta_category)

        def build_pure_dataset(idx_array, is_training):
            unique_cats = np.unique(beta_category[idx_array])
            combined_ds = None
            total_batches = 0
            
            for cat in unique_cats:
                cat_idx = idx_array[beta_category[idx_array] == cat]
                
                ds = Dataset.from_tensor_slices((self.spins[cat_idx], betas_norm[cat_idx]))
                
                if is_training:
                    ds = ds.shuffle(len(cat_idx))
                    total_batches += len(cat_idx) // batch_size

                ds = ds.batch(batch_size, drop_remainder=is_training)
                
                if combined_ds is None:
                    combined_ds = ds
                else:
                    combined_ds = combined_ds.concatenate(ds)
            
            if is_training and total_batches > 0:
                combined_ds = combined_ds.shuffle(total_batches)
                
            # 6. Apply augmentation and map functions
            if is_training and augment:
                combined_ds = combined_ds.map(self.random_augment, num_parallel_calls=tf.data.AUTOTUNE)
                
            return combined_ds

        train_data = build_pure_dataset(train_idx, is_training=True)
        val_data = build_pure_dataset(val_idx, is_training=False)

        train_data = train_data.map(self.cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
        val_data = val_data.map(self.cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)

        return train_data.prefetch(tf.data.AUTOTUNE), val_data.prefetch(tf.data.AUTOTUNE)


if __name__=="__main__":
    import os
    import sys
    import psutil

    def print_memory_usage(message):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"[{message}] RAM Usage: {mem_mb:.2f} MB")
    
    import gc
    import time

    L = 32
    N = 20000
    split_ratio = 0.5
    batch_size = 128
    buffer_size = N*split_ratio
    augment = True

    def norm_array(x):
        min, max = x.min(), x.max()
        return (x - min) / (max-min)
    
    print_memory_usage("before any data")
    betas_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    print(norm_array(np.array(betas_list)))
    samples_per_beta = N // len(betas_list)
    betas = np.repeat(betas_list, samples_per_beta).astype(np.float32)

    rng = np.random.default_rng()
    spins = rng.integers(0, 2, size=(N, L, L), dtype=np.int8)
    print_memory_usage("After spins & betas")

    ROOT = Path(__file__).parent.parent.parent
    spins_file = ROOT/f"data/ising_test_spins_L={L}_N={N}.bin"
    betas_file = ROOT/f"data/ising_test_betas_L={L}_N={N}.bin"

    spins.tofile(spins_file)
    betas.tofile(betas_file)
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    all_indices = np.arange(N)

    print_memory_usage("before loaded spins & betas")
    spins_loaded = loader.get_spins(all_indices)
    betas_loaded = loader.get_betas(all_indices)
    print(sys.getsizeof(spins_loaded)/1024/1024)
    print(sys.getsizeof(betas_loaded)/1024/1024)
    print_memory_usage("after loaded spins & betas")

    gc.collect() # Force Python to clean up temp buffers
    time.sleep(1) # Give the OS a second to settle the Page Cache

    print_memory_usage("After settling")

    print(spins.shape)
    print(spins_loaded.shape)
    print(betas.shape)
    print(betas_loaded.shape)
    print(np.unique(betas_loaded))

    for s,sl in zip(spins, spins_loaded):
        assert np.allclose(s, sl, atol=1e-5), "Spins did not load correctly."
    for b,bl in zip(betas, betas_loaded):
        assert np.allclose(s, sl, atol=1e-5), "Betas did not load correctly."

    train_data, val_data = loader.get_training_data(split_ratio, batch_size, augment)
    print_memory_usage("after get training data")
    for i, (spins_batch, labels_batch) in enumerate(train_data.take(5)):
        print(f"\nBatch {i+1}:")
        print(f"Spins Batch Shape:  {spins_batch.shape}")
        print(f"Labels Batch Shape: {labels_batch.shape}")
        betas_numpy = labels_batch.numpy()
        unique_betas = np.unique(betas_numpy)
        print(f"Unique beta values: {unique_betas}")
