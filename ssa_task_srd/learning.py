"""Optional TensorFlow stage: frozen forecasting proxy and shared component selector."""
import numpy as np


def tensorflow(seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf


def build_proxy(tf, history, horizon):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(history,)),
        tf.keras.layers.Dense(32, activation='elu'),
        tf.keras.layers.Dense(horizon),
    ], name='forecast_proxy')


def build_selector(tf, history):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(history,1)),
        tf.keras.layers.Conv1D(8,9,padding='same',activation='elu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8,activation='elu'),
        tf.keras.layers.Dense(1,activation='sigmoid',bias_initializer=tf.keras.initializers.Constant(4.0)),
    ], name='shared_component_selector')


def learn(config, windows, components):
    tf = tensorflow(config['seed'])
    proxy = build_proxy(tf, config['history'], config['horizon'])
    proxy.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),loss='mse')
    callbacks = lambda:[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['patience'],restore_best_weights=True)]
    print('Pretraining forecasting proxy on reference observations...',flush=True)
    proxy_history = proxy.fit(windows['train']['reference'],windows['train']['y'],
        validation_data=(windows['val']['reference'],windows['val']['y']),
        epochs=config['proxy_epochs'],batch_size=config['batch_size'],callbacks=callbacks(),verbose=0)
    proxy.trainable = False
    frozen_weights = [w.copy() for w in proxy.get_weights()]
    selector = build_selector(tf, config['history'])
    initial_selector = [w.copy() for w in selector.get_weights()]

    class TaskSSA(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.selector = selector
            self.proxy = proxy

        def reconstruct(self, c, training=False):
            c = tf.cast(c,tf.float32)
            shape = tf.shape(c)
            w = self.selector(tf.reshape(c,[-1,config['history'],1]),training=training)
            w = tf.reshape(w,[shape[0],shape[1],1])
            return tf.reduce_sum(c*w,axis=1), w

        def call(self,c,training=False):
            rebuilt,_ = self.reconstruct(c,training=training)
            return self.proxy(rebuilt,training=False)

    task = TaskSSA()
    task.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),loss='mse')
    print('Training component selector; proxy weights frozen...',flush=True)
    gate_history = task.fit(components['train'],windows['train']['y'],
        validation_data=(components['val'],windows['val']['y']),
        epochs=config['selector_epochs'],batch_size=config['batch_size'],callbacks=callbacks(),verbose=0)
    unchanged = all(np.array_equal(a,b) for a,b in zip(frozen_weights,proxy.get_weights()))
    changed = any(not np.array_equal(a,b) for a,b in zip(initial_selector,selector.get_weights()))
    if not unchanged or not changed:
        raise RuntimeError('Failed frozen-proxy / updated-selector verification')
    output, weights = {}, {}
    for split,c in components.items():
        rebuilt,w = task.reconstruct(c)
        output[split],weights[split] = rebuilt.numpy(),w.numpy()[...,0]
    training = {'proxy_loss':proxy_history.history,'selector_loss':gate_history.history,
        'proxy_weights_unchanged':unchanged,'selector_weights_updated':changed,
        'selector_parameters':selector.count_params(),'proxy_parameters':proxy.count_params(),
        'supervision':'Future forecast targets; no clean-history reconstruction loss',
        'proxy_pretraining_input':'Original reference histories, not artificially contaminated histories'}
    return output,weights,training,proxy,selector
