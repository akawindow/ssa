"""Optional TensorFlow stage: frozen forecasting proxy and shared component selector."""
import numpy as np
import hashlib


def weight_digest(arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.asarray(array)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def weight_summary(weights):
    w = np.asarray(weights)
    return {'min':float(w.min()), 'max':float(w.max()), 'mean':float(w.mean()),
        'mean_within_window_std':float(w.std(axis=1).mean()),
        'mean_by_component':w.mean(axis=0).tolist()}


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


def build_feature_selector(tf, proxy, history, mode, initial_feature_weights):
    """Matched 33-parameter readout; only feature pretraining differs between modes."""
    inputs = tf.keras.Input(shape=(history,1))
    flattened = tf.keras.layers.Flatten()(inputs)
    if mode == 'proxy_features':
        extractor = tf.keras.Model(proxy.inputs, proxy.layers[0].output, name='frozen_proxy_features')
        extractor.trainable = False
        features = extractor(flattened, training=False)
    elif mode == 'random_features':
        extractor = tf.keras.layers.Dense(32, activation='elu', trainable=False, name='frozen_random_features')
        features = extractor(flattened)
        extractor.set_weights(initial_feature_weights)
    else:
        raise ValueError(f'Unknown feature selector mode: {mode}')
    # Equal head initialization for the matched pair, independent of extractor creation.
    output = tf.keras.layers.Dense(1, activation='sigmoid',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=731),
        bias_initializer=tf.keras.initializers.Constant(4.0), name='retention_head')(features)
    return tf.keras.Model(inputs, output, name=mode+'_selector')


def learn(config, windows, components):
    tf = tensorflow(config['seed'])
    proxy = build_proxy(tf, config['history'], config['horizon'])
    initial_feature_weights = [w.copy() for w in proxy.layers[0].get_weights()]
    proxy.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),loss='mse')
    callbacks = lambda:[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['patience'],restore_best_weights=True)]
    print('Pretraining forecasting proxy on reference observations...',flush=True)
    proxy_history = proxy.fit(windows['train']['reference'],windows['train']['y'],
        validation_data=(windows['val']['reference'],windows['val']['y']),
        epochs=config['proxy_epochs'],batch_size=config['batch_size'],callbacks=callbacks(),verbose=0)
    proxy.trainable = False
    frozen_weights = [w.copy() for w in proxy.get_weights()]
    mode = config.get('selector_mode', 'raw_conv')
    if mode == 'raw_conv':
        selector = build_selector(tf, config['history'])
    else:
        selector = build_feature_selector(tf, proxy, config['history'], mode, initial_feature_weights)
    initial_selector = [w.copy() for w in selector.get_weights()]
    initial_trainable_selector = [w.numpy().copy() for w in selector.trainable_variables]

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
    initial_weights = {}
    initial_proxy_mse = {}
    for split,c in components.items():
        before, before_w = task.reconstruct(c)
        initial_weights[split] = weight_summary(before_w.numpy()[...,0])
        initial_proxy_mse[split] = float(np.square(proxy(before,training=False).numpy()-windows[split]['y']).mean())
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
    proxy_diagnostics = {}
    for split, data in windows.items():
        proxy_diagnostics[split] = {}
        for name, x in [('reference',data['reference']),('observed',data['x']),
                        ('fixed_ssa',components[split][:,:-1,:].sum(axis=1)),('task_ssa',output[split])]:
            prediction = proxy(x,training=False).numpy()
            proxy_diagnostics[split][name+'_mse_z'] = float(np.square(prediction-data['y']).mean())
    training = {'proxy_loss':proxy_history.history,'selector_loss':gate_history.history,
        'proxy_weights_unchanged':unchanged,'selector_weights_updated':changed,
        'selector_parameters':selector.count_params(),'proxy_parameters':proxy.count_params(),
        'selector_trainable_parameters':int(sum(np.prod(w.shape) for w in selector.trainable_variables)),
        'selector_mode':mode,'selector_bias_initialization':4.0,
        'pretrained_proxy_sha256':weight_digest(frozen_weights),
        'initial_trainable_selector_sha256':weight_digest(initial_trainable_selector),
        'initial_weight_summary':initial_weights,
        'final_weight_summary':{k:weight_summary(v) for k,v in weights.items()},
        'initial_selector_proxy_mse_z':initial_proxy_mse,
        'frozen_proxy_diagnostics':proxy_diagnostics,
        'supervision':'Future forecast targets; no clean-history reconstruction loss',
        'proxy_pretraining_input':'Original reference histories, not artificially contaminated histories'}
    return output,weights,training,proxy,selector
