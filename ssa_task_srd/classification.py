"""Classification-specific TASK adaptation; forecasting code is unchanged."""
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from .ssa import decompose
from .learning import weight_digest
from .collaborative import equal_weights


def split_indices(labels, seed):
    ix = np.arange(len(labels))
    fit, rest = train_test_split(ix, test_size=.30, stratify=labels, random_state=seed)
    selection, assessment = train_test_split(rest, test_size=.5, stratify=labels[rest], random_state=seed+1)
    return {'fit': np.sort(fit), 'selection': np.sort(selection), 'assessment': np.sort(assessment)}


def full_ssa(x, embedding, threshold):
    """Reuse existing SSA; all elementary RCs plus exact residual enter task branch."""
    if not 0 < threshold <= 1:
        raise ValueError('Energy threshold must be in (0,1]')
    c = decompose(x, embedding, embedding)
    trajectory = np.lib.stride_tricks.sliding_window_view(x, embedding).T
    energy = np.maximum(np.linalg.eigvalsh(trajectory @ trajectory.T)[::-1], 0.)
    cumulative = np.cumsum(energy)
    rank = 1 if cumulative[-1] == 0 else min(embedding, int(np.searchsorted(cumulative, threshold*cumulative[-1]))+1)
    return c.astype(np.float32), rank


def build_classifier(tf, length):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(length,)),
        tf.keras.layers.Reshape((length, 1)),
        tf.keras.layers.Conv1D(8, 7, strides=2, padding='same', activation='elu'),
        tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', activation='elu'),
        tf.keras.layers.GlobalAveragePooling1D()], name='shared_encoder')
    return tf.keras.Sequential([tf.keras.layers.Input(shape=(length,)), encoder,
                                tf.keras.layers.Dense(2)], name='classification_proxy')


def ce(tf, y, logits):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.cast(y, tf.int32), logits, from_logits=True))


def scores(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits-logits.max(axis=1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    pred = logits.argmax(axis=1)
    return {'accuracy': float(accuracy_score(labels, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
            'macro_f1': float(f1_score(labels, pred, average='macro', zero_division=0)),
            'cross_entropy': float(-logp[np.arange(len(labels)), labels].mean()),
            'confusion_matrix': confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def make_optimizer(tf, cfg, n):
    steps = cfg.get('schedule_epochs', cfg['collaborative_epochs'])*int(np.ceil(n/cfg['batch_size']))
    schedule = tf.keras.optimizers.schedules.CosineDecay(cfg['learning_rate'], steps, alpha=.9)
    return tf.keras.optimizers.AdamW(schedule, weight_decay=.004, amsgrad=True)


def make_proxy_step(tf, proxy, opt):
    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = ce(tf, y, proxy(tf.stop_gradient(x), training=True))
        grads = tape.gradient(loss, proxy.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.)
        opt.apply_gradients(zip(grads, proxy.trainable_variables))
        return loss
    return step


def build_gate(tf, proxy, branch, cfg, n):
    class Gate:
        def __init__(self):
            self.proxy = proxy
            self.head_optimizer = make_optimizer(tf, cfg, n)
            if branch == 'global_gate':
                self.logit = tf.Variable(0., name='global_logit', trainable=True)
            else:
                self.head = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(32,)),
                    tf.keras.layers.Dense(1, activation='sigmoid',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=731),
                        bias_initializer=tf.keras.initializers.Constant(cfg['selector_bias']),
                        kernel_constraint=tf.keras.constraints.max_norm(cfg['selector_max_norm']))])

        def parameters(self):
            return [self.logit] if branch == 'global_gate' else self.head.trainable_variables

        def weights(self):
            return [np.array(v.numpy(), copy=True) for v in self.parameters()]

        def restore(self, values):
            for v, a in zip(self.parameters(), values):
                v.assign(a)

        def reconstruct(self, x):
            x = tf.cast(x, tf.float32)
            if branch == 'global_gate':
                w = tf.sigmoid(self.logit)
                return x*w, w
            shape = tf.shape(x)
            features = proxy.layers[0](tf.reshape(x, [-1, cfg['length']]), training=False)
            w = tf.reshape(self.head(features, training=False), [shape[0], shape[1], 1])
            return tf.reduce_sum(x*w, axis=1), w

        @tf.function(reduce_retracing=True)
        def selector_step(self, x, y):
            with tf.GradientTape() as tape:
                rebuilt, _ = self.reconstruct(x)
                loss = ce(tf, y, proxy(rebuilt, training=False))
            variables = self.parameters()
            grads = tape.gradient(loss, variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.)
            self.head_optimizer.apply_gradients(zip(grads, variables))
            return tf.stop_gradient(rebuilt)
    return Gate()


def predict(proxy, gate, x, batch_size=64):
    parts = []
    for start in range(0, len(x), batch_size):
        part = x[start:start+batch_size]
        rebuilt = part if gate is None else gate.reconstruct(part)[0]
        parts.append(proxy(rebuilt, training=False).numpy())
    return np.concatenate(parts)


def pretrain(tf, proxy, fit_x, fit_y, selection_x, selection_y, cfg, seed):
    opt = tf.keras.optimizers.Adam(cfg['learning_rate'])
    update = make_proxy_step(tf, proxy, opt)
    rng = np.random.default_rng(seed)
    best = scores(predict(proxy, None, selection_x), selection_y)['cross_entropy']
    best_weights = proxy.get_weights(); best_epoch = 0; wait = 0; curves = []
    for epoch in range(cfg['pretrain_epochs']):
        order = rng.permutation(len(fit_y))
        for start in range(0, len(order), cfg['batch_size']):
            ix = order[start:start+cfg['batch_size']]
            update(fit_x[ix], fit_y[ix])
        loss = scores(predict(proxy, None, selection_x), selection_y)['cross_entropy']
        curves.append({'epoch': epoch+1, 'selection_cross_entropy': loss})
        if loss < best:
            best = loss; best_weights = proxy.get_weights(); best_epoch = epoch+1; wait = 0
        else:
            wait += 1
        if wait >= cfg['pretrain_patience']:
            break
    proxy.set_weights(best_weights)
    np.testing.assert_allclose(scores(predict(proxy, None, selection_x), selection_y)['cross_entropy'], best, atol=1e-7)
    return {'curve': curves, 'best_epoch': best_epoch, 'best_selection_cross_entropy': best,
            'classifier_optimizer_steps': int(opt.iterations.numpy()), 'best_explicitly_restored': True}


def train_branch(tf, proxy, branch, fit_x, fit_y, selection_x, selection_y, cfg, seed, snapshot_callback=None):
    if branch not in cfg['branches']:
        raise ValueError('Unknown branch')
    gate = build_gate(tf, proxy, branch, cfg, len(fit_y)) if branch in ['task_full', 'global_gate'] else None
    initial = weight_digest(proxy.get_weights())
    opt = make_optimizer(tf, cfg, len(fit_y)); update = make_proxy_step(tf, proxy, opt)
    best = scores(predict(proxy, gate, selection_x), selection_y)['cross_entropy']
    best_proxy = proxy.get_weights(); best_gate = gate.weights() if gate else []
    best_epoch = 0; curves = [{'epoch': 0, 'selection_cross_entropy': best}]
    rng = np.random.default_rng(seed+1000); digest = hashlib.sha256(); audit = None; snapshots = []
    for epoch in range(cfg['collaborative_epochs']):
        order = rng.permutation(len(fit_y)); digest.update(order.astype('<i8').tobytes())
        for start in range(0, len(order), cfg['batch_size']):
            ix = order[start:start+cfg['batch_size']]; first = audit is None
            if first:
                p0 = proxy.get_weights(); h0 = gate.weights() if gate else []
                before = fit_x[ix] if gate is None else gate.reconstruct(fit_x[ix])[0].numpy()
            cached = tf.cast(fit_x[ix], tf.float32) if gate is None else gate.selector_step(fit_x[ix], fit_y[ix])
            if first:
                selector_isolated = equal_weights(p0, proxy.get_weights())
                head_changed = not equal_weights(h0, gate.weights()) if gate else False
                np.testing.assert_allclose(cached.numpy(), before, atol=2e-6, rtol=2e-6)
                h1 = gate.weights() if gate else []; p1 = proxy.get_weights()
            update(cached, fit_y[ix])
            if first:
                audit = {'selector_proxy_unchanged': selector_isolated, 'head_updated': head_changed,
                         'cached_before_head_update': True, 'proxy_phase_head_unchanged': equal_weights(h1, gate.weights() if gate else []),
                         'proxy_updated': not equal_weights(p1, proxy.get_weights())}
                assert all(audit[k] for k in ['selector_proxy_unchanged', 'cached_before_head_update', 'proxy_phase_head_unchanged', 'proxy_updated'])
                assert head_changed == (gate is not None)
        loss = scores(predict(proxy, gate, selection_x), selection_y)['cross_entropy']
        curves.append({'epoch': epoch+1, 'selection_cross_entropy': loss})
        if loss < best:
            best = loss; best_epoch = epoch+1; best_proxy = proxy.get_weights(); best_gate = gate.weights() if gate else []
        if epoch+1 in cfg.get('snapshot_epochs', []):
            record = {'epoch': epoch+1, 'best_epoch': best_epoch, 'best_selection_cross_entropy': best,
                'last_selection_cross_entropy': loss, 'classifier_optimizer_steps': int(opt.iterations.numpy()),
                'batch_order_prefix_sha256': digest.hexdigest(), 'best_proxy_sha256': weight_digest(best_proxy),
                'best_head_sha256': weight_digest(best_gate)}
            snapshots.append(record)
            if snapshot_callback is not None:
                snapshot_callback(record, best_proxy, best_gate, proxy.get_weights(), gate.weights() if gate else [])
        if (epoch+1) % 5 == 0:
            print(f'  seed={seed} {branch} epoch={epoch+1} selection CE={loss:.5f}', flush=True)
    proxy.set_weights(best_proxy)
    if gate:
        gate.restore(best_gate)
    assert weight_digest(proxy.get_weights()) == weight_digest(best_proxy)
    assert weight_digest(gate.weights() if gate else []) == weight_digest(best_gate)
    np.testing.assert_allclose(scores(predict(proxy, gate, selection_x), selection_y)['cross_entropy'], best, atol=1e-7)
    steps = cfg['collaborative_epochs']*int(np.ceil(len(fit_y)/cfg['batch_size']))
    assert int(opt.iterations.numpy()) == steps
    assert (int(gate.head_optimizer.iterations.numpy()) if gate else 0) == (steps if gate else 0)
    return gate, {'branch': branch, 'seed': seed, 'initial_proxy_sha256': initial,
                  'final_proxy_sha256': weight_digest(proxy.get_weights()), 'final_head_sha256': weight_digest(best_gate),
                  'batch_order_sha256': digest.hexdigest(), 'classifier_optimizer_steps': steps,
                  'head_optimizer_steps': steps if gate else 0, 'classifier_parameters': proxy.count_params(),
                  'head_parameters': 33 if branch == 'task_full' else 1 if branch == 'global_gate' else 0,
                  'selected_global_scale': float(tf.sigmoid(gate.logit).numpy()) if branch == 'global_gate' else None,
                  'curve': curves, 'best_epoch': best_epoch, 'best_selection_cross_entropy': best,
                  'snapshots': snapshots,
                  'first_batch_audit': audit, 'coupled_checkpoint_restored': True,
                  'assessment_passed_to_training': False}
