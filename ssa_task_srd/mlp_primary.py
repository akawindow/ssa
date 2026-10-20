"""Same-architecture MLP controls; model selection never receives assessment data."""
import hashlib
import numpy as np
from .collaborative import build_alternating_task, equal_weights
from .learning import weight_digest


def optimizer(tf, config, n_train):
    steps=config['selector_epochs']*int(np.ceil(n_train/config['batch_size']))
    schedule=tf.keras.optimizers.schedules.CosineDecay(config['learning_rate'],steps,alpha=.9)
    return tf.keras.optimizers.AdamW(schedule,weight_decay=.004,amsgrad=True)


def proxy_step(tf, proxy, opt):
    @tf.function
    def step(x,y):
        with tf.GradientTape() as tape:
            loss=tf.reduce_mean(tf.square(proxy(tf.stop_gradient(x),training=True)-tf.cast(y,tf.float32)))
        grads=tape.gradient(loss,proxy.trainable_variables)
        grads,norm=tf.clip_by_global_norm(grads,1.)
        opt.apply_gradients(zip(grads,proxy.trainable_variables))
        return loss,norm
    return step


def build_global_gate(tf, proxy, config, n_train):
    class GlobalGate(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.proxy=proxy
            self.logit=self.add_weight(name='global_logit',shape=(),initializer='zeros',trainable=True)
            self.head_optimizer=optimizer(tf,config,n_train)
            self.proxy_optimizer=optimizer(tf,config,n_train)
            self.proxy_step=proxy_step(tf,proxy,self.proxy_optimizer)

        def reconstruct(self,x,training=False):
            scale=tf.sigmoid(self.logit)
            return tf.cast(x,tf.float32)*scale,scale

        @tf.function
        def selector_step(self,x,y):
            with tf.GradientTape() as tape:
                rebuilt,_=self.reconstruct(x)
                loss=tf.reduce_mean(tf.square(self.proxy(rebuilt,training=False)-tf.cast(y,tf.float32)))
            grads=tape.gradient(loss,[self.logit])
            grads,norm=tf.clip_by_global_norm(grads,1.)
            self.head_optimizer.apply_gradients(zip(grads,[self.logit]))
            return loss,tf.stop_gradient(rebuilt),norm
    return GlobalGate()


def gate_weights(gate, branch):
    if gate is None:return []
    if branch=='global_gate':return [np.array(gate.logit.numpy(),copy=True)]
    return [a.copy() for a in gate.head.get_weights()]


def restore_gate(gate, branch, weights):
    if gate is None:return
    if branch=='global_gate':gate.logit.assign(weights[0])
    else:gate.head.set_weights(weights)


def predict(proxy, gate, x):
    rebuilt=x if gate is None else gate.reconstruct(x)[0]
    return proxy(rebuilt,training=False).numpy()


def train_branch(tf, proxy, branch, fit_x, fit_y, selection_x, selection_y, config):
    """Only fit and checkpoint-selection data enter this training function."""
    initial_proxy=weight_digest(proxy.get_weights())
    proxy.trainable=True
    gate=None
    if branch=='task_full':
        gate=build_alternating_task(tf,proxy,config['history'],dict(config,n_train=len(fit_y)))
    elif branch=='global_gate':
        gate=build_global_gate(tf,proxy,config,len(fit_y))
    elif branch not in ['raw_continue','half_continue','ssa7_continue']:
        raise ValueError('Unknown MLP branch')
    opt=gate.proxy_optimizer if gate is not None else optimizer(tf,config,len(fit_y))
    update=gate.proxy_step if gate is not None else proxy_step(tf,proxy,opt)
    assert initial_proxy==weight_digest(proxy.get_weights())
    initial_head=weight_digest(gate_weights(gate,branch))
    best_loss=float(np.square(predict(proxy,gate,selection_x)-selection_y).mean())
    best_epoch=0
    best_proxy=[a.copy() for a in proxy.get_weights()]
    best_gate=gate_weights(gate,branch)
    curves=[{'epoch':0,'selection_mse_z':best_loss,'fit_online_mse_z':None}]
    rng=np.random.default_rng(config['seed']+1000);order_digest=hashlib.sha256()
    first_audit=None
    for epoch in range(config['selector_epochs']):
        order=rng.permutation(len(fit_y));order_digest.update(order.astype('<i8').tobytes())
        fit_loss=0.
        for start in range(0,len(order),config['batch_size']):
            ix=order[start:start+config['batch_size']];first=first_audit is None
            if first:
                p0=[a.copy() for a in proxy.get_weights()]
                h0=gate_weights(gate,branch)
                before=fit_x[ix] if gate is None else gate.reconstruct(fit_x[ix])[0].numpy()
            if gate is not None:
                _,cached,_=gate.selector_step(fit_x[ix],fit_y[ix])
            else:cached=tf.cast(fit_x[ix],tf.float32)
            if first:
                selector_proxy_unchanged=equal_weights(p0,proxy.get_weights())
                selector_head_changed=not equal_weights(h0,gate_weights(gate,branch)) if gate is not None else False
                np.testing.assert_allclose(cached.numpy(),before,atol=1e-6,rtol=1e-6)
                h1=gate_weights(gate,branch)
                p1=[a.copy() for a in proxy.get_weights()]
            loss,_=update(cached,fit_y[ix]);fit_loss+=float(loss)*len(ix)
            if first:
                first_audit={'selector_phase_proxy_unchanged':selector_proxy_unchanged,
                    'selector_head_updated':selector_head_changed,'cached_pre_head_update_reconstruction':True,
                    'proxy_phase_head_unchanged':equal_weights(h1,gate_weights(gate,branch)),
                    'proxy_updated':not equal_weights(p1,proxy.get_weights())}
                assert first_audit['selector_phase_proxy_unchanged'] and first_audit['proxy_phase_head_unchanged']
                assert first_audit['proxy_updated'] and selector_head_changed==(gate is not None)
        selection_loss=float(np.square(predict(proxy,gate,selection_x)-selection_y).mean())
        curves.append({'epoch':epoch+1,'selection_mse_z':selection_loss,'fit_online_mse_z':fit_loss/len(order)})
        if selection_loss<best_loss:
            best_loss=selection_loss;best_epoch=epoch+1
            best_proxy=[a.copy() for a in proxy.get_weights()];best_gate=gate_weights(gate,branch)
    proxy.set_weights(best_proxy);restore_gate(gate,branch,best_gate)
    restored=weight_digest(proxy.get_weights())==weight_digest(best_proxy) and weight_digest(gate_weights(gate,branch))==weight_digest(best_gate)
    assert restored
    expected=config['selector_epochs']*int(np.ceil(len(fit_y)/config['batch_size']))
    assert int(opt.iterations.numpy())==expected
    head_steps=int(gate.head_optimizer.iterations.numpy()) if gate is not None else 0
    assert head_steps==(expected if gate is not None else 0)
    diagnostics={'branch':branch,'seed':config['seed'],'initial_proxy_sha256':initial_proxy,
        'initial_head_sha256':initial_head,'final_proxy_sha256':weight_digest(proxy.get_weights()),
        'final_head_sha256':weight_digest(gate_weights(gate,branch)),
        'batch_order_sha256':order_digest.hexdigest(),'proxy_optimizer_steps':expected,'head_optimizer_steps':head_steps,
        'proxy_parameters':proxy.count_params(),'additional_head_parameters':33 if branch=='task_full' else 1 if branch=='global_gate' else 0,
        'best_epoch':best_epoch,'best_selection_mse_z':best_loss,'checkpoint_includes_epoch_zero':True,
        'coupled_checkpoint_restored':restored,'first_batch_audit':first_audit,'curve':curves,
        'assessment_data_passed_to_training':False}
    if branch=='global_gate':diagnostics['selected_global_scale']=float(tf.sigmoid(gate.logit).numpy())
    if branch=='task_full':diagnostics['selected_head_kernel_norm']=float(np.linalg.norm(gate.head.get_weights()[0]))
    return gate,diagnostics
