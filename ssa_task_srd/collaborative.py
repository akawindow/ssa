"""TASK-style alternating optimization and its matched frozen-proxy control.

Shared task features are updated ONLY in the proxy phase. The selector phase
updates the retention head only; the proxy phase consumes detached, cached
reconstruction from before the head update, as in the inspected TASK code.
"""
import hashlib
import numpy as np
from .learning import tensorflow,build_proxy,weight_digest,weight_summary


def equal_weights(before,after):
    return len(before)==len(after) and all(np.array_equal(a,b) for a,b in zip(before,after))


def build_alternating_task(tf,proxy,history,config):
    proxy.trainable=True
    extractor=tf.keras.Model(proxy.inputs,proxy.layers[0].output,name='shared_task_features')
    extractor.trainable=True
    bias=float(config.get('selector_bias',4.0))
    max_norm=config.get('selector_max_norm')
    head=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        tf.keras.layers.Dense(1,activation='sigmoid',name='retention_head',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=731),
            bias_initializer=tf.keras.initializers.Constant(bias),
            kernel_constraint=tf.keras.constraints.max_norm(float(max_norm)) if max_norm is not None else None)],name='selector_head')
    selector_input=tf.keras.Input(shape=(history,1))
    selector=tf.keras.Model(selector_input,
        head(extractor(tf.keras.layers.Flatten()(selector_input))),name='shared_task_selector')
    steps=config['selector_epochs']*int(np.ceil(config['n_train']/config['batch_size']))
    def optimizer():
        schedule=tf.keras.optimizers.schedules.CosineDecay(config['learning_rate'],steps,alpha=.9)
        return tf.keras.optimizers.AdamW(schedule,weight_decay=.004,amsgrad=True)

    class AlternatingTask(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.proxy=proxy;self.feature_extractor=extractor
            self.head=head;self.selector=selector
            self.head_optimizer=optimizer();self.proxy_optimizer=optimizer()

        def reconstruct(self,c,training=False):
            c=tf.cast(c,tf.float32);shape=tf.shape(c)
            features=self.feature_extractor(tf.reshape(c,[-1,history]),training=False)
            weights=tf.reshape(self.head(features,training=training),[shape[0],shape[1],1])
            return tf.reduce_sum(c*weights,axis=1),weights

        def call(self,c,training=False):
            rebuilt,_=self.reconstruct(c,training=training)
            return self.proxy(rebuilt,training=False)

        @tf.function
        def selector_step(self,c,y):
            with tf.GradientTape() as tape:
                rebuilt,_=self.reconstruct(c,training=True)
                prediction=self.proxy(rebuilt,training=False)
                loss=tf.reduce_mean(tf.square(prediction-tf.cast(y,tf.float32)))
            variables=self.head.trainable_variables
            gradients=tape.gradient(loss,variables)
            gradients,norm=tf.clip_by_global_norm(gradients,1.0)
            self.head_optimizer.apply_gradients(zip(gradients,variables))
            return loss,tf.stop_gradient(rebuilt),norm

        @tf.function
        def proxy_step(self,cached_reconstruction,y):
            with tf.GradientTape() as tape:
                prediction=self.proxy(tf.stop_gradient(cached_reconstruction),training=True)
                loss=tf.reduce_mean(tf.square(prediction-tf.cast(y,tf.float32)))
            variables=self.proxy.trainable_variables
            gradients=tape.gradient(loss,variables)
            gradients,norm=tf.clip_by_global_norm(gradients,1.0)
            self.proxy_optimizer.apply_gradients(zip(gradients,variables))
            return loss,norm

    return AlternatingTask()


def learn_comparison(config,windows,components):
    tf=tensorflow(config['seed'])
    mode=config['training_mode']
    if mode not in ['paired_frozen','collaborative'] or config.get('selector_mode')!='proxy_features':
        raise ValueError('Alternating comparison requires proxy_features and an explicit comparison mode')
    proxy=build_proxy(tf,config['history'],config['horizon'])
    proxy.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),loss='mse')
    print('Pretraining identical forecasting proxy for the paired comparison...',flush=True)
    pretraining=proxy.fit(windows['train']['reference'],windows['train']['y'],
        validation_data=(windows['val']['reference'],windows['val']['y']),
        epochs=config['proxy_epochs'],batch_size=config['batch_size'],verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['patience'],restore_best_weights=True)])
    initial_proxy=[w.copy() for w in proxy.get_weights()]
    local=dict(config,n_train=len(windows['train']['y']))
    task=build_alternating_task(tf,proxy,config['history'],local)
    initial_head=[w.copy() for w in task.head.get_weights()]
    initial_features=[w.copy() for w in task.feature_extractor.get_weights()]
    initial_summary={};initial_mse={};initial_proxy_predictions={}
    for split,c in components.items():
        rebuilt,w=task.reconstruct(c)
        initial_summary[split]=weight_summary(w.numpy()[...,0])
        initial_proxy_predictions[split]=proxy(rebuilt,training=False).numpy()
        initial_mse[split]=float(np.square(initial_proxy_predictions[split]-windows[split]['y']).mean())
    initial_sigmoid_slope={k:float(np.mean(v*(1-v))) for k,v in
        ((split,task.reconstruct(c)[1].numpy()) for split,c in components.items())}

    histories={'loss':[],'val_loss':[]}
    epoch_rows=[];first_step_audit=None
    rng=np.random.default_rng(config['seed']+1000);order_hash=hashlib.sha256()
    best_loss=np.inf;best_epoch=None;best_pair=None;best_digests=None
    print(f'Training mode={mode}; fixed {config["selector_epochs"]} epochs, shared task features...',flush=True)
    for epoch in range(config['selector_epochs']):
        order=rng.permutation(len(windows['train']['y']));order_hash.update(order.astype('<i8').tobytes())
        head_sum=0.;proxy_sum=0.;head_grad=[];proxy_grad=[]
        for start in range(0,len(order),config['batch_size']):
            ix=order[start:start+config['batch_size']]
            c=components['train'][ix];y=windows['train']['y'][ix]
            first=first_step_audit is None
            if first:
                p0=[w.copy() for w in proxy.get_weights()]
                h0=[w.copy() for w in task.head.get_weights()]
            head_loss,cached,head_norm=task.selector_step(c,y)
            if first:
                head_phase={
                    'proxy_unchanged':equal_weights(p0,proxy.get_weights()),
                    'head_updated':not equal_weights(h0,task.head.get_weights()),
                    'head_gradient_norm':float(head_norm)}
                p1=[w.copy() for w in proxy.get_weights()]
                h1=[w.copy() for w in task.head.get_weights()]
                q1=[w.copy() for w in task.feature_extractor.get_weights()]
            if mode=='collaborative':
                proxy_loss,proxy_norm=task.proxy_step(cached,y)
                proxy_sum+=float(proxy_loss)*len(ix);proxy_grad.append(float(proxy_norm))
            if first:
                proxy_phase={
                    'executed':mode=='collaborative',
                    'head_unchanged':equal_weights(h1,task.head.get_weights()),
                    'proxy_updated':not equal_weights(p1,proxy.get_weights()),
                    'shared_features_updated':not equal_weights(q1,task.feature_extractor.get_weights())}
                first_step_audit={'selector_phase':head_phase,'proxy_phase':proxy_phase}
                if not head_phase['proxy_unchanged'] or not head_phase['head_updated']:
                    raise RuntimeError('Selector phase parameter isolation failed')
                if not proxy_phase['head_unchanged'] or proxy_phase['proxy_updated']!=(mode=='collaborative'):
                    raise RuntimeError('Proxy phase parameter isolation failed')
                if mode=='collaborative' and not proxy_phase['shared_features_updated']:
                    raise RuntimeError('Shared encoder was accidentally frozen')
            head_sum+=float(head_loss)*len(ix);head_grad.append(float(head_norm))
        val_prediction=task(components['val'],training=False).numpy()
        val_loss=float(np.square(val_prediction-windows['val']['y']).mean())
        histories['loss'].append(head_sum/len(order));histories['val_loss'].append(val_loss)
        epoch_rows.append({'epoch':epoch+1,'selector_loss':histories['loss'][-1],'val_reconstructed_proxy_mse_z':val_loss,
            'proxy_step_loss':proxy_sum/len(order) if mode=='collaborative' else None,
            'selector_gradient_norm_mean':float(np.mean(head_grad)),
            'proxy_gradient_norm_mean':float(np.mean(proxy_grad)) if proxy_grad else None})
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch+1
            best_pair=([w.copy() for w in proxy.get_weights()],[w.copy() for w in task.head.get_weights()])
            best_digests={'proxy':weight_digest(best_pair[0]),'head':weight_digest(best_pair[1])}
    # Restore both sides of the same validation-selected checkpoint together.
    proxy.set_weights(best_pair[0]);task.head.set_weights(best_pair[1])
    restored=(weight_digest(proxy.get_weights())==best_digests['proxy'] and weight_digest(task.head.get_weights())==best_digests['head'])
    if not restored:raise RuntimeError('Coupled checkpoint restoration failed')
    proxy_unchanged=equal_weights(initial_proxy,proxy.get_weights())
    head_updated=not equal_weights(initial_head,task.head.get_weights())
    if proxy_unchanged!=(mode=='paired_frozen') or not head_updated:
        raise RuntimeError('Final paired training state did not match requested mode')
    rebuilt={};weights={};diagnostics={}
    for split,c in components.items():
        x,w=task.reconstruct(c);rebuilt[split]=x.numpy();weights[split]=w.numpy()[...,0]
        diagnostics[split]={}
        for name,a in [('reference',windows[split]['reference']),('observed',windows[split]['x']),
            ('fixed_ssa',c[:,:-1,:].sum(axis=1)),('task_ssa',rebuilt[split])]:
            prediction=proxy(a,training=False).numpy()
            diagnostics[split][name+'_mse_z']=float(np.square(prediction-windows[split]['y']).mean())
    training={'proxy_loss':pretraining.history,'selector_loss':histories,'epoch_records':epoch_rows,
        'training_mode':mode,'selector_mode':'proxy_features',
        'selector_bias_initialization':float(config.get('selector_bias',4.0)),
        'selector_max_norm':config.get('selector_max_norm'),
        'initial_sigmoid_slope':initial_sigmoid_slope,
        'final_sigmoid_slope':{k:float(np.mean(v*(1-v))) for k,v in weights.items()},
        'final_head_kernel_norm':float(np.linalg.norm(task.head.get_weights()[0])),
        'proxy_weights_unchanged':proxy_unchanged,'selector_weights_updated':head_updated,
        'selector_head_weights_updated':head_updated,
        'shared_features_updated':not equal_weights(initial_features,task.feature_extractor.get_weights()),
        'selector_parameters':task.selector.count_params(),'proxy_parameters':proxy.count_params(),
        'selector_trainable_parameters':task.head.count_params(),
        'selector_parameter_count_note':'33 parameters updated in selector phase; shared feature parameters update only in proxy phase',
        'selector_step_parameters':task.head.count_params(),
        'proxy_step_parameters':proxy.count_params() if mode=='collaborative' else 0,
        'pretrained_proxy_sha256':weight_digest(initial_proxy),
        'initial_trainable_selector_sha256':weight_digest(initial_head),
        'final_proxy_sha256':weight_digest(proxy.get_weights()),'final_head_sha256':weight_digest(task.head.get_weights()),
        'batch_order_sha256':order_hash.hexdigest(),
        'selector_optimizer_steps':int(task.head_optimizer.iterations.numpy()),
        'proxy_optimizer_steps':int(task.proxy_optimizer.iterations.numpy()),
        'fixed_epoch_budget':config['selector_epochs'],'best_epoch':best_epoch,'best_validation_proxy_mse_z':best_loss,
        'checkpoint_rule':'Best reconstructed validation proxy MSE among epochs 1..N; restore proxy and head together',
        'coupled_checkpoint_restored':restored,'first_batch_phase_audit':first_step_audit,
        'initial_weight_summary':initial_summary,'final_weight_summary':{k:weight_summary(v) for k,v in weights.items()},
        'initial_selector_proxy_mse_z':initial_mse,'proxy_diagnostics':diagnostics,
        'optimizer':'Separate AdamW, weight_decay=.004, AMSGrad=True, cosine alpha=.9, global gradient clip=1',
        'supervision':'Future forecast targets; no clean-history reconstruction loss; no SRD gradients',
        'proxy_pretraining_input':'Original reference histories, not artificially contaminated histories'}
    if config.get('raw_proxy_control',False):
        # Budget-matched proxy adaptation on raw inputs. Run after the main task
        # so this diagnostic cannot affect its initialization or optimization.
        control=build_proxy(tf,config['history'],config['horizon'])
        control.set_weights(initial_proxy)
        steps=config['selector_epochs']*int(np.ceil(len(windows['train']['x'])/config['batch_size']))
        schedule=tf.keras.optimizers.schedules.CosineDecay(config['learning_rate'],steps,alpha=.9)
        opt=tf.keras.optimizers.AdamW(schedule,weight_decay=.004,amsgrad=True)
        @tf.function
        def raw_step(x,y):
            with tf.GradientTape() as tape:
                loss=tf.reduce_mean(tf.square(control(x,training=True)-tf.cast(y,tf.float32)))
            grad=tape.gradient(loss,control.trainable_variables)
            grad,_=tf.clip_by_global_norm(grad,1.0)
            opt.apply_gradients(zip(grad,control.trainable_variables))
        initial_control=weight_digest(control.get_weights())
        baseline_pred=control(windows['val']['x'],training=False).numpy()
        rng_control=np.random.default_rng(config['seed']+1000)
        control_order=hashlib.sha256();best_control=np.inf;best_control_epoch=None
        for epoch in range(config['selector_epochs']):
            order=rng_control.permutation(len(windows['train']['x']));control_order.update(order.astype('<i8').tobytes())
            for start in range(0,len(order),config['batch_size']):
                ix=order[start:start+config['batch_size']]
                raw_step(windows['train']['x'][ix],windows['train']['y'][ix])
            prediction=control(windows['val']['x'],training=False).numpy()
            loss=float(np.square(prediction-windows['val']['y']).mean())
            if loss<best_control:
                best_control=loss;best_control_epoch=epoch+1
        assert initial_control==weight_digest(initial_proxy) and control_order.hexdigest()==order_hash.hexdigest()
        training['raw_proxy_control']={'initial_proxy_sha256':initial_control,
            'initial_validation_mse_z':float(np.square(baseline_pred-windows['val']['y']).mean()),
            'best_validation_mse_z':best_control,'best_epoch':best_control_epoch,
            'optimizer_steps':int(opt.iterations.numpy()),'batch_order_sha256':control_order.hexdigest(),
            'scope':'Raw histories, no selector; same proxy initialization, batches, optimizer and 1..N checkpoint budget; diagnostic, no new denoiser'}
    return rebuilt,weights,training,proxy,task.selector
