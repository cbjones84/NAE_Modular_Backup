"""
Auto-implemented improvement from GitHub
Source: yptsang/DeepPack3D/agent.py
Implemented: 2025-12-09T11:11:06.540889
Usefulness Score: 100
Keywords: def , class , optimize, tensorflow, model, train, fit, gradient, loss, var, size, loss
"""

# Original source: yptsang/DeepPack3D
# Path: agent.py


# Function: select
def select(self, state):
        items, h_maps, actions = state
        action_space = indices(actions)
        
#         print('actions: ', len(action_space))
        
        r = np.random.random()
        if r < self.eps:
            action = action_space[np.random.choice(len(action_space))]
        else:
            q = self.Q(state)
            action = action_space[np.argmax(q)]
            r = np.max(q)
            
        return action, r
    
    def Q_inputs(self, state, action=None):
        W, H, D = self.env.size
        
        items, h_maps, actions = state
        if action is None:
            action_space = indices(actions)
        else:
            i, j, k = action
            action_space = [(i, j, k)]
            
        imaps = [self.env.i_map(i, items) for i in range(len(self.env.packers))]
            
        hmap_in = []
        amap_in = []
        imap_in = []
        
        # item, bin, rotation_placement
        for i, j, k in action_space:
            _, (x, y, z), (w, h, d), _ = actions[i][j][k]
            amap = self.env.p_map(j, (x, y, z, w, h, d))
            amap = np.where(amap == 0, h_maps[j], y + h) / H

            hmap = np.full(h_maps[j].shape, np.amax(amap))

            imap = imaps[j][np.arange(len(items)) != i]
#             print(hmap, amap, imap)

            hmap_in.append(hmap)
            amap_in.append(amap)
            imap_in.append(imap)
            
        hmap_in, amap_in, imap_in = map(np.asarray, (hmap_in, amap_in, imap_in))
        hmap_in = hmap_in[..., None]
        amap_in = amap_in[..., None]
        const_in = np.ones(hmap_in.shape)
        
        return [const_in, hmap_in, amap_in, imap_in]
    
    def Q(self, state, action=None):
        const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
        
        batch_size = self.batch_size
        sections = np.cumsum([self.batch_size] * int(np.ceil(const_in.shape[0] / batch_size) - 1))
        batches = map(lambda data: map(lambda x: x.copy(), np.split(data, sections, axis=0)), (const_in, hmap_in, amap_in, imap_in))
        
        outputs = []
        for const_in, hmap_in, amap_in, imap_in in zip(*batches):
            # print(const_in.shape)
            q = self.q_net([const_in, hmap_in, amap_in, imap_in])
#             print('const_in.shape')
            outputs.append(q)
        q = np.concatenate(outputs, axis=0)
#         print(q.shape)
        return q

    def lr_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.learning_rate * (0.5 ** (epoch / self.lr_drop))
        return max(self.lr_min, lr)
            
    def train(self, history):
        q_inputs = []
        q_targets = []
        
        for state, action, next_state, reward, done in history:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            if done:
                q_target = reward
            else:
#                 print(np.amax(self.Q(next_state)))
                q_target = reward + self.gamma * np.amax(self.Q(next_state))
#                 print(q_target)
            q_targets.append([q_target])
            
        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs = [np.asarray(const_in), np.asarray(hmap_in), np.asarray(amap_in), np.asarray(imap_in)]
        q_targets = np.asarray(q_targets)
#         print('q_targets', q_targets)
#         print([result for result in map(lambda inps: (inps.shape, np.amin(inps), np.amax(inps), np.mean(inps)), q_inputs)], q_targets)
        return self.fit(q_inputs, q_targets)
    
    def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
#             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
#             print('q', q)
#             print('loss', loss)
            
#         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)
        
#         gradient clipping
#         if self.epoch < self.warmup_epochs:
#             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]
        
        self.q_optimizer.apply_gradients(zip(grad, self.q_net_target.trainable_variables))
        
        self.q_optimizer.lr.assign(self.lr_scheduler(self.epoch))
        
        self.epoch += 1
    
#         print([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net.set_weights([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        return loss
    
    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.lr.numpy() if self.q_optimizer is not None else None}')



# Function: Q_inputs
def Q_inputs(self, state, action=None):
        W, H, D = self.env.size
        
        items, h_maps, actions = state
        if action is None:
            action_space = indices(actions)
        else:
            i, j, k = action
            action_space = [(i, j, k)]
            
        imaps = [self.env.i_map(i, items) for i in range(len(self.env.packers))]
            
        hmap_in = []
        amap_in = []
        imap_in = []
        
        # item, bin, rotation_placement
        for i, j, k in action_space:
            _, (x, y, z), (w, h, d), _ = actions[i][j][k]
            amap = self.env.p_map(j, (x, y, z, w, h, d))
            amap = np.where(amap == 0, h_maps[j], y + h) / H

            hmap = np.full(h_maps[j].shape, np.amax(amap))

            imap = imaps[j][np.arange(len(items)) != i]
#             print(hmap, amap, imap)

            hmap_in.append(hmap)
            amap_in.append(amap)
            imap_in.append(imap)
            
        hmap_in, amap_in, imap_in = map(np.asarray, (hmap_in, amap_in, imap_in))
        hmap_in = hmap_in[..., None]
        amap_in = amap_in[..., None]
        const_in = np.ones(hmap_in.shape)
        
        return [const_in, hmap_in, amap_in, imap_in]
    
    def Q(self, state, action=None):
        const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
        
        batch_size = self.batch_size
        sections = np.cumsum([self.batch_size] * int(np.ceil(const_in.shape[0] / batch_size) - 1))
        batches = map(lambda data: map(lambda x: x.copy(), np.split(data, sections, axis=0)), (const_in, hmap_in, amap_in, imap_in))
        
        outputs = []
        for const_in, hmap_in, amap_in, imap_in in zip(*batches):
            # print(const_in.shape)
            q = self.q_net([const_in, hmap_in, amap_in, imap_in])
#             print('const_in.shape')
            outputs.append(q)
        q = np.concatenate(outputs, axis=0)
#         print(q.shape)
        return q

    def lr_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.learning_rate * (0.5 ** (epoch / self.lr_drop))
        return max(self.lr_min, lr)
            
    def train(self, history):
        q_inputs = []
        q_targets = []
        
        for state, action, next_state, reward, done in history:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            if done:
                q_target = reward
            else:
#                 print(np.amax(self.Q(next_state)))
                q_target = reward + self.gamma * np.amax(self.Q(next_state))
#                 print(q_target)
            q_targets.append([q_target])
            
        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs = [np.asarray(const_in), np.asarray(hmap_in), np.asarray(amap_in), np.asarray(imap_in)]
        q_targets = np.asarray(q_targets)
#         print('q_targets', q_targets)
#         print([result for result in map(lambda inps: (inps.shape, np.amin(inps), np.amax(inps), np.mean(inps)), q_inputs)], q_targets)
        return self.fit(q_inputs, q_targets)
    
    def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
#             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
#             print('q', q)
#             print('loss', loss)
            
#         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)
        
#         gradient clipping
#         if self.epoch < self.warmup_epochs:
#             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]
        
        self.q_optimizer.apply_gradients(zip(grad, self.q_net_target.trainable_variables))
        
        self.q_optimizer.lr.assign(self.lr_scheduler(self.epoch))
        
        self.epoch += 1
    
#         print([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net.set_weights([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        return loss
    
    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.lr.numpy() if self.q_optimizer is not None else None}')



# Function: Q
def Q(self, state, action=None):
        const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
        
        batch_size = self.batch_size
        sections = np.cumsum([self.batch_size] * int(np.ceil(const_in.shape[0] / batch_size) - 1))
        batches = map(lambda data: map(lambda x: x.copy(), np.split(data, sections, axis=0)), (const_in, hmap_in, amap_in, imap_in))
        
        outputs = []
        for const_in, hmap_in, amap_in, imap_in in zip(*batches):
            # print(const_in.shape)
            q = self.q_net([const_in, hmap_in, amap_in, imap_in])
#             print('const_in.shape')
            outputs.append(q)
        q = np.concatenate(outputs, axis=0)
#         print(q.shape)
        return q

    def lr_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.learning_rate * (0.5 ** (epoch / self.lr_drop))
        return max(self.lr_min, lr)
            
    def train(self, history):
        q_inputs = []
        q_targets = []
        
        for state, action, next_state, reward, done in history:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            if done:
                q_target = reward
            else:
#                 print(np.amax(self.Q(next_state)))
                q_target = reward + self.gamma * np.amax(self.Q(next_state))
#                 print(q_target)
            q_targets.append([q_target])
            
        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs = [np.asarray(const_in), np.asarray(hmap_in), np.asarray(amap_in), np.asarray(imap_in)]
        q_targets = np.asarray(q_targets)
#         print('q_targets', q_targets)
#         print([result for result in map(lambda inps: (inps.shape, np.amin(inps), np.amax(inps), np.mean(inps)), q_inputs)], q_targets)
        return self.fit(q_inputs, q_targets)
    
    def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
#             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
#             print('q', q)
#             print('loss', loss)
            
#         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)
        
#         gradient clipping
#         if self.epoch < self.warmup_epochs:
#             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]
        
        self.q_optimizer.apply_gradients(zip(grad, self.q_net_target.trainable_variables))
        
        self.q_optimizer.lr.assign(self.lr_scheduler(self.epoch))
        
        self.epoch += 1
    
#         print([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net.set_weights([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        return loss
    
    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.lr.numpy() if self.q_optimizer is not None else None}')



# Function: lr_scheduler
def lr_scheduler(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.learning_rate * (0.5 ** (epoch / self.lr_drop))
        return max(self.lr_min, lr)
            
    def train(self, history):
        q_inputs = []
        q_targets = []
        
        for state, action, next_state, reward, done in history:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            if done:
                q_target = reward
            else:
#                 print(np.amax(self.Q(next_state)))
                q_target = reward + self.gamma * np.amax(self.Q(next_state))
#                 print(q_target)
            q_targets.append([q_target])
            
        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs = [np.asarray(const_in), np.asarray(hmap_in), np.asarray(amap_in), np.asarray(imap_in)]
        q_targets = np.asarray(q_targets)
#         print('q_targets', q_targets)
#         print([result for result in map(lambda inps: (inps.shape, np.amin(inps), np.amax(inps), np.mean(inps)), q_inputs)], q_targets)
        return self.fit(q_inputs, q_targets)
    
    def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
#             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
#             print('q', q)
#             print('loss', loss)
            
#         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)
        
#         gradient clipping
#         if self.epoch < self.warmup_epochs:
#             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]
        
        self.q_optimizer.apply_gradients(zip(grad, self.q_net_target.trainable_variables))
        
        self.q_optimizer.lr.assign(self.lr_scheduler(self.epoch))
        
        self.epoch += 1
    
#         print([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net.set_weights([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        return loss
    
    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.lr.numpy() if self.q_optimizer is not None else None}')



# Function: train
def train(self, history):
        q_inputs = []
        q_targets = []
        
        for state, action, next_state, reward, done in history:
            const_in, hmap_in, amap_in, imap_in = self.Q_inputs(state, action)
            q_inputs.append([const_in[0], hmap_in[0], amap_in[0], imap_in[0]])
            if done:
                q_target = reward
            else:
#                 print(np.amax(self.Q(next_state)))
                q_target = reward + self.gamma * np.amax(self.Q(next_state))
#                 print(q_target)
            q_targets.append([q_target])
            
        const_in, hmap_in, amap_in, imap_in = zip(*q_inputs)
        q_inputs = [np.asarray(const_in), np.asarray(hmap_in), np.asarray(amap_in), np.asarray(imap_in)]
        q_targets = np.asarray(q_targets)
#         print('q_targets', q_targets)
#         print([result for result in map(lambda inps: (inps.shape, np.amin(inps), np.amax(inps), np.mean(inps)), q_inputs)], q_targets)
        return self.fit(q_inputs, q_targets)
    
    def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
#             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
#             print('q', q)
#             print('loss', loss)
            
#         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)
        
#         gradient clipping
#         if self.epoch < self.warmup_epochs:
#             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]
        
        self.q_optimizer.apply_gradients(zip(grad, self.q_net_target.trainable_variables))
        
        self.q_optimizer.lr.assign(self.lr_scheduler(self.epoch))
        
        self.epoch += 1
    
#         print([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net.set_weights([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        return loss
    
    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.lr.numpy() if self.q_optimizer is not None else None}')



# Function: fit
def fit(self, q_inputs, q_targets):
        with tf.GradientTape() as tape:
#             print(q_inputs)
            q = self.q_net_target(q_inputs)
            # print(tf.keras.losses.MeanSquaredError()(q_targets, q))
            loss = tf.reduce_mean(tf.square(q_targets - q))
            loss = tf.keras.losses.MeanSquaredError()(q_targets, q)
#             print('q', q)
#             print('loss', loss)
            
#         print(list(zip(q, q_targets)))
        grad = tape.gradient(loss, self.q_net_target.trainable_variables)
        
#         gradient clipping
#         if self.epoch < self.warmup_epochs:
#             grad = [tf.clip_by_value(value, -1e-5, 1e-5) for value in grad]
        
        self.q_optimizer.apply_gradients(zip(grad, self.q_net_target.trainable_variables))
        
        self.q_optimizer.lr.assign(self.lr_scheduler(self.epoch))
        
        self.epoch += 1
    
#         print([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        if self.epoch % self.update_epochs == 0:
            print('update')
            self.q_net.set_weights([a * 0.5 + b * (1 - 0.5) for a, b in zip(self.q_net.get_weights(), self.q_net_target.get_weights())])
        return loss
    
    def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.lr.numpy() if self.q_optimizer is not None else None}')



# Function: run
def run(self, max_ep=1, verbose=False, train=None):
        if train is None:
            train = self.__train
            
        iters = (i for i, _ in enumerate(iter(bool, True))) if max_ep == -1 else range(max_ep)

        for ep in iters:
            if verbose:
                print(f'ep {ep}:')
                
            state = self.env.reset()
            ep_reward = 0
            
            history = []
            
            for step in itertools.count():
                if verbose:
                    print(f'\nstep {step}')
                    
                items, h_map, actions = state
                if len(actions) == 0: raise Exception('0 actions')
                action, r = self.select(state)
                
                if verbose:
                    print(f'possible actions: {len(actions)}')
                    print(f'action: {action}')
                    print(f'placement: {actions[action[0]][action[1]][action[2]]}')
                
                yield actions[action[0]][action[1]][action[2]]
                
                next_state, reward, done = self.env.step(action)
                
                history.append((state, action, next_state, reward, done))
                
                if self.visualize:
                    for i, packer in enumerate(self.env.packers):
                        packer.render().savefig(f'./outputs/{ep}_{step}_{i}.jpg')
                    
                ep_reward += reward
                if done:
                    break
                state = next_state
                
            loss = None
            if train:
                self.memory.extend(history)
                if len(self.memory) > 1000:
                    print('update model')
                    history = [self.memory[i] for i in np.random.choice(len(self.memory), 128)]
                    loss = self.train(history)
            
            self.ep_history.append(([packer.space_utilization() for packer in self.env.used_packers], self.env.used_bins, ep_reward))
            
            yield None
            
            utils = [round(packer.space_utilization() * 100, 2) for packer in self.env.used_packers]
            if self.verbose: print(f'Episode {ep}, util: {utils}, used bins: {self.env.used_bins}, ep_reward: {ep_reward:.2f}, memory: {len(self.memory) if self.memory is not None else None}, eps: {self.eps:.2f}, loss: {loss}, lr: {self.q_optimizer.lr.numpy() if self.q_optimizer is not None else None}')


