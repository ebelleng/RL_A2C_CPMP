
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
      # Convert state into a batched tensor (batch size = 1)
      state = tf.expand_dims(state, 0)

      # Run the model and to get action probabilities and critic value
      action_logits_t, value = model(state)

      # Sample next action from the action probability distribution
      action = tf.random.categorical(action_logits_t, 1)
      action_probs_t = tf.nn.softmax(action_logits_t)

      # Store critic values
      values = values.write(t, tf.squeeze(value))

      # Store log probability of the action chosen
      action_probs = action_probs.write(t, action_probs_t[0, action])

      # Apply action to the environment to get next state and reward
      state, reward, done = tf_env_step(action)
      state.set_shape(initial_state_shape)

      # Store reward
      rewards = rewards.write(t, reward)

      if tf.cast(done, tf.bool):
        break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards                                                                
       
  

