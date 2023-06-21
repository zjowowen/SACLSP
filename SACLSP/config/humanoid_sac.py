import easydict

action_dim=17
obs_dim=376

cfg=dict(
    env=dict(
        name='Humanoid-v4',
    ),
    algo=dict(
        policy=dict(
            model_type='GaussianTanh',
            model=dict(
                mu_model=dict(
                    hidden_sizes=[obs_dim, 512, 512],
                    activation='tanh',
                    output_size=action_dim,
                    dropout=0,
                    layernorm=False,
                    final_activation='tanh',
                    scale=5.0,
                    shrink=0.01,
                ),
                cov=dict(
                    dim=action_dim,
                    functional=True,
                    random_init=False,
                    sigma_lambda=dict(
                        hidden_sizes=[obs_dim, 512, 512],
                        activation='tanh',
                        output_size=action_dim,
                        dropout=0,
                        layernorm=False,
                    ),
                    sigma_offdiag=dict(
                        hidden_sizes=[obs_dim, 512, 512],
                        activation='tanh',
                        output_size=int(action_dim*(action_dim-1)//2),
                        dropout=0,
                        layernorm=False,
                    ),
                ),
            ),
        ),
        q_model=dict(
            model_num=2,
            model=dict(
                hidden_sizes=[obs_dim+action_dim, 512, 512],
                activation='relu',
                output_size=1,
                dropout=0,
                layernorm=False,
            ),
        ),
        train=dict(
            random_collect_size=1e5,
            gamma=0.99,
            q_lr=0.0003,
            policy_lr=0.0001,
            num_iters=1e6,
            num_steps_collected=1,
            num_steps_training=2048,
            num_epochs=1,
            batch_size=2048,
            q_target_parameter_decay=0.99,
            eval_freq=100,
            entropy_coeffi=0.2,
            train_entropy_coeffi=True,
            target_entropy=-20,
            entropy_coeffi_lr=0.001,
            q_grad_clip=2000,
            policy_grad_clip=0.5,
            weight_decay=0.0001,
        ),
        replay_buffer=dict(
            capacity=1e6,
        ),
    ),
)

cfg = easydict.EasyDict(cfg)

