import easydict

cfg=dict(
    env=dict(
        name='Hopper-v3',
    ),
    algo=dict(
        policy=dict(
            model_type='GaussianTanh',
            model=dict(
                mu_model=dict(
                    hidden_sizes=[11, 128, 128, 32],
                    activation='softplus',
                    output_size=3,
                    dropout=0,
                    layernorm=False,
                    final_activation='tanh',
                    scale=5.0,
                ),
                cov=dict(
                    dim=3,
                    random_init=False,
                ),
            ),
        ),
        q_model=dict(
            model=dict(
                hidden_sizes=[14, 128, 128],
                activation='softplus',
                output_size=1,
                dropout=0,
                layernorm=False,
            ),
        ),
        train=dict(
            random_collect_size=1e4,
            gamma=0.98,
            q_lr=0.0001,
            policy_lr=0.0001,
            num_iters=1e6,
            num_episodes=10,
            train_collect_data_num_ratio=5,
            num_epochs=1,
            batch_size=256,
            q_target_parameter_decay=0.995,
            eval_freq=1,
            entropy_coeffi=0.2,
            grad_clip=10,
            weight_decay=0.001,
        ),
        replay_buffer=dict(
            capacity=1e5,
        ),

    ),
)

cfg = easydict.EasyDict(cfg)

