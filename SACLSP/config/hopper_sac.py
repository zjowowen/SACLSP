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
                    hidden_sizes=[11, 128, 128],
                    activation='relu',
                    output_size=3,
                    dropout=0.01,
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
            model_num=2,
            model=dict(
                hidden_sizes=[14, 128, 128],
                activation='relu',
                output_size=1,
                dropout=0.01,
                layernorm=False,
            ),
        ),
        train=dict(
            random_collect_size=1e5,
            gamma=0.98,
            q_lr=0.0001,
            policy_lr=0.0001,
            num_iters=1e4,
            num_episodes=1,
            train_collect_data_num_ratio=20,
            num_epochs=1,
            batch_size=32,
            q_target_parameter_decay=0.95,
            eval_freq=1,
            entropy_coeffi=0.2,
            q_grad_clip=10,
            policy_grad_clip=1,
            weight_decay=0.0001,
        ),
        replay_buffer=dict(
            capacity=1e6,
        ),
    ),
)

cfg = easydict.EasyDict(cfg)

