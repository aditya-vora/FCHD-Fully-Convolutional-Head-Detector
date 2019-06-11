from pprint import pprint

class Config:
    brainwash_dataset_root_path = './data/'
    hollywood_dataset_root_path = './data/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    caffe_pretrain = True
    caffe_pretrain_path = './data/pretrained_model/vgg16_caffe.pth'
    model_save_path = './checkpoints'  # MODIFIED
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    test_output_path = './output'

    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    env = 'head_detector'  # visdom env
    port = 8097
    plot_every = 2  # vis every N iter

    pretrained_model = 'vgg16'

    epoch = 10

    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}
opt = Config()
