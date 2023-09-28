import torch

from timm.models.layers import to_2tuple

def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor(torch.nn.Module):
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model, return_nodes={}, range_norm=False, use_input_norm=True):
        super().__init__()
        self.model = model
        self.return_nodes = return_nodes
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.hook_handlers = []
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []
            if key not in self.return_nodes:
                self.return_nodes[key] = []

        self._register_hooks()

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.model(x)

    def _register_hooks(self, **kwargs):
        img_size = to_2tuple(self.model.patch_embed.img_size)
        patch_size = to_2tuple(self.model.patch_embed.patch_size)

        resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        for block_idx, block in enumerate(self.model.blocks):

            if block_idx in self.return_nodes[SwinExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook(resolution)))
            if block_idx in self.return_nodes[SwinExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.return_nodes[SwinExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook(block.attn.num_heads, resolution)))
            if block_idx in self.return_nodes[SwinExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _init_hooks_data(self):
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _get_block_hook(self, input_resolution):
        def _get_block_output(model, input, output):
            B, N, C = input[0].shape
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output[:, 1:, :].reshape(B, *input_resolution, C))

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self, num_heads, input_resolution):
        def _get_qkv_output(model, inp, output):
            B, N, C = inp[0].shape
            self.outputs_dict[VitExtractor.QKV_KEY].append(output.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)[:, :, :, 1:, :].reshape(3, B, num_heads, *input_resolution, C // num_heads))

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_layer_feature(self):  # List([B, N, D])
        feature = self.outputs_dict[VitExtractor.LAYER_KEY]
        return feature

    def get_block_feature(self):  # List([B, N, D])
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        return feature

    def get_qkv_feature(self):
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        return feature

    def get_attn_feature(self):
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        return feature

    def get_patch_size(self):
        return self.model.patch_size

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_queries_from_qkv(self, qkv):
        return qkv[0]

    def get_keys_from_qkv(self, qkv):
        return qkv[1]

    def get_values_from_qkv(self, qkv):
        return qkv[2]

    def get_keys(self, block_idx):
        qkv_features = self.get_qkv_feature()[self.return_nodes[VitExtractor.QKV_KEY].index(block_idx)]
        keys = self.get_keys_from_qkv(qkv_features)
        return keys

    def get_keys_self_sim(self, block_idx):
        keys = self.get_keys(block_idx)
        N, heads, h, w, d = keys.shape
        concatenated_keys = keys.view(N, heads, h * w, d).transpose(1, 2).reshape(N, h * w, heads * d)
        ssim_map = attn_cosine_sim2(concatenated_keys)
        return ssim_map.reshape(N, h, w, h, w)

    def get_keys_for_all(self):
        results = [self.get_keys(block_idx) for block_idx in self.return_nodes[VitExtractor.QKV_KEY]]
        return results

    def get_keys_self_sim_for_all(self):
        results = [self.get_keys_self_sim(block_idx) for block_idx in self.return_nodes[VitExtractor.QKV_KEY]]
        return results


def attn_cosine_sim2(x, eps=1e-08):
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class SwinExtractor(torch.nn.Module):
    LAYER_KEY = 'layer'
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [LAYER_KEY, BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model, return_nodes={}, range_norm=False, use_input_norm=True):
        super().__init__()
        self.model = model
        self.return_nodes = return_nodes
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.hook_handlers = []
        self.outputs_dict = {}
        for key in SwinExtractor.KEY_LIST:
            self.outputs_dict[key] = []
            if key not in self.return_nodes:
                self.return_nodes[key] = []

        self._register_hooks()

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.model(x)

    def _register_hooks(self, **kwargs):
        block_idx = 0
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx in self.return_nodes[SwinExtractor.LAYER_KEY]:
                self.hook_handlers.append(layer.register_forward_hook(self._get_layer_hook()))

            for block in layer.blocks:
                if block_idx in self.return_nodes[SwinExtractor.BLOCK_KEY]:
                    self.hook_handlers.append(block.register_forward_hook(self._get_block_hook(block.input_resolution)))
                if block_idx in self.return_nodes[SwinExtractor.ATTN_KEY]:
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
                if block_idx in self.return_nodes[SwinExtractor.QKV_KEY]:
                    self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook(block.attn.num_heads, block.input_resolution)))
                if block_idx in self.return_nodes[SwinExtractor.PATCH_IMD_KEY]:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

                block_idx += 1

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _init_hooks_data(self):
        for key in SwinExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _get_layer_hook(self):
        def _get_layer_output(model, input, output):
            self.outputs_dict[SwinExtractor.LAYER_KEY].append(output)

        return _get_layer_output

    def _get_block_hook(self, input_resolution):
        def _get_block_output(model, input, output):
            B, N, C = input[0].shape
            self.outputs_dict[SwinExtractor.BLOCK_KEY].append(output.reshape(B, *input_resolution, C))

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[SwinExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self, num_heads, input_resolution):
        def _get_qkv_output(model, inp, output):
            B, N, C = inp[0].shape
            self.outputs_dict[SwinExtractor.QKV_KEY].append(output.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4).reshape(3, B, num_heads, *input_resolution, C // num_heads))

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[SwinExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_layer_feature(self):  # List([B, N, D])
        feature = self.outputs_dict[SwinExtractor.LAYER_KEY]
        return feature

    def get_block_feature(self):  # List([B, N, D])
        feature = self.outputs_dict[SwinExtractor.BLOCK_KEY]
        return feature

    def get_qkv_feature(self):
        feature = self.outputs_dict[SwinExtractor.QKV_KEY]
        return feature

    def get_attn_feature(self):
        feature = self.outputs_dict[SwinExtractor.ATTN_KEY]
        return feature

    def get_patch_size(self):
        return self.model.patch_size

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_queries_from_qkv(self, qkv):
        return qkv[0]

    def get_keys_from_qkv(self, qkv):
        return qkv[1]

    def get_values_from_qkv(self, qkv):
        return qkv[2]

    def get_keys(self, block_idx):
        qkv_features = self.get_qkv_feature()[self.return_nodes[SwinExtractor.QKV_KEY].index(block_idx)]
        keys = self.get_keys_from_qkv(qkv_features)
        return keys

    def get_keys_self_sim(self, block_idx):
        keys = self.get_keys(block_idx)
        N, heads, h, w, d = keys.shape
        concatenated_keys = keys.view(N, heads, h * w, d).transpose(1, 2).reshape(N, h * w, heads * d)
        ssim_map = attn_cosine_sim2(concatenated_keys)
        return ssim_map.reshape(N, h, w, h, w)

    def get_keys_for_all(self):
        results = [self.get_keys(block_idx) for block_idx in self.return_nodes[SwinExtractor.QKV_KEY]]
        return results

    def get_keys_self_sim_for_all(self):
        results = [self.get_keys_self_sim(block_idx) for block_idx in self.return_nodes[SwinExtractor.QKV_KEY]]
        return results