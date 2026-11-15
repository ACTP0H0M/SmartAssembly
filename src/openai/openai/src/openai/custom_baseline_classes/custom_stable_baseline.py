# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# https://medium.com/@akhileshgogikar/applied-rl-customizing-neural-networks-for-rl-policies-a5a9e2cf763e
import torch as th
import torch.nn as nn
import gym
from gym import spaces
import torchvision  # for imagenet only

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class HexaglideCustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            print("key", key, "shape", subspace.shape)
            if key == "current_image3" or key == "previous_image3":
                model_conv = torchvision.models.resnet50(pretrained=True)  # pretrained on ImageNet
                for param in model_conv.parameters():
                    param.requires_grad = True
                num_ftrs = model_conv.fc.in_features
                model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Linear(128, 64)) # previously also had: nn.ReLU(), nn.Linear(32, 3)
                # model_conv.load_state_dict(th.load("./models/feature_extractor_img_frozen_resnet18_clf_pin_left_right_cam1.pt")) # ToDo: Is this needed?
                extractors[key] = model_conv
                total_concat_size += 64
            elif key == "pressure":
                extractors[key] = nn.Sequential(nn.Flatten(start_dim=1))
                total_concat_size += 1
            elif key == "pose":
                extractors[key] = nn.Sequential(nn.Flatten(start_dim=1))
                total_concat_size += 7
        self.extractors = nn.ModuleDict(extractors)
        self.flattener = nn.Flatten(start_dim=0)
        self.flatten_module = nn.Flatten(start_dim=1)
        self._features_dim = total_concat_size
        print("feature_dim:", self._features_dim)

    def forward(self, observations):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            # print("before:", "observations[key].shape",observations[key].shape)
            # print("after:", "tensorshape after featureextr",key,extractor(observations[key]).shape)
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

class YumiCustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            print("key", key, "shape", subspace.shape)
            if key == "image1":
                model_conv = torchvision.models.resnet18(pretrained=True)
                for param in model_conv.parameters():
                    param.requires_grad = False
                num_ftrs = model_conv.fc.in_features
                model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 32), nn.ReLU(), nn.Linear(32, 3))
                model_conv.load_state_dict(th.load("./models/feature_extractor_img_frozen_resnet18_clf_pin_left_right_cam1.pt"))
                extractors[key] = model_conv
                total_concat_size += 3
            elif key == "image2":
                model_conv = torchvision.models.resnet18(pretrained=True)
                for param in model_conv.parameters():
                    param.requires_grad = False
                num_ftrs = model_conv.fc.in_features
                model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 32), nn.ReLU(), nn.Linear(32, 3))
                model_conv.load_state_dict(
                    th.load("./models/feature_extractor_img_frozen_resnet18_clf_pin_left_right_cam2.pt"))
                extractors[key] = model_conv
                total_concat_size += 3
            elif key == "image1_pin_location":
                extractors[key] = nn.Sequential(nn.Flatten(start_dim=1))
                total_concat_size += 1
            elif key == "image2_pin_location":
                extractors[key] = nn.Sequential(nn.Flatten(start_dim=1))
                total_concat_size += 1
            elif key == "pressure":
                # Just flatten the value?
                extractors[key] = nn.Sequential(nn.Flatten(start_dim=1))
                # Run through a simple MLP
                #extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 16),
                #                                nn.Flatten(start_dim=1))  # linear create second dim, flatten it
                total_concat_size += 1
            elif key == "pose":  # or key == "pixel_error_1": # TODO rename to goal pixel error
                # Run through a simple MLP
                #extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 16),
                #                                nn.Flatten(start_dim=1))  # linear create second dim, flatten it
                extractors[key] = nn.Sequential(nn.Flatten(start_dim=1))
                total_concat_size += 7

        self.extractors = nn.ModuleDict(extractors)
        # Mapping module recommends an action based on 6 values from CNNs
        weights = th.tensor(
            [[0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0.5],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0.5],
             [0, 0, 1, 0, 0]]).T
        self.mapping_module = nn.Linear(weights.shape[1], weights.shape[0])
        # total_concat_size = total_concat_size - 1
        with th.no_grad():
            self.mapping_module.weight = nn.Parameter(th.tensor(weights))
            self.mapping_module.bias = nn.Parameter(th.zeros((weights.shape[0])))
        self.flattener = nn.Flatten(start_dim=0)
        self.flatten_module = nn.Flatten(start_dim=1)
        # Update the features dim manually
        self._features_dim = total_concat_size
        print("feature_dim:", self._features_dim)

    def forward(self, observations):
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print("bevor:", "observations[key].shape",observations[key].shape)
            # print("after:", "tensorshape after featureextr",key,extractor(observations[key]).shape)
            encoded_tensor_list.append(extractor(observations[key]))
            # it is like a non-trainable linear layer (frozen Matrix consists of zeros and ones and frozen bias 0.5)
            # torch.heaviside or torch.where or torch.nn.Hardshrink
            # better let the value flow to the correct output neuron (forward, backward, left, right, down) by changing the linear. Or use Gated torch.nn.GLU?

        # Transform first 6 values (L1, C1, R1, ...) into a 5-vector (S, W, D, A, E)
        # images_interpretation = [self.mapping_module(th.cat(encoded_tensor_list[:2],dim=1))]
        # print('\n (S, W, D, A, E) \n', images_interpretation, '\n')
        # return th.cat(images_interpretation + encoded_tensor_list[2:], dim=1)  # self.flattener(th.Tensor(encoded_tensor_list))
        # print('TYPE TENSOR LIST:', encoded_tensor_list)

        return th.cat(encoded_tensor_list, dim=1)
        # TODO adapt to Batches with th.cat


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )
# model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
# model.learn(1000)
