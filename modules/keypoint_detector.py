from torch import nn
import torch
import torch.nn.functional as F
import mediapipe as mp
import cv2
import numpy as np
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class _KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(
        self,
        block_expansion,
        num_kp,
        num_channels,
        max_features,
        num_blocks,
        temperature,
        estimate_jacobian=False,
        scale_factor=1,
        single_jacobian_map=False,
        pad=0,
    ):
        super(_KPDetector, self).__init__()

        self.predictor = Hourglass(
            block_expansion,
            in_features=num_channels,
            max_features=max_features,
            num_blocks=num_blocks,
        )

        self.kp = nn.Conv2d(
            in_channels=self.predictor.out_filters,
            out_channels=num_kp,
            kernel_size=(7, 7),
            padding=pad,
        )

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(
                in_channels=self.predictor.out_filters,
                out_channels=4 * self.num_jacobian_maps,
                kernel_size=(7, 7),
                padding=pad,
            )
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(
                torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float)
            )
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = (
            make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        )
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {"value": value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(
                final_shape[0],
                self.num_jacobian_maps,
                4,
                final_shape[2],
                final_shape[3],
            )
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out["jacobian"] = jacobian

        return out


class MediapipeKPDetector(_KPDetector):
    def __init__(self, 
                 block_expansion,
                num_kp,
                num_channels,
                max_features,
                num_blocks,
                temperature,
                estimate_jacobian=False,
                scale_factor=1,
                single_jacobian_map=False,
                pad=0,
                kp_variance=0.01):
        super(MediapipeKPDetector, self).__init__(block_expansion=block_expansion,
                num_kp=num_kp,
                num_channels=num_channels,
                max_features=max_features,
                num_blocks=num_blocks,
                temperature=temperature,
                estimate_jacobian=estimate_jacobian,
                scale_factor=scale_factor,
                single_jacobian_map=single_jacobian_map,
                pad=pad,
                kp_variance=kp_variance)
        self.mp_face_mesh = mp.solutions.face_mesh

    def forward(self, x, image_path=None):
        """
        Overrides the KPDetector forward function.
        Accepts an image path, processes it with Mediapipe, and returns the Jacobians and keypoints for detected keypoints.
        """
        if image_path is not None:
            # Step 1: Process image with Mediapipe and get keypoints
            image, mediapipe_keypoints = self.process_image_with_mediapipe(image_path)

            # Step 2: Normalize the Mediapipe keypoints to the model's coordinate system
            normalized_keypoints = self.normalize_mediapipe_keypoints(
                mediapipe_keypoints, image.shape
            )

            # Step 3: Create heatmap using Mediapipe keypoints
            heatmap = self.create_heatmap_from_keypoints(
                normalized_keypoints, spatial_size=(x.shape[2], x.shape[3])
            )

            # Step 4: Predict Jacobians using the KPDetector's internal functions
            feature_map = self.predictor(x)  # Get the feature map
            jacobian_map = self.jacobian(feature_map)  # Jacobian from feature map
            jacobian_map = jacobian_map.view(
                1, self.num_jacobian_maps, 4, x.shape[2], x.shape[3]
            )

            # Step 5: Multiply heatmap with Jacobian map and sum over spatial dimensions
            heatmap = heatmap.unsqueeze(2)  # Shape: (1, num_keypoints, 1, H, W)
            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(1, self.num_jacobian_maps, 4, -1).sum(dim=-1)
            jacobian = jacobian.view(
                1, self.num_jacobian_maps, 2, 2
            )  # Final shape: (1, num_keypoints, 2, 2)

            # Include the 'value' key, i.e., the normalized keypoints
            return {
                "value": normalized_keypoints.unsqueeze(
                    0
                ),  # Shape: (1, num_keypoints, 2)
                "jacobian": jacobian,
            }
        else:
            # If no image is provided, fallback to the default behavior
            return super(MediapipeKPDetector, self).forward(x)

    def process_image_with_mediapipe(self, image_path):
        """
        Process the image using Mediapipe and return the detected keypoints and the processed image.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from path: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with Mediapipe
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:

            results = face_mesh.process(image_rgb)

        # Check if any landmarks were detected
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape

            # Extract the filtered landmarks (face, eyes, eyebrows, nose, mouth, cheeks)
            filtered_landmarks = self.get_filtered_landmarks()
            keypoints = [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in filtered_landmarks
            ]
        else:
            raise ValueError("No face landmarks detected!")

        return image, keypoints

    def get_filtered_landmarks(self):
        """
        Get the specific facial landmarks we want to extract from Mediapipe.
        """
        oval_lmk_ind = sorted(
            list(set(np.array(list(self.mp_face_mesh.FACEMESH_FACE_OVAL)).flatten()))
        )
        left_eye_lmk_ind = sorted(
            list(set(np.array(list(self.mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()))
        )
        right_eye_lmk_ind = sorted(
            list(set(np.array(list(self.mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()))
        )
        left_eye_brow_lmk_ind = sorted(
            list(set(np.array(list(self.mp_face_mesh.FACEMESH_LEFT_EYEBROW)).flatten()))
        )
        right_eye_brow_lmk_ind = sorted(
            list(
                set(np.array(list(self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW)).flatten())
            )
        )
        nose_top = [5, 195, 197, 6, 168]
        nose_bottom = [
            2,
            326,
            327,
            294,
            278,
            344,
            440,
            275,
            4,
            1,
            19,
            94,
            45,
            220,
            115,
            48,
            64,
            98,
            97,
        ]
        mouth_landmarks = [
            0,
            13,
            14,
            17,
            37,
            39,
            40,
            61,
            78,
            80,
            81,
            82,
            84,
            87,
            88,
            91,
            95,
            146,
            178,
            181,
            185,
            191,
            267,
            269,
            270,
            291,
            308,
            310,
            311,
            312,
            314,
            317,
            318,
            321,
            324,
            375,
            402,
            405,
            409,
            415,
        ]
        cheeks = [
            128,
            266,
            142,
            399,
            151,
            412,
            416,
            419,
            36,
            425,
            427,
            174,
            437,
            188,
            192,
            196,
            199,
            205,
            207,
            217,
            345,
            346,
            347,
            348,
            349,
            350,
            351,
            355,
            357,
            371,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            126,
        ]
        return (
            oval_lmk_ind
            + left_eye_lmk_ind
            + right_eye_lmk_ind
            + left_eye_brow_lmk_ind
            + right_eye_brow_lmk_ind
            + nose_top
            + nose_bottom
            + mouth_landmarks
            + cheeks
        )

    def normalize_mediapipe_keypoints(self, keypoints, image_shape):
        """
        Normalize Mediapipe keypoints to the [-1, 1] range relative to image dimensions.
        """
        h, w = image_shape[:2]
        normalized_keypoints = []
        for kp in keypoints:
            x, y = kp
            x = 2 * (x / w) - 1  # Normalize x to [-1, 1]
            y = 2 * (y / h) - 1  # Normalize y to [-1, 1]
            normalized_keypoints.append([x, y])
        return torch.tensor(normalized_keypoints, dtype=torch.float32)

    def create_heatmap_from_keypoints(self, keypoints, spatial_size, sigma=0.1):
        """
        Generate a heatmap for each keypoint.
        """
        grid = self.make_coordinate_grid(
            spatial_size, type=keypoints.type()
        )  # [-1, 1] x [-1, 1] grid
        heatmaps = []

        for kp in keypoints:
            dist = torch.sum(
                (grid - kp.view(1, 1, 2)) ** 2, dim=-1
            )  # L2 distance between grid points and keypoint
            heatmap = torch.exp(-dist / (2 * sigma**2))  # Gaussian distribution
            heatmaps.append(heatmap)

        heatmaps = torch.stack(heatmaps, dim=0)
        return heatmaps.unsqueeze(0)  # Shape: (1, num_keypoints, H, W)
