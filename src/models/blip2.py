import torch
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import numpy as np
import PIL 
import itertools

from torch import nn
from transformers import Blip2VisionModel, Blip2QFormerModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipImageProcessor
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
)
from src.image_utils import validate_preprocess_arguments
from transformers.utils import TensorType, filter_out_non_signature_kwargs, is_vision_available, logging

from transformers import Blip2Config
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

logger = logging.get_logger(__name__)

class Blip2VisionModel(Blip2VisionModel):
    "다중 처리 가능하도록 수정"
    def __init__(self, config):
        super().__init__(config)
        
        
# class Blip2QFormerModel(Blip2QFormerModel):
#     def __init__(self, config):
#         super().__init__(config)
#         pass

# class Blip2ForConditionalGeneration(Blip2ForConditionalGeneration):
    
#     def __init__(self, config: Blip2Config):
#         super().__init__(config)

#         self.vision_model = Blip2VisionModel(config.vision_config)

#         self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
#         self.qformer = Blip2QFormerModel(config.qformer_config)

#         self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
#         if config.use_decoder_only_language_model:
#             language_model = AutoModelForCausalLM.from_config(config.text_config)
#         else:
#             language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

#         # Update _tied_weights_keys using the base model used.
#         if language_model._tied_weights_keys is not None:
#             self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

#         self.language_model = language_model

#         # Initialize weights and apply final processing
#         self.post_init()

class SurroundBlipImageProcessor(BlipImageProcessor):
    """
    BlipImageProcessor를 상속받아, 이미지 전처리 기능을 추가한 클래스입니다.
    overlap_ratio: 
    오리지널 이미지 Reisze -> Crop 
    """
    def __init__(
        self,
        do_resize: bool = True,
        do_crop: bool = True,
        overlap_ratio: float = 0.5,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 384, "width": 384} # default size for BLIP-2
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.do_crop = do_crop
        self.overlap_ratio = overlap_ratio
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize with PILImageResampling.BILINEAR->PILImageResampling.BICUBIC
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        
    def crop(self, image: np.ndarray, **kwargs) -> List[np.ndarray]:
        """
        image: numpy.ndarray of shape (H, W, C)
        returns: list of patches, each as numpy.ndarray (patch_size, patch_size, C)
        """
        image_pixel_values = []
        patch_size = self.size["height"]
        stride = int(patch_size * (1 - self.overlap_ratio))
        H, W, C = image.shape

        # 1) 슬라이딩 크롭 (numpy slicing)
        for x in range(0, W - patch_size + 1, stride):
            # height 전체, width 구간, 채널 전체
            patch = image[:, x : x + patch_size, :]
            image_pixel_values.append(patch)

        # 2) 래핑 패치 (끝과 처음 50%)
        # 마지막 슬라이딩 패치에서 슬라이딩 스트라이드만큼 오른쪽 끝을, 
        # 첫 패치에서 왼쪽 스트라이드만큼을 이어 붙임
        last_part  = image[:, W - stride : W, :]       # 오른쪽 끝 stride
        first_part = image[:, 0 : stride, :]           # 왼쪽 처음 stride

        # 빈 패치 배열 준비
        wrap = np.empty((patch_size, patch_size, C), dtype=image.dtype)
        # 왼쪽 절반에 last_part, 오른쪽 절반에 first_part 배치
        wrap[:, : last_part.shape[1], :] = last_part
        wrap[:, last_part.shape[1] :, :] = first_part

        image_pixel_values.append(wrap)

        return image_pixel_values
    
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        do_crop: Optional[bool] = None,
        overlap_ratio: Optional[float] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The shortest edge of the image is resized to
                `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
                is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
                edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            do_crop = do_crop,
            overlap_ratio=overlap_ratio,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        # expect images to be in (height, width, num_channels) format
        if do_crop:
            # 먼저 각 이미지의 패치 리스트를 모은 뒤
            cropped_lists = [
                self.crop(image=image, overlap_ratio=overlap_ratio, input_data_format=input_data_format)
                for image in images
            ]
            # chain.from_iterable 로 2중 리스트를 평탄화
            images = list(itertools.chain.from_iterable(cropped_lists))
            
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]
        
        encoded_outputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

        return encoded_outputs
    
    