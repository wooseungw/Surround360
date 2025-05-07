# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import numpy as np
from PIL import Image
from py360convert import e2p
import itertools

from torch import nn
from transformers import Blip2VisionModel
from transformers import BlipImageProcessor
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
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
"""
Processor class for Custom BLIP-2.
"""
logger = logging.get_logger(__name__)


class Blip2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {
            "do_crop": False,
            "fov": 90.0,
            "overlap_ratio": 0.5,
            },
    }


class Blip2Processor(ProcessorMixin):
    r"""
    Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

    [`BlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        num_query_tokens (`int`, *optional*):
            Number of tokens used by the Qformer as queries, should be same as in model's config.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["num_query_tokens"]
    image_processor_class = ("BlipImageProcessor", "BlipImageProcessorFast")
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, num_query_tokens=None, **kwargs):
        tokenizer.return_token_type_ids = False
        self.current_processor = image_processor
        if not hasattr(tokenizer, "image_token"):
            self.image_token = AddedToken("<image>", normalized=False, special=True)
            tokenizer.add_tokens([self.image_token], special_tokens=True)
        else:
            self.image_token = tokenizer.image_token
        self.num_query_tokens = num_query_tokens

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Blip2ProcessorKwargs],
    ) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        Args:
            images (`ImageInput`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")
        output_kwargs = self._merge_kwargs(
            Blip2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # BC for explicit return_tensors
        if "return_tensors" in output_kwargs["common_kwargs"]:
            return_tensors = output_kwargs["common_kwargs"].pop("return_tensors", None)
        else:
            return_tensors = None
        encoding = BatchFeature(tensor_type=return_tensors)
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            text_encoding = {}

            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            _text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
            output_kwargs["text_kwargs"]["return_tensors"] = return_tensors

            # if we know how many query tokens, expand text inside processor. We need this hacky manipulation
            # because BLIP expects image tokens to be at the beginning even before BOS token
            if self.num_query_tokens is not None:
                image_tokens = self.image_token.content * self.num_query_tokens
                image_token_encoding = self.tokenizer(
                    [image_tokens] * len(text), add_special_tokens=False, return_tensors=None
                )
                for k in _text_encoding:
                    text_encoding[k] = [
                        img_encoding + txt_encoding
                        for img_encoding, txt_encoding in zip(image_token_encoding[k], _text_encoding[k])
                    ]
            else:
                text_encoding = _text_encoding
                logger.warning_once(
                    "Expanding inputs for image tokens in BLIP-2 should be done in processing. "
                    "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your BLIP-2 model. "
                    "Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
                )

            # cast to desired return tensors type
            encoding.update(BatchEncoding(text_encoding, tensor_type=return_tensors))
        # add pixel_values encoding. If we also have text_encoding, update image encoding and return it.
        # else, return the text encoding.

        if images is not None:
            image_encoding = self.image_processor(images, **output_kwargs["images_kwargs"])
            encoding.update(image_encoding)
        return encoding

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

class SurroundBlipImageProcessor(BlipImageProcessor):
    """
    BlipImageProcessor를 상속받아, 이미지 전처리 기능을 추가한 클래스입니다.
    do_crop=True일 때와 False일 때 do_resize 동작을 분리 처리합니다.
    """
    def __init__(
        self,
        do_resize: bool = True,
        do_crop: bool = False,
        fov: float = 90.0,
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
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.do_crop = do_crop
        self.fov = fov
        self.overlap_ratio = overlap_ratio
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or OPENAI_CLIP_MEAN
        self.image_std = image_std or OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        size = get_size_dict(size)
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def crop(
        self,
        image: np.ndarray,
        fov: Optional[float] = None,
        overlap_ratio: Optional[float] = None,
        **kwargs
    ) -> List[np.ndarray]:
        fov = fov if fov is not None else self.fov
        overlap_ratio = overlap_ratio if overlap_ratio is not None else self.overlap_ratio
        H, W, C = image.shape
        out_h = self.size["height"]
        out_w = self.size["width"]
        step_angle = fov * (1.0 - overlap_ratio)
        yaw_centers = np.arange(fov / 2.0, 360.0, step_angle)

        patches: List[np.ndarray] = []
        for yaw in yaw_centers:
            patch = e2p(
                image,
                FOV_deg=fov,
                u_center_deg=float(yaw),
                v_center_deg=0.0,
                out_hw=(out_h, out_w)
            )
            patches.append(patch)
        return patches

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        do_crop: Optional[bool] = None,
        fov: Optional[float] = None,
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
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_crop = do_crop if do_crop is not None else self.do_crop
        fov = fov if fov is not None else self.fov
        overlap_ratio = overlap_ratio if overlap_ratio is not None else self.overlap_ratio
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

        if do_convert_rgb:
            images = [convert_to_rgb(img) for img in images]
        images = [to_numpy_array(img) for img in images]

        # 1) do_resize 분기
        if do_resize:
            resized = []
            for img in images:
                if do_crop:
                    target = {"height": self.size["height"] * 2, "width": self.size["width"] * 4}
                else:
                    target = {"height": self.size["height"], "width": self.size["width"]}
                tgt = get_size_dict(target, default_to_square=False)
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((tgt["width"], tgt["height"]), resample=resample)
                resized.append(np.array(pil_img))
            images = resized

        # 2) do_rescale
        if do_rescale:
            images = [self.rescale(image=img, scale=rescale_factor, input_data_format=input_data_format) for img in images]

        # 3) do_crop
        if do_crop:
            cropped = [patch for img in images for patch in self.crop(image=img, fov=fov, overlap_ratio=overlap_ratio)]
            images = cropped

        # 4) do_normalize
        if do_normalize:
            images = [self.normalize(image=img, mean=image_mean, std=image_std, input_data_format=input_data_format) for img in images]

        # 5) to tensor format
        images = [to_channel_dimension_format(img, data_format, input_channel_dim=input_data_format) for img in images]
        return BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
