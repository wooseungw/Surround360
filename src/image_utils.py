from typing import Dict, List, Optional, Union

def validate_preprocess_arguments(
    do_rescale: Optional[bool] = None,
    do_crop: Optional[bool] = None,
    fov: Optional[float] = None,
    overlap_ratio: Optional[float] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, list[float]]] = None,
    image_std: Optional[Union[float, list[float]]] = None,
    do_pad: Optional[bool] = None,
    size_divisibility: Optional[int] = None,
    do_center_crop: Optional[bool] = None,
    crop_size: Optional[dict[str, int]] = None,
    do_resize: Optional[bool] = None,
    size: Optional[dict[str, int]] = None,
    resample: Optional["PILImageResampling"] = None,
):
    """
    Checks validity of typically used arguments in an `ImageProcessor` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    Many incompatibilities are model-specific. `do_pad` sometimes needs `size_divisor`,
    sometimes `size_divisibility`, and sometimes `size`. New models and processors added should follow
    existing arguments when possible.

    """
    if do_rescale and rescale_factor is None:
        raise ValueError("`rescale_factor` must be specified if `do_rescale` is `True`.")

    if do_pad and size_divisibility is None:
        # Here, size_divisor might be passed as the value of size
        raise ValueError(
            "Depending on the model, `size_divisibility`, `size_divisor`, `pad_size` or `size` must be specified if `do_pad` is `True`."
        )

    if do_normalize and (image_mean is None or image_std is None):
        raise ValueError("`image_mean` and `image_std` must both be specified if `do_normalize` is `True`.")

    if do_center_crop and crop_size is None:
        raise ValueError("`crop_size` must be specified if `do_center_crop` is `True`.")

    if do_resize and (size is None or resample is None):
        raise ValueError("`size` and `resample` must be specified if `do_resize` is `True`.")