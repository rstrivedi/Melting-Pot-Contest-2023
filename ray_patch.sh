FILE_TO_PATCH=`python -c "from ray.rllib.models.torch import complex_input_net; print(complex_input_net.__file__)"`
echo Attempting to patch $FILE_TO_PATCH
patch -f $FILE_TO_PATCH < ray_patches/complex_input_net.patch

FILE_TO_PATCH=`python -c "from ray.rllib.policy import sample_batch; print(sample_batch.__file__)"`
echo Attempting to patch $FILE_TO_PATCH
patch -f $FILE_TO_PATCH < ray_patches/sample_batch.patch
