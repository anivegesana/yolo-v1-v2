�($	j�J���@�@���!@��� !ʧ?!�Z�kBzA@	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�Z�kBzA@1[rP��:@IKZ��g @"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails5�u��?1��F����?I�ɧ�6�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H5���:�?x�ܙ	��?1A�
��?Ist���?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails(��&2s�?1���'ׄ?I�y�3Mغ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{�%9`W�?&�"�dTI?1��.��y?I2��z�p�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails|��l;m�?1Mjh��?I3�PlM�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�(&o���?K;5��?1�7�0��?IL<��?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U.T����?x` �C��?1d?��H��?IPn�����?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsh�.�K�?1�d��7i�?I�tp�x�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	EF$a��?��7�ܘ~?1Ժj���?ID�b*��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
�7��w�?1�&�|�w?I)���^�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6��x"��?����?k�?1b����?I`9B���?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_�Qڻ?1��I���b?IøDkE�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&��f��}�?�9[@h=�?1��DJ�y�?I��,g�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��� !ʧ?1yxρ�y?I��p��?*333335q@B`�����@2K
Iterator::Model::Map��|#�_;@!`��$@�X@)UL��p*@1��(�G@:Preprocessing2_
(Iterator::Model::Map::PaddedBatchV2::Map��>9
P#@!qؼ�dA@)	5C�(F#@1ԇ���[A@:Preprocessing2Z
#Iterator::Model::Map::PaddedBatchV2@mT��,@!"�!��I@)�+��@1a����0@:Preprocessing2�
�Iterator::Model::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap[0]::TFRecordW[����?!kq:2��?)W[����?1kq:2��?:Advanced file read2�
�Iterator::Model::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap/N|��8�?!E���O�?)�-�\o��?1�؋��?:Preprocessing2�
cIterator::Model::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinalityf���,Ц?!�XY켋�?)���5�?1����?:Preprocessing2�
yIterator::Model::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4#-��#��?!����\�?)#-��#��?1����\�?:Preprocessing2x
AIterator::Model::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2�����?!�Xꑲ?)�����?1�Xꑲ?:Preprocessing2i
2Iterator::Model::Map::PaddedBatchV2::Map::Prefetch��a�Ó?!v9�Ṉ?)��a�Ó?1v9�Ṉ?:Preprocessing2�
PIterator::Model::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2'���?!J�D��?)'���?1J�D��?:Preprocessing2F
Iterator::Model;�ʃ�`;@!��F[�X@)E���s?1�Z���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�29.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�-��Sx�?�P̳�W�?!K;5��?	!       "$	Fl*��?�G �י@��I���b?![rP��:@*	!       2	!       :$	���ZY~�?�K��ԁ @��p��?!KZ��g @B	!       J	!       R	!       Z	!       JGPUb �"R
6yolov3/darknet53/DarkRes_1_0/dark_conv/conv2d_1/Conv2DConv2Dj����?!j����?"T
8yolov3/darknet53/DarkRes_1_0/dark_conv_2/conv2d_3/Conv2DConv2D��A]ǭ?!Z��F��?"�
dgradient_tape/yolov3/regular/dark_route_process_2/dark_conv_70/conv2d_73/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�Z��u4�?!fq�9��?"�
agradient_tape/yolov3/regular/dark_route_process/dark_conv_56/conv2d_57/Conv2D/Conv2DBackpropInputConv2DBackpropInputY�x)t��?!�ϼ���?"V
:yolov3/darknet53/DarkRes_8_0/dark_conv_42/conv2d_43/Conv2DConv2D��q���?!���9���?"V
:yolov3/darknet53/DarkRes_8_0/dark_conv_44/conv2d_45/Conv2DConv2D�Eڶ�?!��|���?"�
bgradient_tape/yolov3/regular/dark_route_process/dark_conv_56/conv2d_57/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ��>�?!=a����?"T
8yolov3/darknet53/DarkRes_2_0/dark_conv_5/conv2d_6/Conv2DConv2D�k��ˉ�?!�J�p�^�?"T
8yolov3/darknet53/DarkRes_2_0/dark_conv_3/conv2d_4/Conv2DConv2D�Գ�E��?!1��,h�?"�
dgradient_tape/yolov3/regular/dark_route_process_1/dark_conv_63/conv2d_65/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterH{��H�?!MJ���6�?It3}HH�D@Q�̂��9M@Y'��Ѥ^.@a���e+4U@q�RNf�N�?y��b��`?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�29.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no:
Refer to the TF2 Profiler FAQ2"GPU(: B 