�!$	�j��.�#@�d6��5@t	���?!0c
�8�H@	R��t�z(@w��-^;@!��+�%�N@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-0c
�8�H@�eM,��?1����=�1@I�\��7�?Y��O>@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�^�D��?1��M�q�?I�����?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�'�$��?1�<,Ԛ�?I��oB!�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails7U�q7�?1����yj?I˜.����?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailst	���?1�g�,{r?I'g(�x��?*��"���q@ףpM�@2W
 Iterator::Model::FiniteTake::Mapb�A
�n@@!�bT�X@)ԛQ��0@1�w~��PI@:Preprocessing2k
4Iterator::Model::FiniteTake::Map::PaddedBatchV2::Map��^

'@!Y;�"NA@)�Σ���&@1��Vd�EA@:Preprocessing2f
/Iterator::Model::FiniteTake::Map::PaddedBatchV2�p��H0@!�8�
H@)�����@1�dXy��*@:Preprocessing2�
�Iterator::Model::FiniteTake::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[4]::FlatMap[0]::TFRecord�R@�� �?!>�~_��?)�R@�� �?1>�~_��?:Advanced file read2�
�Iterator::Model::FiniteTake::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4��q��Q�?! NT���?)��q��Q�?1 NT���?:Preprocessing2�
oIterator::Model::FiniteTake::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality=�Ƃ �?!ҥ�ȧ��?)��8�~ߟ?1��(���?:Preprocessing2u
>Iterator::Model::FiniteTake::Map::PaddedBatchV2::Map::Prefetch��_vO�?!oS����?)��_vO�?1oS����?:Preprocessing2�
�Iterator::Model::FiniteTake::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[4]::FlatMap�o'�_�?!��*���?)��qn�?1��]��x�?:Preprocessing2�
MIterator::Model::FiniteTake::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2�1%��?!�@.k�t�?)�1%��?1�@.k�t�?:Preprocessing2�
\Iterator::Model::FiniteTake::Map::PaddedBatchV2::Map::Prefetch::ParallelMapV2::ParallelMapV2
�]�V�?!$�g���?)
�]�V�?1$�g���?:Preprocessing2R
Iterator::Model::FiniteTake ��Wo@@!6gw}k�X@)��	m9w?1hd���q�?:Preprocessing2F
Iterator::Model��7�o@@!����]�X@)���<,t?1��wE�M�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 60.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9Z�$�cN@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Pq#�D�?{U����?!�eM,��?	!       "$	���� @\��@ @����yj?!����=�1@*	!       2	!       :$	�&�WKK�?�gVEd0�?'g(�x��?!�\��7�?B	!       J	kL�r@te�:��*@!��O>@R	!       Z	kL�r@te�:��*@!��O>@JGPUYZ�$�cN@b �	"V
:yolov3/darknet53/DarkRes_9_0/dark_conv_46/conv2d_47/Conv2DConv2D�]�1��?!�]�1��?".
IteratorGetNext/_55_Send@Jɹ���?!�S�u��?"�
agradient_tape/yolov3/regular/dark_route_process/dark_conv_56/conv2d_57/Conv2D/Conv2DBackpropInputConv2DBackpropInput�'s/��?!�]{R`_�?"�
agradient_tape/yolov3/regular/dark_route_process/dark_conv_52/conv2d_53/Conv2D/Conv2DBackpropInputConv2DBackpropInputsF|��ҍ?!�7� 
�?"�
agradient_tape/yolov3/regular/dark_route_process/dark_conv_54/conv2d_55/Conv2D/Conv2DBackpropInputConv2DBackpropInputل<4�ȍ?!W�4�$��?"�
bgradient_tape/yolov3/regular/dark_route_process/dark_conv_56/conv2d_57/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterG��?!��aŹ?"�
bgradient_tape/yolov3/regular/dark_route_process/dark_conv_54/conv2d_55/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter$W�c�?!�Ӳ��?"�
bgradient_tape/yolov3/regular/dark_route_process/dark_conv_52/conv2d_53/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltery7����?!H͡��?"V
:yolov3/darknet53/DarkRes_9_1/dark_conv_48/conv2d_49/Conv2DConv2D�2��.�?!u����?"V
:yolov3/darknet53/DarkRes_8_0/dark_conv_44/conv2d_45/Conv2DConv2D��ĳ&�?!� M?[�?IY���uLT@Q��)4)�2@Y9*�0@a�q5�<�T@qK�i&��?yP�@��?z?"�
host�Your program is HIGHLY input-bound because 60.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"GPU(: B 