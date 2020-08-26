package org.jetbrains.research.kotlin.inference.benchmark.operators

import org.jetbrains.research.kotlin.inference.BenchmarkUtils.KIState
import org.jetbrains.research.kotlin.inference.BenchmarkUtils.OrtState
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole

@State(Scope.Benchmark)
@Warmup(iterations = 1)
@Measurement(iterations = 5)
open class OperatorsBenchmark {
    // Mask: operator.test_name.data_set
    @Param(
        "split.split_variable_parts_2d.0",
        "split.split_equal_parts_default_axis.0",
        "split.split_variable_parts_1d.0",
        "split.split_variable_parts_default_axis.0",
//        "split.split_zero_size_splits.0",
        "split.split_equal_parts_1d.0",
        "split.split_equal_parts_2d.0",
        "reshape.reshape_reordered_all_dims.0",
        "reshape.reshape_one_dim.0",
        "reshape.reshape_extended_dims.0",
        "reshape.reshape_negative_dim.0",
        "reshape.reshape_reduced_dims.0",
        "reshape.reshape_zero_and_negative_dim.0",
        "reshape.reshape_zero_dim.0",
        "reshape.reshape_reordered_last_dims.0",
        "reshape.reshape_negative_extended_dims.0",
        "concat.concat_2d_axis_negative_2.0",
        "concat.concat_3d_axis_2.0",
        "concat.concat_3d_axis_negative_2.0",
        "concat.concat_1d_axis_negative_1.0",
        "concat.concat_3d_axis_negative_3.0",
        "concat.concat_1d_axis_0.0",
        "concat.concat_2d_axis_negative_1.0",
        "concat.concat_2d_axis_0.0",
        "concat.concat_2d_axis_1.0",
        "concat.concat_3d_axis_negative_1.0",
        "concat.concat_3d_axis_1.0",
        "concat.concat_3d_axis_0.0",
        "attention.unidirectional_masked_multi_head.0",
        "gather.gather_negative_indices.0",
        "gather.gather_1.0",
        "gather.gather_0.0",
        "shape.shape.0",
        "shape.shape_example.0",
        "identity.identity.0",
        "slice.slice_negative_axes.0",
        "slice.slice_default_steps.0",
        "slice.slice.0",
        "slice.slice_end_out_of_bounds.0",
        "slice.slice_neg.0",
        "slice.slice_default_axes.0",
        "slice.slice_neg_steps.0",
        "slice.slice_start_out_of_bounds.0",
        "unsqueeze.unsqueeze_three_axes.0",
        "unsqueeze.unsqueeze_axis_3.0",
        "unsqueeze.unsqueeze_axis_2.0",
        "unsqueeze.unsqueeze_negative_axes.0",
        "unsqueeze.unsqueeze_two_axes.0",
        "unsqueeze.unsqueeze_axis_0.0",
        "unsqueeze.unsqueeze_unsorted_axes.0",
        "unsqueeze.unsqueeze_axis_1.0",
        "transpose.transpose_all_permutations_1.0",
        "transpose.transpose_all_permutations_0.0",
        "transpose.transpose_all_permutations_2.0",
        "transpose.transpose_all_permutations_5.0",
        "transpose.transpose_all_permutations_4.0",
        "transpose.transpose_all_permutations_3.0",
        "transpose.transpose_default.0",
        "tanh.tanh.0",
        "tanh.tanh_scalar.0",
        "tanh.tanh_example.0",
        "add.add_bcast.0",
        "add.add.0",
        "add.add_scalar.0",
        "layer_normalization.negate_axis.0",
        "layer_normalization.layer_normalization_0.0",
//        "constant.constant.0",
//        "constant.scalar_constant.0",
        "softmax.softmax_axis_0.0",
        "softmax.softmax_axis_1.0",
        "softmax.softmax_negative_axis.0",
        "softmax.softmax_large_number.0",
        "softmax.softmax_axis_2.0",
        "softmax.softmax_default_axis.0",
        "relu.relu.0",
        "fastgelu.fastgelu_with_bias.0",
        "fastgelu.fastgelu_without_bias.0",
        "sigmoid.sigmoid.0",
        "sigmoid.sigmoid_example.0",
        "matmul.matmul_3d.0",
        "matmul.matmul_4d.0",
        "matmul.matmul_2d.0",
        "squeeze.squeeze_negative_axes.0",
        "squeeze.squeeze.0",
        "loop.loop.0",
        "lstm.bilstm_defaults.0",
        "lstm.lstm_defaults.0",
        "lstm.bilstm_with_bias.0",
        "lstm.lstm_with_initial_bias.0"
    )
    lateinit var path: String

    lateinit var ortState: OrtState
    lateinit var kiState: KIState

    @Setup(Level.Trial)
    fun setup() {
        ortState = OrtState.create(path)
        kiState = KIState.create(path)
    }

    @Benchmark
    fun benchmarkKI(blackhole: Blackhole) {
        val outputs = kiState.model.predict(kiState.inputs)
        blackhole.consume(outputs)
    }

    @Benchmark
    fun benchmarkORT(blackhole: Blackhole) {
        val outputs = ortState.session.run(ortState.inputs)
        blackhole.consume(outputs)
    }
}
