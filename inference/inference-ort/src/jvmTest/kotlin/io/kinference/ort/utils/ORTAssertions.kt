package io.kinference.ort.utils

import ai.onnxruntime.*
import io.kinference.data.ONNXDataType
import io.kinference.ort.ORTData
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.ArrayAssertions
import kotlin.math.abs

object ORTAssertions {
    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: ORTData<*>, actual: ORTData<*>, delta: Double) {
        require(expected.type == ONNXDataType.ONNX_TENSOR && actual.type == ONNXDataType.ONNX_TENSOR)
        assertTensorEquals(expected as ORTTensor, actual as ORTTensor, delta)
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertTensorEquals(expected: ORTTensor, actual: ORTTensor, delta: Double) {
        require(expected.data.info.type == actual.data.info.type)

        when (expected.data.info.type) {
            OnnxJavaType.FLOAT -> ArrayAssertions.assertArrayEquals(expected.data.floatBuffer.array(), actual.data.floatBuffer.array(), { l, r -> abs(l - r).toDouble() }, delta)
            OnnxJavaType.DOUBLE -> ArrayAssertions.assertArrayEquals(expected.data.doubleBuffer.array(), actual.data.doubleBuffer.array(), { l, r -> abs(l - r) }, delta)
            OnnxJavaType.INT32 -> ArrayAssertions.assertArrayEquals(expected.data.intBuffer.array(), actual.data.intBuffer.array(), { l, r -> abs(l - r).toDouble() }, delta)
            OnnxJavaType.INT64 -> ArrayAssertions.assertArrayEquals(expected.data.longBuffer.array(), actual.data.longBuffer.array(), { l, r -> abs(l - r).toDouble() }, delta)
            OnnxJavaType.INT16 -> ArrayAssertions.assertArrayEquals(expected.data.shortBuffer.array(), actual.data.shortBuffer.array(), { l, r -> abs(l - r).toDouble() }, delta)
            OnnxJavaType.INT8 -> ArrayAssertions.assertArrayEquals(expected.data.byteBuffer.array(), actual.data.byteBuffer.array(), { l, r -> abs(l - r).toDouble() }, delta)
            else -> error("Unsupported data type: ${expected.data.info.type}")
        }
    }
}

