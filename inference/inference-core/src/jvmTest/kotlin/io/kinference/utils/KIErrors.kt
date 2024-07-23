package io.kinference.utils

import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

object KIErrors {
    fun calculateErrors(expected: KITensor, actual: KITensor): Errors.ErrorsData {
        require(expected.info.type == actual.info.type)

        return when (expected.data.type) {
            DataType.FLOAT -> {
                val expectedBlocks = (expected.data as FloatNDArray).array.blocks
                val actualBlocks = (actual.data as FloatNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.DOUBLE -> {
                val expectedBlocks = (expected.data as DoubleNDArray).array.blocks
                val actualBlocks = (actual.data as DoubleNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.BYTE -> {
                val expectedBlocks = (expected.data as ByteNDArray).array.blocks
                val actualBlocks = (actual.data as ByteNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.SHORT -> {
                val expectedBlocks = (expected.data as ShortNDArray).array.blocks
                val actualBlocks = (actual.data as ShortNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.INT -> {
                val expectedBlocks = (expected.data as IntNDArray).array.blocks
                val actualBlocks = (actual.data as IntNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.LONG -> {
                val expectedBlocks = (expected.data as LongNDArray).array.blocks
                val actualBlocks = (actual.data as LongNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.UBYTE -> {
                val expectedBlocks = (expected.data as UByteNDArray).array.blocks
                val actualBlocks = (actual.data as UByteNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.USHORT -> {
                val expectedBlocks = (expected.data as UShortNDArray).array.blocks
                val actualBlocks = (actual.data as UShortNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.UINT -> {
                val expectedBlocks = (expected.data as UIntNDArray).array.blocks
                val actualBlocks = (actual.data as UIntNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.ULONG -> {
                val expectedBlocks = (expected.data as ULongNDArray).array.blocks
                val actualBlocks = (actual.data as ULongNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            DataType.BOOLEAN -> {
                val expectedBlocks = (expected.data as BooleanNDArray).array.blocks
                val actualBlocks = (actual.data as BooleanNDArray).array.blocks
                Errors.computeErrors(expectedBlocks, actualBlocks)
            }

            else -> error("Unsupported tensor data type ${expected.info.type}")
        }
    }

    fun calculateErrors(expected: KIONNXSequence, actual: KIONNXSequence): List<Errors.ErrorsData> {
        val result = mutableListOf<Errors.ErrorsData>()
        for (idx in expected.data.indices) {
            result.addAll(calculateErrors(expected.data[idx], actual.data[idx]))
        }
        return result
    }

    fun calculateErrors(expected: KIONNXMap, actual: KIONNXMap): List<Errors.ErrorsData> {
        val result = mutableListOf<Errors.ErrorsData>()
        for (entry in expected.data.entries) {
            result.addAll(calculateErrors(entry.value, actual.data[entry.key]!!))
        }

        return result
    }

    fun calculateErrors(expected: KIONNXData<*>, actual: KIONNXData<*>): List<Errors.ErrorsData> {
        return when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> listOf(calculateErrors(expected as KITensor, actual as KITensor))
            ONNXDataType.ONNX_MAP -> calculateErrors(expected as KIONNXMap, actual as KIONNXMap)
            ONNXDataType.ONNX_SEQUENCE -> calculateErrors(expected as KIONNXSequence, actual as KIONNXSequence)
        }
    }
}
