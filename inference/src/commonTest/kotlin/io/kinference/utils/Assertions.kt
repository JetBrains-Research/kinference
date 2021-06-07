package io.kinference.utils

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.data.map.ONNXMap
import io.kinference.data.seq.ONNXSequence
import io.kinference.data.tensors.Tensor
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.logger
import io.kinference.onnx.TensorProto
import io.kinference.types.ValueTypeInfo
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.floor
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object Assertions {
    val logger = logger("Errors")

    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: Tensor, actual: Tensor, delta: Double) {
        assertEquals(expected.data.type, actual.data.type, "Types of tensors ${expected.info.name} do not match")
        assertArrayEquals(expected.data.shape.toTypedArray(), actual.data.shape.toTypedArray(), "Shapes are incorrect")

        val typeInfo = expected.info.typeInfo as ValueTypeInfo.TensorTypeInfo
        when (typeInfo.type) {
            TensorProto.DataType.FLOAT -> {
                val expectedArray = (expected.data as FloatNDArray).array
                val actualArray = (actual.data as FloatNDArray).array

                val errorsArray = FloatArray(expectedArray.size) { i ->
                    abs(expectedArray[i] - actualArray[i])
                }

                val averageError = if (errorsArray.size != 0) errorsArray.sum() / errorsArray.size else 0f
                val standardDeviation = errorsArray.sumOf { (it - averageError).pow(2).toDouble() } / (errorsArray.size - 1)

                val sortedErrorsArray = errorsArray.sorted()

                val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
                val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
                val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
                val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

                logger.info { "average error '${actual.info.name}' = $averageError" }
                logger.info { "standard deviation '${actual.info.name}' = $standardDeviation" }
                logger.info { "Percentile 50 '${actual.info.name}' = $percentile50" }
                logger.info { "Percentile 95 '${actual.info.name}' = $percentile95" }
                logger.info { "Percentile 99 '${actual.info.name}' = $percentile99" }
                logger.info { "Percentile 99.9 '${actual.info.name}' = $percentile999\n" }

                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r).toDouble() }, delta, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.DOUBLE -> {
                val expectedArray = (expected.data as DoubleNDArray).array
                val actualArray = (actual.data as DoubleNDArray).array

                val errorsArray = DoubleArray(expectedArray.size) { i ->
                    abs(expectedArray[i] - actualArray[i])
                }

                val averageError = if (errorsArray.size != 0) errorsArray.sum() / errorsArray.size else 0.0
                val standardDeviation = errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1)

                val sortedErrorsArray = errorsArray.sorted()

                val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0.0 }
                val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0.0 }
                val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0.0 }
                val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0.0 }

                logger.info { "average error '${actual.info.name}' = $averageError" }
                logger.info { "standard deviation '${actual.info.name}' = $standardDeviation" }
                logger.info { "Percentile 50 '${actual.info.name}' = $percentile50" }
                logger.info { "Percentile 95 '${actual.info.name}' = $percentile95" }
                logger.info { "Percentile 99 '${actual.info.name}' = $percentile99" }
                logger.info { "Percentile 99.9 '${actual.info.name}' = $percentile999\n" }

                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r) }, delta, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.INT64 -> {
                val expectedArray = (expected.data as LongNDArray).array
                val actualArray = (actual.data as LongNDArray).array

                val errorsArray = LongArray(expectedArray.size) { i ->
                    abs(expectedArray[i] - actualArray[i])
                }

                val averageError = if (errorsArray.size != 0) errorsArray.sum() / errorsArray.size else 0
                val standardDeviation = errorsArray.sumOf { (it - averageError).toDouble().pow(2) } / (errorsArray.size - 1)

                val sortedErrorsArray = errorsArray.sorted()

                val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0 }

                logger.info { "average error '${actual.info.name}' = $averageError" }
                logger.info { "standard deviation '${actual.info.name}' = $standardDeviation" }
                logger.info { "Percentile 50 '${actual.info.name}' = $percentile50" }
                logger.info { "Percentile 95 '${actual.info.name}' = $percentile95" }
                logger.info { "Percentile 99 '${actual.info.name}' = $percentile99" }
                logger.info { "Percentile 99.9 '${actual.info.name}' = $percentile999\n" }

                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r).toDouble() }, delta, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.INT32 -> {
                val expectedArray = (expected.data as IntNDArray).array
                val actualArray = (actual.data as IntNDArray).array

                val errorsArray = IntArray(expectedArray.size) { i ->
                    abs(expectedArray[i] - actualArray[i])
                }

                val averageError = if (errorsArray.size != 0) errorsArray.sum() / errorsArray.size else 0
                val standardDeviation = errorsArray.sumOf { (it - averageError).toDouble().pow(2) } / (errorsArray.size - 1)

                val sortedErrorsArray = errorsArray.sorted()

                val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0 }

                logger.info { "average error '${actual.info.name}' = $averageError" }
                logger.info { "standard deviation '${actual.info.name}' = $standardDeviation" }
                logger.info { "Percentile 50 '${actual.info.name}' = $percentile50" }
                logger.info { "Percentile 95 '${actual.info.name}' = $percentile95" }
                logger.info { "Percentile 99 '${actual.info.name}' = $percentile99" }
                logger.info { "Percentile 99.9 '${actual.info.name}' = $percentile999\n" }

                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l - r).toDouble() }, delta, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.BOOL -> {
                val expectedArray = (expected.data as BooleanNDArray).array.toArray().toTypedArray()
                val actualArray = (actual.data as BooleanNDArray).array.toArray().toTypedArray()
                assertArrayEquals(expectedArray, actualArray, "Tensor ${expected.info.name} does not match")
            }
            TensorProto.DataType.UINT8 -> {
                val expectedArray = (expected.data as UByteNDArray).array
                val actualArray = (actual.data as UByteNDArray).array

                val errorsArray = IntArray(expectedArray.size) { i ->
                    abs(expectedArray[i].toInt() - actualArray[i].toInt())
                }

                val averageError = if (errorsArray.size != 0) errorsArray.sum() / errorsArray.size else 0
                val standardDeviation = errorsArray.sumOf { (it - averageError).toDouble().pow(2) } / (errorsArray.size - 1)

                val sortedErrorsArray = errorsArray.sorted()

                val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0 }
                val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0 }

                logger.info { "average error '${actual.info.name}' = $averageError" }
                logger.info { "standard deviation '${actual.info.name}' = $standardDeviation" }
                logger.info { "Percentile 50 '${actual.info.name}' = $percentile50" }
                logger.info { "Percentile 95 '${actual.info.name}' = $percentile95" }
                logger.info { "Percentile 99 '${actual.info.name}' = $percentile99" }
                logger.info { "Percentile 99.9 '${actual.info.name}' = $percentile999\n" }

                assertArrayEquals(expectedArray, actualArray, { l, r -> abs(l.toInt() - r.toInt()).toDouble() }, delta, "Tensor ${expected.info.name} does not match")
            }
            else -> assertEquals(expected, actual, "Tensor ${expected.info.name} does not match")
        }
    }

    fun assertArrayEquals(left: FloatTiledArray, right: FloatTiledArray, diff: (Float, Float) -> Double, delta: Double, message: String) {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: DoubleTiledArray, right: DoubleTiledArray, diff: (Double, Double) -> Double, delta: Double, message: String) {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: LongTiledArray, right: LongTiledArray, diff: (Long, Long) -> Double, delta: Double, message: String) {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: IntTiledArray, right: IntTiledArray, diff: (Int, Int) -> Double, delta: Double, message: String) {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: UByteTiledArray, right: UByteTiledArray, diff: (UByte, UByte) -> Double, delta: Double, message: String) {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun <T : Comparable<T>> assertArrayEquals(left: Array<T>, right: Array<T>, diff: (T, T) -> Double, delta: Double, message: String) {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun <T> assertArrayEquals(left: Array<T>, right: Array<T>, message: String) {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertEquals(l, r, message)
        }
    }

    fun assertEquals(expected: ONNXMap, actual: ONNXMap, delta: Double) {
        assertEquals(expected.keyType, actual.keyType, "Map key types should match")
        assertEquals(expected.data.keys, actual.data.keys, "Map key sets are not equal")

        for (entry in expected.data.entries) {
            assertEquals(entry.value, actual.data[entry.key]!!, delta)
        }
    }

    fun assertEquals(expected: ONNXSequence, actual: ONNXSequence, delta: Double) {
        assertEquals(expected.length, actual.length, "Sequence lengths do not match")

        for (i in expected.data.indices) {
            assertEquals(expected.data[i], actual.data[i], delta)
        }
    }

    fun assertEquals(expected: ONNXData, actual: ONNXData, delta: Double) {
        when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> assertEquals(expected as Tensor, actual as Tensor, delta)
            ONNXDataType.ONNX_MAP -> assertEquals(expected as ONNXMap, actual as ONNXMap, delta)
            ONNXDataType.ONNX_SEQUENCE -> assertEquals(expected as ONNXSequence, actual as ONNXSequence, delta)
        }
    }
}
