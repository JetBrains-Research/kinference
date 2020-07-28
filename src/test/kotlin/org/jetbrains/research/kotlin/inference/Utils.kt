package org.jetbrains.research.kotlin.inference

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.primitives.toIntArray
import org.jetbrains.research.kotlin.inference.model.Model
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow

object Utils {
    private val delta = (10.0).pow(-5)

    fun getTensor(path: File): Tensor {
        val tensorProto = TensorProto.ADAPTER.decode(path.readBytes())
        return when (DataType.fromValue(tensorProto.data_type!!) ?: 0) {
            DataType.FLOAT -> getTensorFloat(tensorProto)
            DataType.INT64 -> getTensorLong(tensorProto)
            else -> throw UnsupportedOperationException()
        }
    }

    private fun getTensorFloat(tensorProto: TensorProto): Tensor {
        val floatData = if (tensorProto.raw_data != null) {
            val rawFloatData = tensorProto.raw_data!!.toByteArray()
            val chunkedRawFloatData = rawFloatData.asIterable().chunked(4)
            chunkedRawFloatData.map { ByteBuffer.wrap(it.reversed().toByteArray()).float }.toFloatArray()
        } else tensorProto.float_data.toFloatArray()

        val strides = Strides(tensorProto.dims.toIntArray())
        return FloatNDArray(floatData, strides).asTensor(tensorProto.name!!)
    }

    private fun getTensorLong(tensorProto: TensorProto): Tensor {
        val longData = if (tensorProto.raw_data != null) {
            val rawLongData = tensorProto.raw_data!!.toByteArray()
            val chunkedRawLongData = rawLongData.asIterable().chunked(8)
            chunkedRawLongData.map { ByteBuffer.wrap(it.reversed().toByteArray()).long }.toLongArray()
        } else tensorProto.int64_data.toLongArray()

        val strides = Strides(tensorProto.dims.toIntArray())
        return LongNDArray(longData, strides).asTensor(tensorProto.name!!)
    }

    fun assertTensors(expected: Tensor, actual: Tensor) {
        assertEquals(expected.type, actual.type, "Types of tensors ${expected.info.name} do not match")
        assertArrayEquals(expected.data.shape, actual.data.shape)

        @Suppress("UNCHECKED_CAST")
        when (expected.info.type) {
            DataType.FLOAT -> {
                ((expected.data as FloatNDArray).array as FloatArray).forEachIndexed { index, value ->
                    assertEquals(value, (actual.data.array as FloatArray)[index], delta.toFloat(), "Tensor ${expected.info.name} does not match")
                }
            }

            DataType.DOUBLE -> {
                ((expected.data as DoubleNDArray).array as DoubleArray).forEachIndexed { index, value ->
                    assertEquals(value, (actual.data.array as DoubleArray)[index], delta, "Tensor ${expected.info.name} does not match")
                }
            }

            DataType.INT64 -> {
                ((expected.data as LongNDArray).array as LongArray).forEachIndexed { index, value ->
                    assertEquals(value, (actual.data.array as LongArray)[index], "Tensor ${expected.info.name} does not match")
                }
            }

            else -> assertEquals(expected, actual, "Tensor ${expected.info.name} does not match")
        }
    }

    @Suppress("UNCHECKED_CAST")
    fun operatorTestHelper(folderName: String): List<Pair<List<ONNXData>, List<ONNXData>>> {
        val path = javaClass.getResource(folderName).path
        val model = Model.load(path + "model.onnx")

        return File(path).list()!!.filter { "test" in it }.map {
            val inputFiles = File("$path/$it").walk().filter { file -> "input" in file.name }
            val outputFiles = File("$path/$it").walk().filter { file -> "output" in file.name }

            val inputTensors = inputFiles.map { getTensor(it) }.toList()
            val expectedOutputTensors = outputFiles.map { getTensor(it) }.toList()
            val actualOutputTensors = model.predict(inputTensors)
            Pair(expectedOutputTensors, actualOutputTensors)
        }
    }

    fun tensorTestRunner(testDir: String) {
        val dataSets = operatorTestHelper(testDir)
        for (dataSet in dataSets) {
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.info.name }

            for (expectedOutputTensor in expectedOutputTensors){
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.info.name] ?: error("Required tensor not found")
                assertTensors(expectedOutputTensor as Tensor, actualOutputTensor as Tensor)
            }
        }
    }
}
