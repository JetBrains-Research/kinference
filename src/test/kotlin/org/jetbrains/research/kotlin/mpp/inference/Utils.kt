package org.jetbrains.research.kotlin.mpp.inference

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.math.extensions.asBuffer
import org.jetbrains.research.kotlin.mpp.inference.data.ONNXData
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*
import org.jetbrains.research.kotlin.mpp.inference.model.Model
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.asBuffer
import scientifik.kmath.structures.asIterable
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow

object Utils {
    private val delta = (10.0).pow(-5)

    fun getTensor(path: File): BaseTensor {
        val tensorProto = TensorProto.ADAPTER.decode(path.readBytes())
        return when (DataType.fromValue(tensorProto.data_type!!) ?: 0) {
            DataType.FLOAT -> getTensorFloat(tensorProto)
            DataType.INT64 -> getTensorLong(tensorProto)
            else -> throw UnsupportedOperationException()
        }
    }

    private fun getTensorFloat(tensorProto: TensorProto): BaseTensor {
        val floatData = if (tensorProto.raw_data != null) {
            val rawFloatData = tensorProto.raw_data!!.toByteArray()
            val chunkedRawFloatData = rawFloatData.asIterable().chunked(4)
            chunkedRawFloatData.map { ByteBuffer.wrap(it.reversed().toByteArray()).float }.toFloatArray().asBuffer()
        } else tensorProto.float_data.toFloatArray().asBuffer()

        return if (tensorProto.dims.isEmpty()) {
            ScalarTensor.create(tensorProto)
        } else {
            val structure = BufferNDStructure(TensorStrides(tensorProto.dims.toIntArray()), floatData) as BufferNDStructure<Any>
            Tensor(tensorProto.name, structure, DataType.FLOAT)
        }
    }

    private fun getTensorLong(tensorProto: TensorProto): BaseTensor {
        val longData = if (tensorProto.raw_data != null) {
            val rawLongData = tensorProto.raw_data!!.toByteArray()
            val chunkedRawLongData = rawLongData.asIterable().chunked(8)
            chunkedRawLongData.map { ByteBuffer.wrap(it.reversed().toByteArray()).long }.toLongArray().asBuffer()
        } else tensorProto.int64_data.toLongArray().asBuffer()

        return if (tensorProto.dims.isEmpty()) {
            ScalarTensor.create(tensorProto)
        } else {
            val structure = BufferNDStructure(TensorStrides(tensorProto.dims.toIntArray()), longData) as BufferNDStructure<Any>
            Tensor(tensorProto.name, structure, DataType.INT64)
        }
    }

    fun assertTensors(expected: Tensor, actual: Tensor) {
        assertEquals(expected.type, actual.type, "Types of tensors ${expected.info.name} do not match")
        assertArrayEquals(expected.data.shape, actual.data.shape)

        @Suppress("UNCHECKED_CAST")
        when (expected.info.type) {
            DataType.FLOAT -> {
                expected.data.buffer.asIterable().forEachIndexed { index, value ->
                    value as Float
                    assertEquals(value, (actual.data.buffer[index] as Number).toFloat(), delta.toFloat(), "Tensor ${expected.info.name} does not match")
                }
            }

            DataType.DOUBLE -> {
                expected.data.buffer.asIterable().forEachIndexed { index, value ->
                    value as Double
                    assertEquals(value, (actual.data.buffer[index] as Number).toDouble(), delta, "Tensor ${expected.info.name} does not match")
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
