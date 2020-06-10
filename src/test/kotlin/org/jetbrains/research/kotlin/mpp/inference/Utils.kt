package org.jetbrains.research.kotlin.mpp.inference

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.model.Model
import org.jetbrains.research.kotlin.mpp.inference.space.SpaceStrides
import org.jetbrains.research.kotlin.mpp.inference.space.TensorRing
import org.jetbrains.research.kotlin.mpp.inference.space.resolveSpace
import org.jetbrains.research.kotlin.mpp.inference.space.toIntArray
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.NDBuffer
import scientifik.kmath.structures.asBuffer
import scientifik.kmath.structures.asIterable
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow

object Utils {
    private val delta = (10.0).pow(-6)

    fun getTensor(path: File): Tensor {
        val tensorProto = TensorProto.ADAPTER.decode(path.readBytes())
        return when (DataType.fromValue(tensorProto.data_type!!) ?: 0) {
            DataType.FLOAT -> getTensorFloat(tensorProto)
            DataType.INT64 -> getTensorLong(tensorProto)
            else -> throw UnsupportedOperationException()
        }
    }

    private fun getTensorFloat(tensorProto: TensorProto): Tensor {
        val rawFloatData = tensorProto.raw_data!!.toByteArray()
        val chunkedRawFloatData = rawFloatData.asIterable().chunked(4)
        val floatData = chunkedRawFloatData.map { ByteBuffer.wrap(it.reversed().toByteArray()).float }.asBuffer()
        val structure = BufferNDStructure(SpaceStrides(tensorProto.dims.toIntArray()), floatData) as NDBuffer<Any>
        return Tensor(tensorProto.name, structure, DataType.FLOAT, resolveSpace<Float>(tensorProto.dims) as TensorRing<Any>)
    }

    private fun getTensorLong(tensorProto: TensorProto): Tensor {
        val rawLongData = tensorProto.raw_data!!.toByteArray()
        val chunkedRawLongData = rawLongData.asIterable().chunked(8)
        val longData = chunkedRawLongData.map { ByteBuffer.wrap(it.reversed().toByteArray()).long }.asBuffer()
        val structure = BufferNDStructure(SpaceStrides(tensorProto.dims.toIntArray()), longData) as NDBuffer<Any>
        return Tensor(tensorProto.name, structure, DataType.INT64, resolveSpace<Long>(tensorProto.dims) as TensorRing<Any>)
    }

    fun assertTensors(expected: Tensor, actual: Tensor) {
        assertEquals(expected.type, actual.type, "Types of tensors ${expected.name} do not match")
        assertArrayEquals(expected.data.shape, actual.data.shape)
        @Suppress("UNCHECKED_CAST")
        when (expected.type) {
            DataType.FLOAT -> {
                expected.data.buffer.asIterable().forEachIndexed() { index, value ->
                    value as Float
                    assertEquals(value, (actual.data.buffer[index] as Number).toFloat(), delta.toFloat(), "Tensor ${expected.name} does not match")
                }
            }

            DataType.DOUBLE -> {
                expected.data.buffer.asIterable().forEachIndexed() { index, value ->
                    value as Double
                    assertEquals(value, (actual.data.buffer[index] as Number).toDouble(), delta, "Tensor ${expected.name} does not match")
                }
            }

            else -> assertEquals(expected, actual, "Tensor ${expected.name} does not match")
        }
    }

    @Suppress("UNCHECKED_CAST")
    fun operatorTestHelper(folderName: String): List<Pair<List<Tensor>, List<Tensor>>> {
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

    fun singleTestHelper(testDir: String) {
        val dataSets = operatorTestHelper(testDir)
        for (dataSet in dataSets) {
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.name }

            for (expectedOutputTensor in expectedOutputTensors){
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.name] ?: error("Required tensor not found")
                assertTensors(expectedOutputTensor, actualOutputTensor)
            }
        }
    }
}
