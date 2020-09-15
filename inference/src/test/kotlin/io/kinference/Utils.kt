package io.kinference

import io.kinference.ndarray.Strides
import io.kinference.ndarray.toIntArray
import io.kinference.data.ONNXData
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.model.Model
import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.onnx.TensorProto
import io.kinference.onnx.TensorProto.DataType
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.pow

@ExperimentalUnsignedTypes
object Utils {
    private val delta = (10.0).pow(-3)

    fun getTensor(file: File): Tensor = getTensor(file.readBytes())

    fun getTensor(byteArray: ByteArray): Tensor = getTensor(TensorProto.ADAPTER.decode(byteArray))

    fun getTensor(tensorProto: TensorProto): Tensor {
        return when (val type = DataType.fromValue(tensorProto.data_type!!) ?: 0) {
            DataType.FLOAT -> getTensorFloat(tensorProto)
            DataType.DOUBLE -> getTensorDouble(tensorProto)
            DataType.INT64 -> getTensorLong(tensorProto)
            DataType.INT32 -> getTensorInt(tensorProto)
            DataType.INT8 -> getTensorByte(tensorProto)
            DataType.UINT8 -> getTensorUByte(tensorProto)
            DataType.BOOL -> getTensorBoolean(tensorProto)
            else -> error("Unsupported proto data type: $type")
        }
    }

    private fun getTensorBoolean(tensorProto: TensorProto): Tensor {
        val booleanData = if (tensorProto.raw_data != null) {
            val rawBooleanData = tensorProto.raw_data!!.toByteArray()
            BooleanArray(rawBooleanData.size) { rawBooleanData[it].toInt() != 0 }
        } else BooleanArray(tensorProto.int32_data.size) { tensorProto.int32_data[it] != 0 }

        val strides = Strides(tensorProto.dims.toIntArray())
        return BooleanNDArray(booleanData, strides).asTensor(tensorProto.name!!)
    }

    private fun getTensorByte(tensorProto: TensorProto): Tensor {
        val byteData = if (tensorProto.raw_data != null) {
            tensorProto.raw_data!!.toByteArray()
        } else ByteArray(tensorProto.int32_data.size) { tensorProto.int32_data[it].toByte() }

        val strides = Strides(tensorProto.dims.toIntArray())
        return ByteNDArray(byteData, strides).asTensor(tensorProto.name!!)
    }

    private fun getTensorUByte(tensorProto: TensorProto): Tensor {
        val uByteData = if (tensorProto.raw_data != null) {
            val bytes = tensorProto.raw_data!!.toByteArray()
            UByteArray(bytes.size) { bytes[it].toUByte() }
        } else UByteArray(tensorProto.int32_data.size) { tensorProto.int32_data[it].toUByte() }

        val strides = Strides(tensorProto.dims.toIntArray())
        return UByteNDArray(uByteData, strides).asTensor(tensorProto.name!!)
    }

    private fun getTensorDouble(tensorProto: TensorProto): Tensor {
        val doubleData = if (tensorProto.raw_data != null) {
            val rawFloatData = tensorProto.raw_data!!.toByteArray()
            val chunkedRawFloatData = rawFloatData.asIterable().chunked(8)
            chunkedRawFloatData.map { ByteBuffer.wrap(it.reversed().toByteArray()).double }.toDoubleArray()
        } else tensorProto.double_data.toDoubleArray()

        val strides = Strides(tensorProto.dims.toIntArray())
        return DoubleNDArray(doubleData, strides).asTensor(tensorProto.name!!)
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

    private fun getTensorInt(tensorProto: TensorProto): Tensor {
        val intData = if (tensorProto.raw_data != null) {
            val rawIntData = tensorProto.raw_data!!.toByteArray()
            val chunkedRawIntData = rawIntData.asIterable().chunked(4)
            chunkedRawIntData.map { ByteBuffer.wrap(it.reversed().toByteArray()).int }.toIntArray()
        } else tensorProto.int32_data.toIntArray()

        val strides = Strides(tensorProto.dims.toIntArray())
        return IntNDArray(intData, strides).asTensor(tensorProto.name!!)
    }

    fun assertTensors(expected: Tensor, actual: Tensor, delta: Double) {
        assertEquals(expected.type, actual.type, "Types of tensors ${expected.info.name} do not match")
        assertArrayEquals(expected.data.shape, actual.data.shape)

        @Suppress("UNCHECKED_CAST")
        when (expected.info.type) {
            DataType.FLOAT -> {
                ((expected.data as FloatNDArray).array).forEachIndexed { index, value ->
                    assertEquals(value, actual.data[index] as Float, delta.toFloat(), "Tensor ${expected.info.name} does not match")
                }
            }

            DataType.DOUBLE -> {
                ((expected.data as DoubleNDArray).array).forEachIndexed { index, value ->
                    assertEquals(value, actual.data[index] as Double, delta, "Tensor ${expected.info.name} does not match")
                }
            }

            DataType.INT64 -> {
                ((expected.data as LongNDArray).array).forEachIndexed { index, value ->
                    assertEquals(value, actual.data[index] as Long, "Tensor ${expected.info.name} does not match")
                }
            }

            DataType.INT32 -> {
                ((expected.data as IntNDArray).array).forEachIndexed { index, value ->
                    assertEquals(value, actual.data[index] as Int, "Tensor ${expected.info.name} does not match")
                }
            }

            DataType.BOOL -> {
                ((expected.data as BooleanNDArray).array).forEachIndexed { index, value ->
                    assertEquals(value, actual.data[index] as Boolean, "Tensor ${expected.info.name} does not match")
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

    fun tensorTestRunner(testDir: String, delta: Double = this.delta) {
        val dataSets = operatorTestHelper(testDir)
        for (dataSet in dataSets) {
            val (expectedOutputTensors, actualOutputTensors) = dataSet

            val mappedActualOutputTensors = actualOutputTensors.associateBy { it.info.name }

            for (expectedOutputTensor in expectedOutputTensors) {
                val actualOutputTensor = mappedActualOutputTensors[expectedOutputTensor.info.name] ?: error("Required tensor not found")
                assertTensors(expectedOutputTensor as Tensor, actualOutputTensor as Tensor, delta)
            }
        }
    }
}
