package io.kinference.multik

import ai.onnxruntime.*
import io.kinference.data.ONNXDataAdapter
import io.kinference.ndarray.toIntArray
import io.kinference.ndarray.toLongArray
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.data.tensor.ORTTensor
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.nio.*

object ORTMultikTensorAdapter : ONNXDataAdapter<MultiArray<Number, Dimension>, ORTTensor> {
    override fun fromONNXData(data: ORTTensor): MultiArray<Number, Dimension> {
        val tensor = data.data
        val dtype = tensor.info.type
        val view = when (dtype) {
            OnnxJavaType.FLOAT -> MemoryViewFloatArray(tensor.floatBuffer.array())
            OnnxJavaType.DOUBLE -> MemoryViewDoubleArray(tensor.doubleBuffer.array())
            OnnxJavaType.INT8 -> MemoryViewByteArray(tensor.byteBuffer.array())
            OnnxJavaType.INT16 -> MemoryViewShortArray(tensor.shortBuffer.array())
            OnnxJavaType.INT32 -> MemoryViewIntArray(tensor.intBuffer.array())
            OnnxJavaType.INT64 -> MemoryViewLongArray(tensor.longBuffer.array())
            else -> error("$dtype type is not supported by Multik")
        } as MemoryView<Number>
        return NDArray(view, shape = tensor.info.shape.toIntArray(), dtype = dtype.resolveMultikDataType(), dim = dimensionOf(tensor.info.shape.size))
    }

    override fun toONNXData(name: String, data: MultiArray<Number, Dimension>): ORTTensor {
        val arrayData = data.data
        val env = OrtEnvironment.getEnvironment()
        val tensor = when (val array = arrayData.data) {
            is DoubleArray -> OnnxTensor.createTensor(env, DoubleBuffer.wrap(array), data.shape.toLongArray())
            is FloatArray -> OnnxTensor.createTensor(env, FloatBuffer.wrap(array), data.shape.toLongArray())
            is LongArray -> OnnxTensor.createTensor(env, LongBuffer.wrap(array), data.shape.toLongArray())
            is IntArray -> OnnxTensor.createTensor(env, IntBuffer.wrap(array), data.shape.toLongArray())
            is ShortArray -> OnnxTensor.createTensor(env, ShortBuffer.wrap(array), data.shape.toLongArray())
            is ByteArray -> OnnxTensor.createTensor(env, ByteBuffer.wrap(array), data.shape.toLongArray())
            else -> error("Unsupported data type")
        }
        return ORTTensor(name, tensor)
    }
}

object ORTMultikMapAdapter : ONNXDataAdapter<Map<Any, *>, ORTMap> {
    override fun fromONNXData(data: ORTMap): Map<Any, *> {
        val valueType = data.data.info.valueType
        return data.data.value.mapValues { it.toScalarNDArray(valueType) }
    }

    override fun toONNXData(name: String, data: Map<Any, *>): ORTMap {
        error("ONNXRuntime backend does not support map conversion")
    }
}

object ORTMultikSequenceAdapter : ONNXDataAdapter<List<*>, ORTSequence> {
    override fun fromONNXData(data: ORTSequence): List<*> {
        val elements = data.data.value
        return if (data.data.info.sequenceOfMaps) {
            val mapType = data.data.info.mapInfo.valueType
            (elements as List<Map<*, *>>).map { entry -> entry.mapValues { it.value!!.toScalarNDArray(mapType) } }
        } else {
            val valueType = data.data.info.sequenceType
            elements.map { it.toScalarNDArray(valueType) }
        }
    }

    override fun toONNXData(name: String, data: List<*>): ORTSequence {
        error("ONNXRuntime backend does not support sequence conversion")
    }
}

private fun Any.toScalarNDArray(type: OnnxJavaType): NDArray<Number, Dimension> {
    val dtype = type.resolveMultikDataType()
    val view = when (type) {
        OnnxJavaType.FLOAT -> MemoryViewFloatArray(floatArrayOf(this as Float))
        OnnxJavaType.DOUBLE -> MemoryViewDoubleArray(doubleArrayOf(this as Double))
        OnnxJavaType.INT8 -> MemoryViewByteArray(byteArrayOf(this as Byte))
        OnnxJavaType.INT16 -> MemoryViewShortArray(shortArrayOf(this as Short))
        OnnxJavaType.INT32 -> MemoryViewIntArray(intArrayOf(this as Int))
        OnnxJavaType.INT64 -> MemoryViewLongArray(longArrayOf(this as Long))
        else -> error("$type type is not supported by Multik")
    }
    return NDArray(view, shape = intArrayOf(1), dim = dimensionOf(1), dtype = dtype) as NDArray<Number, Dimension>
}

private fun OnnxJavaType.resolveMultikDataType() = when (this) {
    OnnxJavaType.FLOAT -> DataType.FloatDataType
    OnnxJavaType.DOUBLE -> DataType.DoubleDataType
    OnnxJavaType.INT8 -> DataType.ByteDataType
    OnnxJavaType.INT16 -> DataType.ShortDataType
    OnnxJavaType.INT32 -> DataType.IntDataType
    OnnxJavaType.INT64 -> DataType.LongDataType
    else -> error("$this type is not supported by Multik")
}
