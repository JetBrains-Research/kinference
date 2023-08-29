package io.kinference.multik

import ai.onnxruntime.*
import io.kinference.data.*
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.toIntArray
import io.kinference.utils.toLongArray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.nio.*

sealed class ORTMultikData<T>(override val name: String?) : BaseONNXData<T> {
    abstract override fun rename(name: String): ORTMultikData<T>
    abstract override fun clone(newName: String?): ORTMultikData<T>

    class MultikTensor(name: String?, override val data: MultiArray<Number, Dimension>, val dataType: OnnxJavaType) : ORTMultikData<MultiArray<Number, Dimension>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
        override fun rename(name: String): MultikTensor = MultikTensor(name, data, dataType)
        override fun clone(newName: String?): MultikTensor {
            return MultikTensor(newName, data.deepCopy(), dataType)
        }
    }

    class MultikMap(name: String?, override val data: Map<Any, ORTMultikData<*>>) : ORTMultikData<Map<Any, ORTMultikData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_MAP
        override fun rename(name: String): MultikMap = MultikMap(name, data)
        override fun clone(newName: String?): MultikMap {
            val newMap = HashMap<Any, ORTMultikData<*>>(data.size)
            for ((key, value) in data.entries) {
                newMap[key] = value.clone()
            }
            return MultikMap(newName, newMap)
        }
    }

    class MultikSequence(name: String?, override val data: List<ORTMultikData<*>>) : ORTMultikData<List<ORTMultikData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
        override fun rename(name: String): MultikSequence =MultikSequence(name, data)
        override fun clone(newName: String?): MultikSequence {
            return MultikSequence(newName, data.map { it.clone() })
        }
    }
}

object ORTMultikTensorAdapter : ONNXDataAdapter<ORTMultikData.MultikTensor, ORTTensor> {
    override fun fromONNXData(data: ORTTensor): ORTMultikData.MultikTensor {
        val tensor = data.data
        val dtype = tensor.info.type
        val view = when (dtype) {
            OnnxJavaType.FLOAT -> MemoryViewFloatArray(tensor.floatBuffer.array())
            OnnxJavaType.DOUBLE -> MemoryViewDoubleArray(tensor.doubleBuffer.array())
            OnnxJavaType.INT8 -> MemoryViewByteArray(tensor.byteBuffer.array())
            OnnxJavaType.INT16 -> MemoryViewShortArray(tensor.shortBuffer.array())
            OnnxJavaType.INT32 -> MemoryViewIntArray(tensor.intBuffer.array())
            OnnxJavaType.INT64 -> MemoryViewLongArray(tensor.longBuffer.array())
            OnnxJavaType.BOOL -> MemoryViewByteArray(tensor.byteBuffer.array())
            else -> error("$dtype type is not supported by Multik")
        } as MemoryView<Number>
        val ndArray = NDArray(view, shape = tensor.info.shape.toIntArray()/*, dtype = dtype.resolveMultikDataType()*/, dim = dimensionOf(tensor.info.shape.size))
        return ORTMultikData.MultikTensor(data.name, ndArray, dtype)
    }

    override fun toONNXData(data: ORTMultikData.MultikTensor): ORTTensor {
        val arrayData = data.data.data
        val env = OrtEnvironment.getEnvironment()
        val shapeLong = data.data.shape.toLongArray()
        val tensor = when (val array = arrayData.data) {
            is DoubleArray -> OnnxTensor.createTensor(env, DoubleBuffer.wrap(array), shapeLong)
            is FloatArray -> OnnxTensor.createTensor(env, FloatBuffer.wrap(array), shapeLong)
            is LongArray -> OnnxTensor.createTensor(env, LongBuffer.wrap(array), shapeLong)
            is IntArray -> OnnxTensor.createTensor(env, IntBuffer.wrap(array), shapeLong)
            is ShortArray -> OnnxTensor.createTensor(env, ShortBuffer.wrap(array), shapeLong)
            is ByteArray -> {
                //TODO: rewrite it normally
                if (data.dataType == OnnxJavaType.BOOL) {
                    val booleanArray = BooleanArray(array.size) { array[it] == (1).toByte() }
                    return ORTTensor.invoke(booleanArray, shapeLong, data.name)
                } else {
                    OnnxTensor.createTensor(env, ByteBuffer.wrap(array), shapeLong)
                }
            }
            else -> error("Unsupported data type")
        }
        return ORTTensor(data.name, tensor)
    }
}

object ORTMultikMapAdapter : ONNXDataAdapter<ORTMultikData.MultikMap, ORTMap> {
    override fun fromONNXData(data: ORTMap): ORTMultikData.MultikMap {
        val valueType = data.data.info.valueType
        val mapValues = data.data.value.mapValues { ORTMultikData.MultikTensor(null, it.toScalarNDArray(valueType), valueType) }
        return ORTMultikData.MultikMap(data.name, mapValues)
    }

    override fun toONNXData(data: ORTMultikData.MultikMap): ORTMap {
        error("ONNXRuntime backend does not support map conversion")
    }
}

object ORTMultikSequenceAdapter : ONNXDataAdapter<ORTMultikData.MultikSequence, ORTSequence> {
    override fun fromONNXData(data: ORTSequence): ORTMultikData.MultikSequence {
        val elements = data.data.value
        val seq = if (data.data.info.sequenceOfMaps) {
            val mapType = data.data.info.mapInfo.valueType
            (elements as List<Map<*, *>>)
                .map { entry -> entry.mapValues { ORTMultikData.MultikTensor(null, it.value!!.toScalarNDArray(mapType), mapType) } }
                .map { ORTMultikData.MultikMap(null, it as Map<Any, ORTMultikData<*>>) }
        } else {
            val valueType = data.data.info.sequenceType
            elements.map { ORTMultikData.MultikTensor(null, it.toScalarNDArray(valueType), valueType) }
        }
        return ORTMultikData.MultikSequence(data.name, seq)
    }

    override fun toONNXData(data: ORTMultikData.MultikSequence): ORTSequence {
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
    return NDArray(view, shape = intArrayOf(1), dim = dimensionOf(1)/*, dtype = dtype*/) as NDArray<Number, Dimension>
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
