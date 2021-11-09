package io.kinference.kmath

import ai.onnxruntime.*
import io.kinference.data.*
import io.kinference.ndarray.toIntArray
import io.kinference.ndarray.toLongArray
import io.kinference.ort.data.map.ORTMap
import io.kinference.ort.data.seq.ORTSequence
import io.kinference.ort.data.tensor.ORTTensor
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import java.nio.*

sealed class ORTKMathData<T>(override val name: String?) : BaseONNXData<T> {
    class KMathTensor(name: String?, override val data: NDStructure<*>) : ORTKMathData<NDStructure<*>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
        override fun rename(name: String): ORTKMathData<NDStructure<*>> = KMathTensor(name, data)
    }

    class KMathMap(name: String?, override val data: Map<Any, ORTKMathData<*>>) : ORTKMathData<Map<Any, ORTKMathData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_MAP
        override fun rename(name: String): ORTKMathData<Map<Any, ORTKMathData<*>>> = KMathMap(name, data)
    }

    class KMathSequence(name: String?, override val data: List<ORTKMathData<*>>) : ORTKMathData<List<ORTKMathData<*>>>(name) {
        override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
        override fun rename(name: String): ORTKMathData<List<ORTKMathData<*>>> = KMathSequence(name, data)
    }
}

object ORTKMathTensorAdapter : ONNXDataAdapter<ORTKMathData.KMathTensor, ORTTensor> {
    override fun fromONNXData(data: ORTTensor): ORTKMathData.KMathTensor {
        val info = data.data.info
        val linearSize = info.shape.toIntArray().fold(1, Int::times)
        val buffer = when (val type = info.type) {
            OnnxJavaType.FLOAT -> {
                val data = data.data.floatBuffer
                Buffer.auto(linearSize) { data.get() }
            }
            OnnxJavaType.DOUBLE -> {
                val data = data.data.doubleBuffer
                Buffer.auto(linearSize) { data.get() }
            }
            OnnxJavaType.INT8 -> {
                val data = data.data.byteBuffer
                Buffer.auto(linearSize) { data.get() }
            }
            OnnxJavaType.INT16 -> {
                val data = data.data.shortBuffer
                Buffer.auto(linearSize) { data.get() }
            }
            OnnxJavaType.INT32 -> {
                val data = data.data.intBuffer
                Buffer.auto(linearSize) { data.get() }
            }
            OnnxJavaType.INT64 -> {
                val data = data.data.longBuffer
                Buffer.auto(linearSize) { data.get() }
            }
            OnnxJavaType.UINT8 -> {
                val data = data.data.byteBuffer
                Buffer.auto(linearSize) { data.get().toUByte() }
            }
            else -> error("Unsupported data type: $type")
        }
        return ORTKMathData.KMathTensor(data.name ?: "", NDBuffer(DefaultStrides(info.shape.toIntArray()), buffer))
    }

    override fun toONNXData(data: ORTKMathData.KMathTensor): ORTTensor {
        val env = OrtEnvironment.getEnvironment()
        val elements = data.data.elements().map { it.second!! }.iterator()
        val linSize = data.data.shape.fold(1, Int::times)
        val shapeLong = data.data.shape.toLongArray()
        val tensor = when (val element = data.data.elements().first().second!!) {
            is Byte -> OnnxTensor.createTensor(env, ByteBuffer.wrap(ByteArray(linSize) { elements.next() as Byte }), shapeLong)
            is UByte -> OnnxTensor.createTensor(env, ByteBuffer.wrap(ByteArray(linSize) { (elements.next() as UByte).toByte() }), shapeLong, OnnxJavaType.UINT8)
            is Short -> OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(linSize) { elements.next() as Short }), shapeLong)
            is Int -> OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(linSize) { elements.next() as Int }), shapeLong)
            is Long -> OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(linSize) { elements.next() as Long }), shapeLong)
            is Double -> OnnxTensor.createTensor(env, DoubleBuffer.wrap(DoubleArray(linSize) { elements.next() as Double }), shapeLong)
            is Float -> OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(linSize) { elements.next() as Float }), shapeLong)
            else -> error("Cannot convert from StructureND of ${element::class} to ONNXTensor")
        }
        return ORTTensor(data.name, tensor)
    }
}

object ORTKMathMapAdapter : ONNXDataAdapter<ORTKMathData.KMathMap, ORTMap> {
    override fun fromONNXData(data: ORTMap): ORTKMathData.KMathMap {
        return ORTKMathData.KMathMap(data.name, data.data.value.mapValues { ORTKMathData.KMathTensor(null, it.toScalarBuffer()) })
    }

    override fun toONNXData(data: ORTKMathData.KMathMap): ORTMap {
        error("ONNXRuntime backend does not support map conversion")
    }
}

object ORTKMathSequenceAdapter : ONNXDataAdapter<ORTKMathData.KMathSequence, ORTSequence> {
    override fun fromONNXData(data: ORTSequence): ORTKMathData.KMathSequence {
        val elements = data.data.value
        val seq = if (data.data.info.sequenceOfMaps) {
            (elements as List<Map<*, *>>)
                .map { entry -> entry.mapValues { ORTKMathData.KMathTensor(null, it.value!!.toScalarBuffer()) } }
                .map { ORTKMathData.KMathMap(null, it as Map<Any, ORTKMathData<*>>) }
        } else {
            elements.map { ORTKMathData.KMathTensor(null, it.toScalarBuffer()) }
        }
        return ORTKMathData.KMathSequence(data.name, seq)
    }

    override fun toONNXData(data: ORTKMathData.KMathSequence): ORTSequence {
        error("ONNXRuntime backend does not support sequence conversion")
    }
}

private fun Any.toScalarBuffer() = NDBuffer(DefaultStrides(IntArray(0)), Buffer.auto(1) { this })
