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

object ORTKMathTensorAdapter : ONNXDataAdapter<NDStructure<*>, ORTTensor> {
    override fun fromONNXData(data: ORTTensor): NDStructure<*> {
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
            else -> error("Unsupported data type: $type")
        }
        return NDBuffer(DefaultStrides(info.shape.toIntArray()), buffer)
    }

    override fun toONNXData(name: String, data: NDStructure<*>): ORTTensor {
        val env = OrtEnvironment.getEnvironment()
        val elements = data.elements().map { it.second!! }.iterator()
        val linSize = data.shape.fold(1, Int::times)
        val shapeLong = data.shape.toLongArray()
        val tensor = when (val element = data.elements().first().second!!) {
            is Byte -> OnnxTensor.createTensor(env, ByteBuffer.wrap(ByteArray(linSize) { elements.next() as Byte }), shapeLong)
            is Short -> OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(linSize) { elements.next() as Short }), shapeLong)
            is Int -> OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(linSize) { elements.next() as Int }), shapeLong)
            is Long -> OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(linSize) { elements.next() as Long }), shapeLong)
            is Double -> OnnxTensor.createTensor(env, DoubleBuffer.wrap(DoubleArray(linSize) { elements.next() as Double }), shapeLong)
            is Float -> OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(linSize) { elements.next() as Float }), shapeLong)
            else -> error("Cannot convert from StructureND of ${element::class} to ONNXTensor")
        }
        return ORTTensor(name, tensor)
    }
}

object ORTKMathMapAdapter : ONNXDataAdapter<Map<Any, *>, ORTMap> {
    override fun fromONNXData(data: ORTMap): Map<Any, *> {
        return data.data.value.mapValues { it.toScalarBuffer() }
    }

    override fun toONNXData(name: String, data: Map<Any, *>): ORTMap {
        error("ONNXRuntime backend does not support map conversion")
    }
}

object ORTKMathSequenceAdapter : ONNXDataAdapter<List<*>, ORTSequence> {
    override fun fromONNXData(data: ORTSequence): List<*> {
        val elements = data.data.value
        return if (data.data.info.sequenceOfMaps) {
            (elements as List<Map<*, *>>).map { entry -> entry.mapValues { it.value!!.toScalarBuffer() } }
        } else {
            elements.map { it.toScalarBuffer() }
        }
    }

    override fun toONNXData(name: String, data: List<*>): ORTSequence {
        error("ONNXRuntime backend does not support sequence conversion")
    }
}

private fun Any.toScalarBuffer() = NDBuffer(DefaultStrides(IntArray(0)), Buffer.auto(1) { this })
