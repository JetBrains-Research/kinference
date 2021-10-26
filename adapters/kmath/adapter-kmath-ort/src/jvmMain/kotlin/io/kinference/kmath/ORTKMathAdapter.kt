package io.kinference.kmath

import ai.onnxruntime.*
import io.kinference.data.*
import io.kinference.ndarray.toIntArray
import io.kinference.ndarray.toLongArray
import io.kinference.ort.ORTData
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.model.ORTModel
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import java.nio.*

class ORTKMathAdapter(model: ORTModel) : ONNXModelAdapter<NDStructure<*>, ORTData<*>>(model) {
    override fun toONNXData(name: String, data: NDStructure<*>): ORTTensor {
        val tensor = data.toONNXData()
        return ORTTensor(name, tensor)
    }

    override fun fromONNXData(data: ORTData<*>): NDStructure<*> = when (data.type) {
        ONNXDataType.ONNX_TENSOR -> (data.data as OnnxTensor).toStructureND()
        else -> error("Conversion from ${data.type} is not supported by this adapter")
    }
}

fun <T> NDStructure<T>.toONNXData(): OnnxTensor {
    val env = OrtEnvironment.getEnvironment()
    val data = this.elements().map { it.second!! }.iterator()
    val linSize = this.shape.fold(1, Int::times)
    val shapeLong = this.shape.toLongArray()
    return when (val element = this.elements().first().second!!) {
        is Byte -> OnnxTensor.createTensor(env, ByteBuffer.wrap(ByteArray(linSize) { data.next() as Byte }), shapeLong)
        is Short -> OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(linSize) { data.next() as Short }), shapeLong)
        is Int -> OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(linSize) { data.next() as Int }), shapeLong)
        is Long -> OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(linSize) { data.next() as Long }), shapeLong)
        is Double -> OnnxTensor.createTensor(env, DoubleBuffer.wrap(DoubleArray(linSize) { data.next() as Double }), shapeLong)
        is Float -> OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(linSize) { data.next() as Float }), shapeLong)
        else -> error("Cannot convert from StructureND of ${element::class} to ONNXTensor")
    }
}

fun OnnxTensor.toStructureND(): NDBuffer<*> {
    val linearSize = this.info.shape.toIntArray().fold(1, Int::times)
    val buffer = when (val type = this.info.type) {
        OnnxJavaType.FLOAT -> {
            val data = this.floatBuffer
            Buffer.auto(linearSize) { data.get() }
        }
        OnnxJavaType.DOUBLE -> {
            val data = this.doubleBuffer
            Buffer.auto(linearSize) { data.get() }
        }
        OnnxJavaType.INT8 -> {
            val data = this.byteBuffer
            Buffer.auto(linearSize) { data.get() }
        }
        OnnxJavaType.INT16 -> {
            val data = this.shortBuffer
            Buffer.auto(linearSize) { data.get() }
        }
        OnnxJavaType.INT32 -> {
            val data = this.intBuffer
            Buffer.auto(linearSize) { data.get() }
        }
        OnnxJavaType.INT64 -> {
            val data = this.longBuffer
            Buffer.auto(linearSize) { data.get() }
        }
        else -> error("Unsupported data type: $type")
    }
    return NDBuffer(DefaultStrides(this.info.shape.toIntArray()), buffer)
}
