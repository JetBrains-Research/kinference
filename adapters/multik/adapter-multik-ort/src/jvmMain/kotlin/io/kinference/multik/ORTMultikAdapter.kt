package io.kinference.multik

import ai.onnxruntime.*
import io.kinference.data.*
import io.kinference.ndarray.toIntArray
import io.kinference.ndarray.toLongArray
import io.kinference.ort.ORTData
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.ort.model.ORTModel
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import java.nio.*

class ORTMultikAdapter(model: ORTModel) : ONNXModelAdapter<MultiArray<Number, Dimension>, ORTData<*>>(model) {
    override fun toONNXData(name: String, data: MultiArray<Number, Dimension>): ORTTensor {
        val onnxTensor = data.asOrtTensor()
        return ORTTensor(name, onnxTensor)
    }

    override fun fromONNXData(data: ORTData<*>): MultiArray<Number, Dimension> = when (data.type) {
        ONNXDataType.ONNX_TENSOR -> (data as ORTTensor).data.asMultiArray()
        else -> error("Conversion from ${data.type} is not supported by this adapter")
    }
}

fun MultiArray<Number, Dimension>.asOrtTensor(): OnnxTensor {
    val env = OrtEnvironment.getEnvironment()
    return when (val array = this.data.data) {
        is DoubleArray -> OnnxTensor.createTensor(env, DoubleBuffer.wrap(array), shape.toLongArray())
        is FloatArray -> OnnxTensor.createTensor(env, FloatBuffer.wrap(array), shape.toLongArray())
        is LongArray -> OnnxTensor.createTensor(env, LongBuffer.wrap(array), shape.toLongArray())
        is IntArray -> OnnxTensor.createTensor(env, IntBuffer.wrap(array), shape.toLongArray())
        is ShortArray -> OnnxTensor.createTensor(env, ShortBuffer.wrap(array), shape.toLongArray())
        is ByteArray -> OnnxTensor.createTensor(env, ByteBuffer.wrap(array), shape.toLongArray())
        else -> error("Unsupported data type")
    }
}

fun OnnxTensor.asMultiArray(): MultiArray<Number, Dimension> {
    val dtype = this.info.type
    val view = when (dtype) {
        OnnxJavaType.FLOAT -> MemoryViewFloatArray(this.floatBuffer.array())
        OnnxJavaType.DOUBLE -> MemoryViewDoubleArray(this.doubleBuffer.array())
        OnnxJavaType.INT8 -> MemoryViewByteArray(this.byteBuffer.array())
        OnnxJavaType.INT16 -> MemoryViewShortArray(this.shortBuffer.array())
        OnnxJavaType.INT32 -> MemoryViewIntArray(this.intBuffer.array())
        OnnxJavaType.INT64 -> MemoryViewLongArray(this.longBuffer.array())
        else -> error("$dtype type is not supported by Multik")
    } as MemoryView<Number>
    return NDArray(view, shape = this.info.shape.toIntArray(), dtype = dtype.resolveMultikDataType(), dim = dimensionOf(this.info.shape.size))
}

fun OnnxJavaType.resolveMultikDataType() = when (this) {
    OnnxJavaType.FLOAT -> DataType.FloatDataType
    OnnxJavaType.DOUBLE -> DataType.DoubleDataType
    OnnxJavaType.INT8 -> DataType.ByteDataType
    OnnxJavaType.INT16 -> DataType.ShortDataType
    OnnxJavaType.INT32 -> DataType.IntDataType
    OnnxJavaType.INT64 -> DataType.LongDataType
    else -> error("$this type is not supported by Multik")
}
