package io.kinference.kmath

import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.*
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer

object KMathStructureNDAdapter : ONNXDataAdapter<StructureND<*>> {
    override fun toONNXData(name: String, data: StructureND<*>): KITensor {
        return data.toONNXData(name)
    }

    override fun fromONNXData(data: ONNXData<*>): StructureND<*> = when (data.type) {
        ONNXDataType.ONNX_TENSOR -> (data.data as NDArray).toStructureND()
        else -> error("Conversion from ${data.type} is not supported by this adapter")
    }
}

fun <T> StructureND<T>.toONNXData(name: String): KITensor {
    val data = this.elements().map { it.second!! }.iterator()
    return when (val element = this.elements().first().second!!) {
        is Byte -> ByteNDArray(shape) { data.next() as Byte }
        is Short -> ShortNDArray(shape) { data.next() as Short }
        is Int -> IntNDArray(shape) { data.next() as Int }
        is Long -> LongNDArray(shape) { data.next() as Long }
        is UByte -> UByteNDArray(shape) { data.next() as UByte }
        is UShort -> UShortNDArray(shape) { data.next() as UShort }
        is UInt -> UIntNDArray(shape) { data.next() as UInt }
        is ULong -> ULongNDArray(shape) { data.next() as ULong }
        is Float -> FloatNDArray(shape) { data.next() as Float }
        is Double -> DoubleNDArray(shape) { data.next() as Double }
        is Boolean -> BooleanNDArray(shape) { data.next() as Boolean }
        else -> error("Cannot convert from StructureND of ${element::class} to ONNXTensor")
    }.asTensor(name)
}

fun NDArray.toStructureND(): BufferND<*> {
    val buffer = when (type) {
        DataType.BYTE -> {
            val pointer = (this as ByteNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.SHORT -> {
            val pointer = (this as ShortNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.INT -> {
            val pointer = (this as IntNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.LONG -> {
            val pointer = (this as LongNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.UBYTE -> {
            val pointer = (this as UByteNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.USHORT -> {
            val pointer = (this as UShortNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.UINT -> {
            val pointer = (this as UIntNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.ULONG -> {
            val pointer = (this as ULongNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.FLOAT -> {
            val pointer = (this as FloatNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.DOUBLE -> {
            val pointer = (this as DoubleNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        DataType.BOOLEAN -> {
            val pointer = (this as BooleanNDArray).array.pointer()
            Buffer.auto(linearSize) { pointer.getAndIncrement() }
        }
        else -> error("Usupported data type ${this.type}")
    }
    return BufferND(DefaultStrides(shape), buffer)
}
