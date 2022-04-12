package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.mapTo
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class Cast(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in CastVer6.VERSION.asRange() -> CastVer6(attributes, inputs, outputs)
            else -> error("Unsupported version of Cast operator: $version")
        }
    }
}

@ExperimentalTime
class CastVer6(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Cast(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("to", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Cast", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val toType: Int by attribute("to") { it: Number -> it.toInt() }


    private fun castByte(array: ByteNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> array
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toByte() }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }

    private fun castShort(array: ShortNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> array
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toShort() }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castInt(array: IntNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                output
            }
            TensorProto.DataType.INT32 -> array
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != 0 }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castLong(array: LongNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> array
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != 0L }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castUByte(array: UByteNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> array
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toUByte() }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castUShort(array: UShortNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> array
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toUShort() }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castUInt(array: UIntNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toUInt() }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> array
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castULong(array: ULongNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toULong() }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> array
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castFloat(array: FloatNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> array
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != 0f }
                output
            }
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castDouble(array: DoubleNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                output
            }
            TensorProto.DataType.BOOL -> {
                val output = BooleanNDArray(BooleanTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != 0.0 }
                output
            }
            TensorProto.DataType.DOUBLE -> array
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }


    private fun castBoolean(array: BooleanNDArray, to: TensorProto.DataType): NDArray {
        return when (to) {
            TensorProto.DataType.FLOAT -> {
                val output = FloatNDArray(FloatTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1f else 0f }
                output
            }
            TensorProto.DataType.UINT8 -> {
                val output = UByteNDArray(UByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toUByte() else (0).toUByte() }
                output
            }
            TensorProto.DataType.INT8 -> {
                val output = ByteNDArray(ByteTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toByte() else (0).toByte() }
                output
            }
            TensorProto.DataType.UINT16 -> {
                val output = UShortNDArray(UShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toUShort() else (0).toUShort() }
                output
            }
            TensorProto.DataType.INT16 -> {
                val output = ShortNDArray(ShortTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toShort() else (0).toShort() }
                output
            }
            TensorProto.DataType.INT32 -> {
                val output = IntNDArray(IntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1 else 0 }
                output
            }
            TensorProto.DataType.INT64 -> {
                val output = LongNDArray(LongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1L else 0L }
                output
            }
            TensorProto.DataType.BOOL -> array
            TensorProto.DataType.DOUBLE -> {
                val output = DoubleNDArray(DoubleTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1.0 else 0.0 }
                output
            }
            TensorProto.DataType.UINT32 -> {
                val output = UIntNDArray(UIntTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toUInt() else (0).toUInt() }
                output
            }
            TensorProto.DataType.UINT64 -> {
                val output = ULongNDArray(ULongTiledArray(array.shape), array.strides)
                array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toULong() else (0).toULong() }
                output
            }
            else -> throw IllegalStateException("Unsupported type")
        }
    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val tensor = inputs.first()!!
        val to = TensorProto.DataType.fromValue(toType)!!

        val casted: NDArray = when (tensor.data.type) {
            DataType.BYTE -> castByte(tensor.data as ByteNDArray, to)
            DataType.SHORT -> castShort(tensor.data as ShortNDArray, to)
            DataType.INT -> castInt(tensor.data as IntNDArray, to)
            DataType.LONG -> castLong(tensor.data as LongNDArray, to)
            DataType.UBYTE -> castUByte(tensor.data as UByteNDArray, to)
            DataType.USHORT -> castUShort(tensor.data as UShortNDArray, to)
            DataType.UINT -> castUInt(tensor.data as UIntNDArray, to)
            DataType.ULONG -> castULong(tensor.data as ULongNDArray, to)
            DataType.FLOAT -> castFloat(tensor.data as FloatNDArray, to)
            DataType.DOUBLE -> castDouble(tensor.data as DoubleNDArray, to)
            DataType.BOOLEAN -> castBoolean(tensor.data as BooleanNDArray, to)
            else -> throw IllegalStateException("Unsupported type ${tensor.data.type}")
        }

        return listOf(casted.asTensor("output"))
    }
}
