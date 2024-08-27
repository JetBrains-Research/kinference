package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.memory.contexts.ManualAllocatorContext
import io.kinference.ndarray.arrays.pointers.mapTo
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.FLOAT_TENSOR_TYPES
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.coroutines.coroutineContext

sealed class Cast(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in CastVer6.VERSION.asRange() -> CastVer6(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Cast operator: $version")
        }
    }
}


class CastVer6(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Cast(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("to", setOf(AttributeProto.AttributeType.INT), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Cast", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private suspend fun castByte(array: ByteNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.UBYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> array
                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toByte() }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castShort(array: ShortNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.UBYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> array
                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toShort() }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castInt(array: IntNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.UBYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> array
                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toInt() }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castLong(array: LongNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.UBYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> array
                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != 0L }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castUByte(array: UByteNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> array
                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toUByte() }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castUShort(array: UShortNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.UBYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> array
                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toUShort() }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castUInt(array: UIntNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toUInt() }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> array
                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castULong(array: ULongNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != (0).toULong() }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> array
                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castFloat(array: FloatNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> array
                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != 0f }
                    output
                }

                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toDouble() }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castDouble(array: DoubleNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toFloat() }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong().toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt().toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toInt() }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toLong() }
                    output
                }

                TensorProto.DataType.BOOL -> {
                    val output = (context?.getNDArray(DataType.BOOLEAN, array.strides) ?: BooleanNDArray(BooleanTiledArray(array.shape), array.strides)) as BooleanNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it != 0.0 }
                    output
                }

                TensorProto.DataType.DOUBLE -> array
                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { it.toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        private suspend fun castBoolean(array: BooleanNDArray, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (to) {
                in FLOAT_TENSOR_TYPES -> {
                    val output = (context?.getNDArray(DataType.FLOAT, array.strides) ?: FloatNDArray(FloatTiledArray(array.shape), array.strides)) as FloatNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1f else 0f }
                    output
                }

                TensorProto.DataType.UINT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: UByteNDArray(UByteTiledArray(array.shape), array.strides)) as UByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toUByte() else (0).toUByte() }
                    output
                }

                TensorProto.DataType.INT8 -> {
                    val output = (context?.getNDArray(DataType.BYTE, array.strides) ?: ByteNDArray(ByteTiledArray(array.shape), array.strides)) as ByteNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toByte() else (0).toByte() }
                    output
                }

                TensorProto.DataType.UINT16 -> {
                    val output = (context?.getNDArray(DataType.USHORT, array.strides) ?: UShortNDArray(UShortTiledArray(array.shape), array.strides)) as UShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toUShort() else (0).toUShort() }
                    output
                }

                TensorProto.DataType.INT16 -> {
                    val output = (context?.getNDArray(DataType.SHORT, array.strides) ?: ShortNDArray(ShortTiledArray(array.shape), array.strides)) as ShortNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toShort() else (0).toShort() }
                    output
                }

                TensorProto.DataType.INT32 -> {
                    val output = (context?.getNDArray(DataType.INT, array.strides) ?: IntNDArray(IntTiledArray(array.shape), array.strides)) as IntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1 else 0 }
                    output
                }

                TensorProto.DataType.INT64 -> {
                    val output = (context?.getNDArray(DataType.LONG, array.strides) ?: LongNDArray(LongTiledArray(array.shape), array.strides)) as LongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1L else 0L }
                    output
                }

                TensorProto.DataType.BOOL -> array
                TensorProto.DataType.DOUBLE -> {
                    val output = (context?.getNDArray(DataType.DOUBLE, array.strides) ?: DoubleNDArray(DoubleTiledArray(array.shape), array.strides)) as DoubleNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) 1.0 else 0.0 }
                    output
                }

                TensorProto.DataType.UINT32 -> {
                    val output = (context?.getNDArray(DataType.UINT, array.strides) ?: UIntNDArray(UIntTiledArray(array.shape), array.strides)) as UIntNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toUInt() else (0).toUInt() }
                    output
                }

                TensorProto.DataType.UINT64 -> {
                    val output = (context?.getNDArray(DataType.ULONG, array.strides) ?: ULongNDArray(ULongTiledArray(array.shape), array.strides)) as ULongNDArray
                    array.array.pointer().mapTo(output.array.pointer(), array.linearSize) { if (it) (1).toULong() else (0).toULong() }
                    output
                }

                else -> throw IllegalStateException("Unsupported type")
            }
        }

        internal suspend fun castTo(input: NDArrayCore, to: TensorProto.DataType, context: ManualAllocatorContext? = null): NDArrayCore {
            return when (input.type) {
                DataType.BYTE -> castByte(input as ByteNDArray, to, context)
                DataType.SHORT -> castShort(input as ShortNDArray, to, context)
                DataType.INT -> castInt(input as IntNDArray, to, context)
                DataType.LONG -> castLong(input as LongNDArray, to, context)
                DataType.UBYTE -> castUByte(input as UByteNDArray, to, context)
                DataType.USHORT -> castUShort(input as UShortNDArray, to, context)
                DataType.UINT -> castUInt(input as UIntNDArray, to, context)
                DataType.ULONG -> castULong(input as ULongNDArray, to, context)
                DataType.FLOAT -> castFloat(input as FloatNDArray, to, context)
                DataType.DOUBLE -> castDouble(input as DoubleNDArray, to, context)
                DataType.BOOLEAN -> castBoolean(input as BooleanNDArray, to, context)
                else -> throw IllegalStateException("Unsupported type ${input.type}")
            }
        }
    }

    private val toType: Int by attribute("to") { it: Number -> it.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val manualContext = coroutineContext[ManualAllocatorContext.Key]

        val tensor = inputs.first()!!
        val to = TensorProto.DataType.fromValue(toType)!!

        val casted = castTo(tensor.data, to, manualContext)

        return listOf(casted.asTensor("output", manualContext))
    }
}
