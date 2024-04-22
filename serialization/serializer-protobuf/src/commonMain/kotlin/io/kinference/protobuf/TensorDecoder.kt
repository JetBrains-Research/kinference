package io.kinference.protobuf

import com.squareup.wire.FieldEncoding
import com.squareup.wire.ProtoAdapter
import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.message.StringStringEntryProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.Companion.hasData
import io.kinference.utils.inlines.InlineInt
import io.kinference.utils.toIntArray
import okio.Buffer
import okio.ByteString

abstract class TensorDecoder {
    protected abstract fun initContainer(): ArrayContainer
    protected abstract suspend fun makeArray(type: TensorProto.DataType, shape: IntArray, init: (InlineInt) -> Any): Any
    protected abstract suspend fun parseInt32Data(proto: TensorProto): Any
    protected abstract suspend fun hasIntArray(proto: TensorProto): Boolean

    private suspend fun TensorProto.checkArrayData() {
        if (!hasIntArray(this)) return
        if (dataType == TensorProto.DataType.INT32) return

        require(dataType in int32AvailableTypes) { "Conversion from int32 to $dataType is not supported" }
        val newArray = parseInt32Data(this)
        _arrayData!!.setData(newArray)
    }

    suspend fun decode(reader: ProtobufReader): TensorProto {
        val proto = TensorProto(_arrayData = initContainer())
        var rawData: ByteString? = null
        reader.forEachTag { tag ->
            when (TensorProto.ReaderTag.fromInt(tag)) {
                TensorProto.ReaderTag.DIMS -> proto.dims = reader.readLongArray(tag).toIntArray()
                TensorProto.ReaderTag.DATATYPE -> proto.dataType = reader.readValue(TensorProto.DataType.ADAPTER)
                TensorProto.ReaderTag.SEGMENT -> proto.segment = TensorProto.Segment.decode(reader)
                TensorProto.ReaderTag.FLOAT -> reader.readFloatArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.INT32 -> reader.readIntArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.STRING -> proto.stringData.add(reader.readBytes())
                TensorProto.ReaderTag.INT64 -> reader.readLongArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.NAME -> proto.name = reader.readString()
                TensorProto.ReaderTag.RAW -> rawData = reader.readBytes()
                TensorProto.ReaderTag.DOUBLE -> reader.readDoubleArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.UINT64 -> reader.readULongArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                TensorProto.ReaderTag.EXTERNAL -> proto.externalData.add(StringStringEntryProto.decode(reader))
                TensorProto.ReaderTag.LOCATION -> try {
                    proto.dataLocation = reader.readValue(TensorProto.DataLocation.ADAPTER)
                } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                    reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                }
                null -> reader.readUnknownField(tag)
            }
        }
        if (rawData != null || !proto.hasData()) parseRaw(rawData, proto)
        proto.checkArrayData()
        return proto
    }

    private suspend fun parseRaw(rawData: ByteString?, proto: TensorProto) {
        require(proto._arrayData != null)
        val raw = rawData ?: ByteString.EMPTY
        val buffer = Buffer().apply { write(raw) }
        val shape = proto.dims

        when (proto.dataType) {
            TensorProto.DataType.DOUBLE -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readDoubleLe() })
            TensorProto.DataType.FLOAT -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readFloatLe() })
            TensorProto.DataType.INT64 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readLongLe() })
            TensorProto.DataType.INT32 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readIntLe() })
            TensorProto.DataType.INT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe() })
            TensorProto.DataType.UINT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe().toUShort() })
            TensorProto.DataType.INT8 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readByte() })
            TensorProto.DataType.UINT8 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readByte().toUByte() })
            TensorProto.DataType.BOOL -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readByte() != 0.toByte() })
            TensorProto.DataType.BFLOAT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe().parseAsBFloat() })
            TensorProto.DataType.FLOAT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe().parseAsFloat16() })
            TensorProto.DataType.UINT32 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readIntLe().toUInt() })
            TensorProto.DataType.UINT64 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readLongLe().toULong() })
            TensorProto.DataType.STRING -> error("String data must not be present in rawData field")
            else -> error("Unsupported data type ${proto.dataType}")
        }
    }

    companion object {
        private val int32AvailableTypes = setOf(
            TensorProto.DataType.BOOL, TensorProto.DataType.INT8,
            TensorProto.DataType.UINT8, TensorProto.DataType.INT16,
            TensorProto.DataType.UINT16, TensorProto.DataType.BFLOAT16,
            TensorProto.DataType.FLOAT16
        )

        private const val FLOAT32_FRAC_BITS = 23

        private const val FLOAT16_FRAC_MASK = (1 shl 10) - 1
        private const val FLOAT16_EXP_MASK = 0x7c00
        private const val FLOAT16_EXP_BIAS = 0x38000000

        //float32 frac bits - float16 frac bits == 23 - 10 == 13
        private const val FLOAT16_SHIFT_BITS = 13

        private const val BFLOAT_FRAC_MASK = (1 shl 7) - 1
        private const val BFLOAT_EXP_MASK = 0x7f80

        private const val UNSIGNED_16_BIT_MASK = 0x7fff
        private const val SIGN_16_BIT_MASK = 0x8000

        //bfloat size = 16 bits, exp = 8 bits, frac = 7 bits
        fun Int.parseAsBFloat(): Float {
            val valueUnsigned = this and UNSIGNED_16_BIT_MASK
            //move sign bit to leftmost position
            val sign = (this and SIGN_16_BIT_MASK) shl 16

            //move exp and frac bits to their corresponding positions in 32-bit representation
            val exponent = (valueUnsigned and BFLOAT_EXP_MASK) shl 16
            val frac = (valueUnsigned and BFLOAT_FRAC_MASK) shl 16

            return Float.fromBits(sign or exponent or frac)
        }

        fun Short.parseAsBFloat(): Float = this.toInt().parseAsBFloat()

        //float16 size = 16 bits, exp = 5 bits, frac = 10 bits
        fun Int.parseAsFloat16(): Float {
            val valueUnsigned = this and UNSIGNED_16_BIT_MASK
            //move sign bit to leftmost position
            val sign = (this and SIGN_16_BIT_MASK) shl 16

            val exponent = valueUnsigned and FLOAT16_EXP_MASK
            val frac = valueUnsigned and FLOAT16_FRAC_MASK
            if (exponent == 0) {
                if (frac == 0) return Float.fromBits(sign)

                //process denormalized values
                var exponentDenormalized = 127 - 14
                var fracDenormalized = frac
                while (fracDenormalized and (FLOAT16_FRAC_MASK + 1) == 0) {
                    exponentDenormalized--
                    fracDenormalized = fracDenormalized shl 1
                }
                fracDenormalized = (fracDenormalized and FLOAT16_FRAC_MASK) shl FLOAT16_SHIFT_BITS
                exponentDenormalized = exponentDenormalized shl FLOAT32_FRAC_BITS
                return Float.fromBits(sign or exponentDenormalized or fracDenormalized)
            } else if (exponent == FLOAT16_EXP_MASK) { // check if all exponent bits are set to 1
                return Float.fromBits(sign or 0x7f800000 or frac)
            }

            //move exp and frac bits to their corresponding positions in 32-bit representation
            val valueShift = (valueUnsigned shl FLOAT16_SHIFT_BITS) + FLOAT16_EXP_BIAS

            return Float.fromBits(valueShift or sign)
        }

        fun Short.parseAsFloat16(): Float = this.toInt().parseAsFloat16()
    }
}
