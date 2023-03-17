package io.kinference.protobuf

import com.squareup.wire.FieldEncoding
import com.squareup.wire.ProtoAdapter
import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.message.StringStringEntryProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.Companion.hasData
import okio.Buffer
import okio.ByteString
import kotlin.math.pow

abstract class TensorDecoder {
    protected abstract fun initContainer(): ArrayContainer
    protected abstract fun makeArray(type: TensorProto.DataType, shape: IntArray, init: (Int) -> Any): Any
    protected abstract fun parseInt32Data(proto: TensorProto): Any
    protected abstract fun hasIntArray(proto: TensorProto): Boolean

    private fun TensorProto.checkArrayData() {
        if (!hasIntArray(this)) return
        if (dataType == TensorProto.DataType.INT32) return

        require(dataType in int32AvailableTypes) { "Conversion from int32 to $dataType is not supported" }
        val newArray = parseInt32Data(this)
        _arrayData!!.setData(newArray)
    }

    fun decode(reader: ProtobufReader): TensorProto {
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

    private fun parseRaw(rawData: ByteString?, proto: TensorProto) {
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
            TensorProto.DataType.BFLOAT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe().toInt().parseAsBFloat() })
            TensorProto.DataType.FLOAT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe().toInt().parseAsFloat16() })
            TensorProto.DataType.STRING -> error("String data must not be present in rawData field")
            else -> error("Unsupported data type ${proto.dataType}")
        }
    }

    companion object {
        private const val BFLOAT_EXP_SIZE = 8
        private const val BFLOAT_MANTISSA_SIZE = 7

        private const val FLOAT16_EXP_SIZE = 5
        private const val FLOAT16_MANTISSA_SIZE = 10

        private val SIGN_MASK = 1 shl 31

        private val int32AvailableTypes = setOf(
            TensorProto.DataType.BOOL, TensorProto.DataType.INT8,
            TensorProto.DataType.UINT8, TensorProto.DataType.INT16,
            TensorProto.DataType.UINT16, TensorProto.DataType.BFLOAT16,
            TensorProto.DataType.FLOAT16
        )

        // exp = 8 frac = 7

        fun Int.parseAsBFloat(): Float {
            val exponent = this shr 7
            val precision = this - (exponent shl 7)
            val sign = (exponent and 256) == 256
            val power = (exponent and 255) - 127
            val rawValue = ((precision * 2.0.pow(-7) + 1) * 2.0.pow(power)).toFloat()
            return if (sign) -rawValue else rawValue
        }

        //exp = 5 frac = 10

        fun Int.parseAsFloat16(): Float {
            val exponent = this shr 10
            val precision = this - (exponent shl 10)
            val sign = (exponent and 32) == 32
            val power = (exponent and 31) - 15
            val rawValue = ((precision * 2.0.pow(-10) + 1) * 2.0.pow(power)).toFloat()
            return if (sign) -rawValue else rawValue
        }
    }
}
