package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.arrays.hasData
import okio.*

class TensorProto internal constructor(
    var dims: IntArray = IntArray(0),
    var dataType: DataType? = null,
    var segment: Segment? = null,
    val stringData: MutableList<ByteString> = ArrayList(),
    var name: String? = null,
    val externalData: MutableList<StringStringEntryProto> = ArrayList(),
    var dataLocation: DataLocation? = null,
    internal var _arrayData: ArrayContainer? = null
) {

    val arrayData: Any?
        get() = _arrayData!!.get(dims)

    fun isString(): Boolean = stringData.isNotEmpty()

    companion object {
        internal fun TensorProto.hasData() = _arrayData.hasData() || stringData.isNotEmpty() || externalData.isNotEmpty()
    }

    internal enum class ReaderTag(val tag: Int) {
        DIMS(1),
        DATATYPE(2),
        SEGMENT(3),
        FLOAT(4),
        INT32(5),
        STRING(6),
        INT64(7),
        NAME(8),
        RAW(9),
        DOUBLE(10),
        UINT64(11),
        DOC_STRING(12),
        EXTERNAL(13),
        LOCATION(14);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }

    enum class DataType(override val value: Int) : WireEnum {
        UNDEFINED(0),
        FLOAT(1),
        UINT8(2),
        INT8(3),
        UINT16(4),
        INT16(5),
        INT32(6),
        INT64(7),
        STRING(8),
        BOOL(9),
        FLOAT16(10),
        DOUBLE(11),
        UINT32(12),
        UINT64(13),
        COMPLEX64(14),
        COMPLEX128(15),
        BFLOAT16(16);

        companion object {
            val ADAPTER: ProtoAdapter<DataType> = object : EnumAdapter<DataType>(DataType::class, Syntax.PROTO_2, UNDEFINED) {
                override fun fromValue(value: Int): DataType? = DataType.fromValue(value)
            }

            fun fromValue(value: Int): DataType? = when (value) {
                0 -> UNDEFINED
                1 -> FLOAT
                2 -> UINT8
                3 -> INT8
                4 -> UINT16
                5 -> INT16
                6 -> INT32
                7 -> INT64
                8 -> STRING
                9 -> BOOL
                10 -> FLOAT16
                11 -> DOUBLE
                12 -> UINT32
                13 -> UINT64
                14 -> COMPLEX64
                15 -> COMPLEX128
                16 -> BFLOAT16
                else -> null
            }
        }
    }

    data class Segment(val begin: Long? = null, val end: Long? = null) {
        companion object {
            fun decode(reader: ProtobufReader): Segment {
                var begin: Long? = null
                var end: Long? = null
                reader.forEachTag { tag ->
                    when (ReaderTag.fromInt(tag)) {
                        ReaderTag.BEGIN -> begin = reader.readLong()
                        ReaderTag.END -> end = reader.readLong()
                        null -> reader.readUnknownField(tag)
                    }
                }
                return Segment(begin = begin, end = end)
            }
        }

        private enum class ReaderTag(val tag: Int) {
            BEGIN(1),
            END(2);

            companion object {
                fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
            }
        }
    }

    enum class DataLocation(override val value: Int) : WireEnum {
        DEFAULT(0),
        EXTERNAL(1);

        companion object {
            val ADAPTER: ProtoAdapter<DataLocation> = object : EnumAdapter<DataLocation>(DataLocation::class, Syntax.PROTO_2, DEFAULT) {
                override fun fromValue(value: Int): DataLocation = DataLocation.fromValue(value)
            }

            fun fromValue(value: Int): DataLocation = when (value) {
                0 -> DEFAULT
                1 -> EXTERNAL
                else -> error("Cannot convert from value")
            }
        }
    }
}
