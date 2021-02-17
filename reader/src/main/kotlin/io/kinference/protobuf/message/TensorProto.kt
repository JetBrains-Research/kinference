package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.protobuf.ProtobufReader
import io.kinference.protobuf.arrays.*
import okio.Buffer
import okio.ByteString

class TensorProto(
    //ProtoTag = 1
    var dims: LongArray? = null,

    //ProtoTag = 2
    var data_type: Int? = null,

    //ProtoTag = 3
    var segment: Segment? = null,

    //ProtoTag = 4
    var float_data: FloatArray? = null,

    //ProtoTag = 5
    var int32_data: IntArray? = null,

    //ProtoTag = 6
    val string_data: MutableList<ByteString> = ArrayList(),

    //ProtoTag = 7
    var int64_data: LongArray? = null,

    //ProtoTag = 8
    var name: String? = null,

    //ProtoTag = 12
    var doc_string: String? = null,

    //ProtoTag = 9
    var raw_data: ByteString? = null,

    //ProtoTag = 13
    val external_data: MutableList<StringStringEntryProto> = ArrayList(),

    //ProtoTag = 14
    var data_location: DataLocation? = null,

    //ProtoTag = 10
    var double_data: DoubleArray? = null,

    //ProtoTag = 11
    var uint64_data: LongArray? = null
) {
    companion object {
        fun decode(byteArray: ByteArray): TensorProto {
            val buffer = Buffer().write(byteArray)
            return decode(ProtobufReader(buffer))
        }

        fun decode(reader: ProtobufReader): TensorProto {
            val proto = TensorProto()
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.dims = LongArrayDeserializer.decode(reader, tag)
                    2 -> proto.data_type = reader.readInt()
                    3 -> proto.segment = Segment.decode(reader)
                    4 -> proto.float_data = FloatArrayDeserializer.decode(reader, tag)
                    5 -> proto.int32_data = IntArrayDeserializer.decode(reader, tag)
                    6 -> proto.string_data.add(reader.readBytes())
                    7 -> proto.int64_data = LongArrayDeserializer.decode(reader, tag)
                    8 -> proto.name = reader.readString()
                    12 -> proto.doc_string = reader.readString()
                    9 -> proto.raw_data = reader.readBytes()
                    13 -> proto.external_data.add(StringStringEntryProto.decode(reader))
                    14 -> try {
                        proto.data_location = reader.readValue(DataLocation.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    10 -> proto.double_data = DoubleArrayDeserializer.decode(reader, tag)
                    11 -> proto.uint64_data = ULongArrayDeserializer.decode(reader, tag)
                    else -> reader.readUnknownField(tag)
                }
            }
            return proto
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
            val ADAPTER: ProtoAdapter<DataType> = object : EnumAdapter<DataType>(DataType::class) {
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

    class Segment(
        //ProtoTag = 1
        val begin: Long? = null,

        //ProtoTag = 2
        val end: Long? = null
    ) {
        companion object {
            fun decode(reader: ProtobufReader): Segment {
                var begin: Long? = null
                var end: Long? = null
                reader.forEachTag { tag ->
                    when (tag) {
                        1 -> begin = reader.readLong()
                        2 -> end = reader.readLong()
                        else -> reader.readUnknownField(tag)
                    }
                }
                return Segment(begin = begin, end = end)
            }
        }
    }

    enum class DataLocation(override val value: Int) : WireEnum {
        DEFAULT(0),
        EXTERNAL(1);

        companion object {
            val ADAPTER: ProtoAdapter<DataLocation> = object : EnumAdapter<DataLocation>(DataLocation::class) {
                override fun fromValue(value: Int): DataLocation? = DataLocation.fromValue(value)
            }

            fun fromValue(value: Int): DataLocation? = when (value) {
                0 -> DEFAULT
                1 -> EXTERNAL
                else -> null
            }
        }
    }
}
