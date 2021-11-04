package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.createArray
import io.kinference.ndarray.extensions.createPrimitiveArray
import io.kinference.ndarray.toIntArray
import io.kinference.protobuf.*
import io.kinference.protobuf.arrays.*
import okio.*

class TensorProto(
    val arrayFormat: ArrayFormat,
    var dims: IntArray = IntArray(0),
    var dataType: DataType? = null,
    var segment: Segment? = null,
    val stringData: MutableList<ByteString> = ArrayList(),
    var name: String? = null,
    val externalData: MutableList<StringStringEntryProto> = ArrayList(),
    var dataLocation: DataLocation? = null
) {
    private var _arrayData: ArrayContainer? = arrayFormat.container()

    val arrayData: Any?
        get() = _arrayData!!.get(dims)

    fun isTiled(): Boolean = arrayFormat == ArrayFormat.TILED
    fun isPrimitive(): Boolean = arrayFormat == ArrayFormat.PRIMITIVE
    fun isString(): Boolean = stringData.isNotEmpty()

    companion object {
        private val int32AvailableTypes = setOf(DataType.BOOL, DataType.INT8, DataType.UINT8, DataType.INT16, DataType.UINT16)

        fun decode(reader: ProtobufReader): TensorProto {
            val proto = TensorProto(reader.config.tensorFormat)
            var rawData: ByteString? = null
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.DIMS -> proto.dims = reader.readLongArray(tag).toIntArray()
                    ReaderTag.DATATYPE -> proto.dataType = reader.readValue(DataType.ADAPTER)
                    ReaderTag.SEGMENT -> proto.segment = Segment.decode(reader)
                    ReaderTag.FLOAT -> reader.readFloatArray(tag, proto.dims, proto._arrayData!!)
                    ReaderTag.INT32 -> reader.readIntArray(tag, proto.dims, proto._arrayData!!)
                    ReaderTag.STRING -> proto.stringData.add(reader.readBytes())
                    ReaderTag.INT64 -> reader.readLongArray(tag, proto.dims, proto._arrayData!!)
                    ReaderTag.NAME -> proto.name = reader.readString()
                    ReaderTag.RAW -> rawData = reader.readBytes()
                    ReaderTag.DOUBLE -> reader.readDoubleArray(tag, proto.dims, proto._arrayData!!)
                    ReaderTag.UINT64 -> reader.readULongArray(tag, proto.dims, proto._arrayData!!)
                    ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                    ReaderTag.EXTERNAL -> proto.externalData.add(StringStringEntryProto.decode(reader))
                    ReaderTag.LOCATION -> try {
                        proto.dataLocation = reader.readValue(DataLocation.ADAPTER)
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

        private fun TensorProto.hasData() = _arrayData.hasData() || stringData.isNotEmpty() || externalData.isNotEmpty()

        // convert data stored as int32 to the specified type
        private fun TensorProto.checkArrayData() {
            if (this.arrayData !is IntTiledArray && this.arrayData !is IntArray) return
            if (this.dataType == DataType.INT32) return

            require(dataType in int32AvailableTypes) { "Conversion from int32 to $dataType is not supported" }
            val newArray = when (arrayFormat) {
                ArrayFormat.TILED -> parseInt32TiledData()
                ArrayFormat.PRIMITIVE -> parseInt32PrimitiveData()
                ArrayFormat.OTHER -> error("Cannot read the array. Please, specify tensor backing array type as either TILED or PRIMITIVE")
            }
            _arrayData!!.setData(newArray)
        }

        private fun TensorProto.parseInt32TiledData(): Any {
            val data = arrayData as IntTiledArray
            val pointer = data.pointer()

            @Suppress("IMPLICIT_CAST_TO_ANY")
            return when (dataType) {
                DataType.BOOL -> BooleanTiledArray(dims) { pointer.getAndIncrement() != 0 }
                DataType.INT8 -> ByteTiledArray(dims) { pointer.getAndIncrement().toByte() }
                DataType.UINT8 -> UByteTiledArray(dims) { pointer.getAndIncrement().toUByte() }
                DataType.INT16 -> ShortTiledArray(dims) { pointer.getAndIncrement().toShort() }
                DataType.UINT16 -> UShortTiledArray(dims) { pointer.getAndIncrement().toUShort() }
                else -> error("Conversion from int32 to $dataType is not supported")
            }
        }

        private fun TensorProto.parseInt32PrimitiveData(): Any {
            val data = arrayData as IntArray
            val size = data.size

            @Suppress("IMPLICIT_CAST_TO_ANY")
            return when (dataType) {
                DataType.BOOL -> BooleanArray(size) { data[it] != 0 }
                DataType.INT8 -> ByteArray(size) { data[it].toByte() }
                DataType.UINT8 -> UByteArray(size) { data[it].toUByte() }
                DataType.INT16 -> ShortArray(size) { data[it].toShort() }
                DataType.UINT16 -> UShortArray(size) { data[it].toUShort() }
                else -> error("Conversion from int32 to $dataType is not supported")
            }
        }

        private fun makeArray(arrayFormat: ArrayFormat, type: DataType, shape: IntArray, init: (Int) -> Any) = when (arrayFormat) {
            ArrayFormat.TILED -> createArray(type.resolveLocalDataType(), shape, init)
            ArrayFormat.PRIMITIVE -> createPrimitiveArray(type.resolveLocalDataType(), shape, init)
            ArrayFormat.OTHER -> error("Cannot read the array. Please, specify tensor backing array type as either TILED or PRIMITIVE")
        }

        private fun parseRaw(rawData: ByteString?, proto: TensorProto) {
            require(proto._arrayData != null)
            val raw = rawData ?: ByteString.EMPTY
            val buffer = Buffer().apply { write(raw) }
            val shape = proto.dims

            when (proto.dataType) {
                DataType.DOUBLE -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readDoubleLe() })
                DataType.FLOAT -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readFloatLe() })
                DataType.INT64 -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readLongLe() })
                DataType.INT32 -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readIntLe() })
                DataType.INT16 -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readShortLe() })
                DataType.UINT16 -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readShortLe().toUShort() })
                DataType.INT8 -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readByte() })
                DataType.UINT8 -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readByte().toUByte() })
                DataType.BOOL -> proto._arrayData!!.setData(makeArray(proto.arrayFormat, proto.dataType!!, shape) { buffer.readByte() != 0.toByte() })
                DataType.STRING -> error("String data must not be present in rawData field")
                else -> error("Unsupported data type ${proto.dataType}")
            }
        }
    }

    private enum class ReaderTag(val tag: Int) {
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
