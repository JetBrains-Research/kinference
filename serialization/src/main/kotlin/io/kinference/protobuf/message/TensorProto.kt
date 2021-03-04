package io.kinference.protobuf.message

import com.squareup.wire.*
import io.kinference.ndarray.arrays.pointers.IntPointer
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.toIntArray
import io.kinference.protobuf.*
import io.kinference.protobuf.arrays.TiledArrayContainer
import okio.Buffer
import okio.ByteString
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TensorProto(
    var dims: IntArray = IntArray(0),
    var dataType: DataType? = null,
    var segment: Segment? = null,
    val stringData: MutableList<ByteString> = ArrayList(),
    var name: String? = null,
    val externalData: MutableList<StringStringEntryProto> = ArrayList(),
    var dataLocation: DataLocation? = null,
) {
    private var _tiledData: TiledArrayContainer = TiledArrayContainer()

    val tiledData: Any?
        get() = _tiledData.get(dims)

    fun isTiled(): Boolean = _tiledData.hasData()
    fun isString(): Boolean = stringData.isNotEmpty()

    companion object {
        private val int32AvailableTypes = setOf(DataType.BOOL, DataType.INT8, DataType.UINT8, DataType.INT8, DataType.UINT8)

        fun decode(byteArray: ByteArray): TensorProto {
            val buffer = Buffer().write(byteArray)
            return decode(ProtobufReader(buffer))
        }

        fun decode(reader: ProtobufReader): TensorProto {
            val proto = TensorProto()
            var rawData: ByteString? = null
            reader.forEachTag { tag ->
                when (tag) {
                    1 -> proto.dims = reader.readLongArray(tag).toIntArray()
                    2 -> proto.dataType = DataType.fromValue(reader.readInt())
                    3 -> proto.segment = Segment.decode(reader)
                    4 -> reader.readFloatTiledArray(tag, proto.dims, proto._tiledData)
                    5 -> reader.readIntTiledArray(tag, proto.dims, proto._tiledData)
                    6 -> proto.stringData.add(reader.readBytes())
                    7 -> reader.readLongTiledArray(tag, proto.dims, proto._tiledData)
                    8 -> proto.name = reader.readString()
                    9 -> rawData = reader.readBytes()
                    10 -> reader.readDoubleTiledArray(tag, proto.dims, proto._tiledData)
                    11 -> reader.readULongTiledArray(tag, proto.dims, proto._tiledData)
                    12 -> reader.readString() // skip docstring
                    13 -> proto.externalData.add(StringStringEntryProto.decode(reader))
                    14 -> try {
                        proto.dataLocation = reader.readValue(DataLocation.ADAPTER)
                    } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                        reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                    }
                    else -> reader.readUnknownField(tag)
                }
            }
            if (rawData != null || !proto.hasData()) parseRaw(rawData, proto)
            proto.checkTiledData()
            return proto
        }

        private fun TensorProto.hasData() = _tiledData.hasData() || stringData.isNotEmpty() || externalData.isNotEmpty()

        // convert data stored as int32 to the specified type
        private fun TensorProto.checkTiledData() {
            if (this.tiledData !is IntTiledArray || this.dataType == DataType.INT32) return
            require(dataType in int32AvailableTypes) { "Conversion from int32 to $dataType is not supported" }

            val data = tiledData as IntTiledArray
            val pointer = IntPointer(data)

            @Suppress("IMPLICIT_CAST_TO_ANY")
            val newTiled = when (dataType) {
                DataType.BOOL -> BooleanTiledArray(dims) { pointer.getAndIncrement() != 0 }
                DataType.INT8 -> ByteTiledArray(dims) { pointer.getAndIncrement().toByte() }
                DataType.UINT8 -> UByteTiledArray(dims) { pointer.getAndIncrement().toUByte() }
                DataType.INT16 -> ShortTiledArray(dims) { pointer.getAndIncrement().toShort() }
                DataType.UINT16 -> UShortTiledArray(dims) { pointer.getAndIncrement().toUShort() }
                else -> error("Conversion from int32 to $dataType is not supported")
            }
            _tiledData.setTiled(newTiled)
        }

        private fun parseRaw(rawData: ByteString?, proto: TensorProto) {
            val raw = rawData ?: ByteString.EMPTY
            val buffer = ByteBuffer.wrap(raw.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
            val shape = proto.dims
            when (proto.dataType) {
                DataType.DOUBLE -> {
                    val array = buffer.asDoubleBuffer()
                    proto._tiledData.setTiled(DoubleTiledArray(shape) { array[it] })
                }
                DataType.FLOAT -> {
                    val array = buffer.asFloatBuffer()
                    proto._tiledData.setTiled(FloatTiledArray(shape) { array[it] })
                }
                DataType.INT64 -> {
                    val array = buffer.asLongBuffer()
                    proto._tiledData.setTiled(LongTiledArray(shape) { array[it] })
                }
                DataType.INT32 -> {
                    val array = buffer.asIntBuffer()
                    proto._tiledData.setTiled(IntTiledArray(shape) { array[it] })
                }
                DataType.INT16 -> {
                    val array = buffer.asShortBuffer()
                    proto._tiledData.setTiled(ShortTiledArray(shape) { array[it] })
                }
                DataType.UINT16 -> {
                    val array = buffer.asShortBuffer()
                    proto._tiledData.setTiled(UShortTiledArray(shape) { array[it].toUShort() })
                }
                DataType.INT8 -> proto._tiledData.setTiled(ByteTiledArray(shape) { buffer[it] })
                DataType.UINT8 -> proto._tiledData.setTiled(UByteTiledArray(shape) { buffer[it].toUByte() })
                DataType.BOOL -> proto._tiledData.setTiled(BooleanTiledArray(shape) { buffer[it] != 0.toByte() })
                DataType.STRING -> error("String data must not be present in rawData field")
                else -> error("Unsupported data type ${proto.dataType}")
            }
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

    data class Segment(val begin: Long? = null, val end: Long? = null) {
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
