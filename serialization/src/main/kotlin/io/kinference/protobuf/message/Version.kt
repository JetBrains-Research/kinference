package io.kinference.protobuf.message

import com.squareup.wire.*

enum class Version(override val value: Int) : WireEnum {
    _START_VERSION(0),
    IR_VERSION_2017_10_10(1),
    IR_VERSION_2017_10_30(2),
    IR_VERSION_2017_11_3(3),
    IR_VERSION_2019_1_22(4),
    IR_VERSION_2019_3_18(5),
    IR_VERSION_2019_9_19(6),
    IR_VERSION(7);

    companion object {
        val ADAPTER: ProtoAdapter<Version> = object : EnumAdapter<Version>(Version::class) {
            override fun fromValue(value: Int): Version? = Version.fromValue(value)
        }

        fun fromValue(value: Int): Version? = when (value) {
            0 -> _START_VERSION
            1 -> IR_VERSION_2017_10_10
            2 -> IR_VERSION_2017_10_30
            3 -> IR_VERSION_2017_11_3
            4 -> IR_VERSION_2019_1_22
            5 -> IR_VERSION_2019_3_18
            6 -> IR_VERSION_2019_9_19
            7 -> IR_VERSION
            else -> error("Cannot convert from value $value")
        }
    }
}
