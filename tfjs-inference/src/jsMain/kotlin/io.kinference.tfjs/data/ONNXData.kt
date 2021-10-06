package io.kinference.tfjs.data

import io.kinference.tfjs.types.ValueInfo

abstract class ONNXData(val type: ONNXDataType, val info: ValueInfo) {
    abstract fun rename(name: String): ONNXData
}
