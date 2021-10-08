package io.kinference.tfjs.data

import io.kinference.data.ONNXData
import io.kinference.tfjs.types.ValueInfo

abstract class TFJSData<T>(override val data: T, val info: ValueInfo): ONNXData<T> {
    abstract override fun rename(name: String): TFJSData<T>
    override val name: String? = info.name
}
