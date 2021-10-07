package io.kinference.data

interface ONNXData<T> {
    val name: String?
    val type: ONNXDataType
    val data: T

    fun rename(name: String): ONNXData<T>
}
