package io.kinference.webgpu.ndarray

import io.kinference.utils.webgpu.Buffer

sealed class NDArrayState

class NDArrayData(val data: TypedNDArrayData) : NDArrayState()

sealed class NDArrayBuffer(val buffer: Buffer): NDArrayState()
class NDArrayUninitializedBuffer(buffer: Buffer): NDArrayBuffer(buffer)
class NDArrayInitializedBuffer(val data: TypedNDArrayData, buffer: Buffer): NDArrayBuffer(buffer)
class NDArrayCopyingBuffer(val sourceBuffer: Buffer, val destinationBuffer: Buffer): NDArrayBuffer(sourceBuffer)

object NDArrayDestroyed : NDArrayState()
