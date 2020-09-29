package io.kinference.ndarray.extensions

import io.kinference.ndarray.MutableNDArray
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.Strides
import io.kinference.primitives.types.DataType
import kotlin.math.max

data class BroadcastingTemp(val leftTemp: MutableNDArray, val rightTemp: MutableNDArray, val destinationTemp: MutableNDArray)

data class BroadcastingInfo(val array: NDArray, val offset: Int = 0) {
    fun move(additionalOffset: Int): BroadcastingInfo = BroadcastingInfo(array, offset + additionalOffset)
}

data class MutableBroadcastingInfo(val array: MutableNDArray, val offset: Int = 0) {
    fun move(additionalOffset: Int): MutableBroadcastingInfo = MutableBroadcastingInfo(array, offset + additionalOffset)
}

fun broadcastShape(firstShape: IntArray, secondShape: IntArray): IntArray {
    val totalShapeLength = max(firstShape.size, secondShape.size)

    return IntArray(totalShapeLength) { i ->
        val firstDim = firstShape.getOrNull(firstShape.size - i - 1) ?: 1
        val second = secondShape.getOrNull(secondShape.size - i - 1) ?: 1

        if (firstDim != second && firstDim != 1 && second != 1) error("Cannot broadcast shapes")

        max(firstDim, second)
    }.reversedArray()
}

fun tempShape(left: IntArray, right: IntArray): IntArray {
    var i = left.lastIndex
    while (i >= 0 && left[i] == right[i])
        i--
    return left.copyOfRange(i + 1, left.size)
}

// TODO remove to different module
fun unsqueezeFirst(shape: IntArray, newShapeSize: Int): IntArray {
    val wrapSize = newShapeSize - shape.size

    val wrappedShape = IntArray(newShapeSize)
    wrappedShape.fill(1, 0, wrapSize)
    shape.copyInto(wrappedShape, wrapSize)
    return wrappedShape
}

fun NDArray.applyWithBroadcast(other: NDArray, destination: MutableNDArray, ordered: Boolean = false, op: (NDArray, NDArray, MutableNDArray) -> Unit): MutableNDArray {
    val newShape = broadcastShape(this.shape, other.shape)

    require(destination.shape.contentEquals(newShape) && (!ordered || newShape.contentEquals(this.shape)))

    val leftShape = if (ordered) shape else unsqueezeFirst(shape, newShape.size)
    val rightShape = unsqueezeFirst(other.shape, newShape.size)

    val leftReshaped = this.reshapeView(leftShape)
    val rightReshaped = other.reshapeView(rightShape)

    val tempShape = tempShape(leftShape, rightShape)

    val leftTempArray = allocateNDArray(this.type, tempShape)
    val rightTempArray = allocateNDArray(other.type, tempShape)
    val destinationTempArray = allocateNDArray(destination.type, tempShape)

    val broadcastingTemp = BroadcastingTemp(leftTempArray, rightTempArray, destinationTempArray)

    broadcast(
        BroadcastingInfo(leftReshaped),
        BroadcastingInfo(rightReshaped),
        MutableBroadcastingInfo(destination),
        broadcastingTemp, 0, op)
    return destination
}

fun NDArray.applyWithBroadcast(other: NDArray, destType: DataType = this.type, ordered: Boolean = false, op: (NDArray, NDArray, MutableNDArray) -> Unit): MutableNDArray {
    val newShape = broadcastShape(this.shape, other.shape)
    val destination = allocateNDArray(destType, Strides(newShape))
    return applyWithBroadcast(other, destination, ordered, op)
}

fun IntArray.contentEquals(other: IntArray, offset: Int): Boolean {
    for (i in offset until size) {
        if (this[i] != other[i])
            return false
    }

    return true
}

fun broadcast(leftInfo: BroadcastingInfo,
              rightInfo: BroadcastingInfo,
              destinationInfo: MutableBroadcastingInfo,
              temp: BroadcastingTemp, index: Int,
              op: (NDArray, NDArray, MutableNDArray) -> Unit) {
    if (leftInfo.array.shape.contentEquals(rightInfo.array.shape, index)) {
        temp.leftTemp.placeFrom(0, leftInfo.array, leftInfo.offset, leftInfo.offset + temp.leftTemp.linearSize)
        temp.rightTemp.placeFrom(0, rightInfo.array, rightInfo.offset, rightInfo.offset + temp.rightTemp.linearSize)

        op(temp.leftTemp, temp.rightTemp, temp.destinationTemp)

        destinationInfo.array.placeAllFrom(destinationInfo.offset, temp.destinationTemp)
    } else {
        innerBroadcast(leftInfo, rightInfo, destinationInfo, index) { fstArray, sndArray, dest -> broadcast(fstArray, sndArray, dest, temp, index + 1, op) }
    }
}

fun innerBroadcast(leftInfo: BroadcastingInfo,
                   rightInfo: BroadcastingInfo,
                   destinationInfo: MutableBroadcastingInfo,
                   index: Int, recurrentBack: (BroadcastingInfo, BroadcastingInfo, MutableBroadcastingInfo) -> Unit) {
    if (leftInfo.array.shape[index] != rightInfo.array.shape[index]) {
        val arrayWithOne = if (leftInfo.array.shape[index] == 1) leftInfo else rightInfo
        val arrayWithoutOne = if (leftInfo.array.shape[index] != 1) leftInfo else rightInfo

        for (i in 0 until arrayWithoutOne.array.shape[index]) {
            val additionalOffsetWithoutOne = i * arrayWithoutOne.array.strides.strides[index]
            val additionalOffsetDestination = i * destinationInfo.array.strides.strides[index]

            val movedArrayWithoutOne = arrayWithoutOne.move(additionalOffsetWithoutOne)
            val movedDestination = destinationInfo.move(additionalOffsetDestination)

            if (arrayWithOne === leftInfo)
                recurrentBack(arrayWithOne, movedArrayWithoutOne, movedDestination)
            else
                recurrentBack(movedArrayWithoutOne, arrayWithOne, movedDestination)
        }
    } else {
        for (i in 0 until leftInfo.array.shape[index]) {
            val leftAdditionalOffset = i * leftInfo.array.strides.strides[index]
            val rightAdditionalOffset = i * rightInfo.array.strides.strides[index]
            val destinationAdditionalOffset = i * destinationInfo.array.strides.strides[index]

            val movedLeft = leftInfo.move(leftAdditionalOffset)
            val movedRight = rightInfo.move(rightAdditionalOffset)
            val movedDestination = destinationInfo.move(destinationAdditionalOffset)

            recurrentBack(movedLeft, movedRight, movedDestination)
        }
    }
}
