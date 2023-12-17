package io.kinference.ndarray.arrays

import kotlinx.coroutines.CoroutineScope
import kotlin.coroutines.AbstractCoroutineContextElement
import kotlin.coroutines.CoroutineContext

interface NDArrayDispatcher {
    fun trackCreation()

    companion object {

        private val arrayInfo: MutableMap<String, MutableMap<Int, Int>> = mutableMapOf()

        fun logCreation(type: String, size: Int) {
            // Log general creation details here
            val sizeCountMap = arrayInfo.getOrPut(type) { mutableMapOf() }
            val currentCount = sizeCountMap.getOrPut(size) { 0 }
            sizeCountMap[size] = currentCount + 1
        }

        fun printProfile() {
            for ((type, sizeMap) in arrayInfo) {
                for ((size, count) in sizeMap) {
                    println("Type: $type, Size: $size, Count: $count")
                }
            }
        }
    }
}
//
//class OperatorContext(val name: String) : AbstractCoroutineContextElement(Key) {
//    companion object Key : CoroutineContext.Key<OperatorContext>
//}
