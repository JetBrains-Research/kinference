package io.kinference.utils.time

fun Collection<Duration>.average(): Duration {
    return Duration(this.sumOf { it.millis } / this.size)
}
