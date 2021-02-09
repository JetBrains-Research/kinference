package io.kinference

fun <T> MutableSet<T>.removeIf(body: (T) -> Boolean) {
    val filtered = this.filter(body)
    this.removeAll(filtered)
}
