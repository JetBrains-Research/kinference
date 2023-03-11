package io.kinference.utils.time

class Timer(val timestamp: Timestamp) {
    companion object {
        fun start() = Timer(Timestamp.now())
        inline fun measure(body: () -> Unit): Duration {
            val timer = start()
            body()
            return timer.elapsed()
        }
    }

    fun elapsed() = Duration(Timestamp.now().millis - timestamp.millis)
}
