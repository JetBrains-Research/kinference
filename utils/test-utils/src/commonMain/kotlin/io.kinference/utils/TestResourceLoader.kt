package io.kinference.utils

interface TestDataLoader {
    class Path(parts: List<String>) {
        private val parts = parts.map { it.trim().trim('/') }.filter { it.isNotBlank() }

        constructor(vararg path: String): this(path.flatMap { it.split("/") })
        constructor(path: String): this(path.split("/"))
        constructor(path: Path, part: String) : this(path.parts + part)
        constructor(part: String, path: Path) : this(listOf(part) + path.parts)

        fun toRelativePath() = parts.joinToString(separator = "/")
        fun toAbsolutePath() = parts.joinToString(prefix = "/", separator = "/")
    }

    suspend fun bytes(path: Path): ByteArray
    suspend fun text(path: Path): String
}

expect object ResourcesTestDataLoader : TestDataLoader {
    override suspend fun bytes(path: TestDataLoader.Path): ByteArray
    override suspend fun text(path: TestDataLoader.Path): String
}

expect object S3TestDataLoader : TestDataLoader {
    override suspend fun bytes(path: TestDataLoader.Path): ByteArray
    override suspend fun text(path: TestDataLoader.Path): String
}
