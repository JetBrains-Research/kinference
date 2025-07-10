package io.kinference.gradle.s3

import org.gradle.api.Project
import org.gradle.api.Task
import java.io.File

class S3Dependency(project: Project, private val path: String) {
    private val s3 = S3Client(Config(project.rootProject.file("credentials.conf")))
    private val testData = project.rootProject.file("test-data/s3")

    companion object {
        fun withDefaultS3Dependencies(task: Task) = with(task) {
            val dependencies = Context(this.project).apply { defaultS3Deps() }
            dependencies.resolve()
        }

        fun s3Test(project: Project, name: String): S3Dependency {
            return S3Dependency(project, "tests/${name.replace(":", "/")}/")
        }
    }

    fun resolve() {
        testData.mkdirs()
        s3.copyObjects(path, File(testData, path))
    }

    class Context(private val project: Project, internal val dependencies: ArrayList<S3Dependency> = ArrayList()) {
        fun s3Test(name: String) {
            dependencies.add(Companion.s3Test(project, name))
        }

        fun resolve() = dependencies.forEach { it.resolve() }
    }
}
