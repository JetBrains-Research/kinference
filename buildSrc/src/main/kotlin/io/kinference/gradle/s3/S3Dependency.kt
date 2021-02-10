package io.kinference.gradle.s3

import org.gradle.api.Project
import java.io.File

class S3Dependency(project: Project, private val path: String) {
    private val s3 = S3Client(Config(project.rootProject.file("credentials.conf")))
    private val testData = project.rootProject.file("build/test-data")

    companion object {

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
    }
}
