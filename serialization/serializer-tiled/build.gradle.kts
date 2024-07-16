group = rootProject.group
version = rootProject.version

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":serialization:serializer-protobuf"))
                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(libs.wire.runtime)
            }
        }
    }
}
