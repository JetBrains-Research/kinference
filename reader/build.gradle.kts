group = rootProject.group
version = rootProject.version

plugins {
    kotlin("plugin.serialization") version "1.4.21"
}

dependencies {
    api(kotlin("stdlib"))
    api("com.squareup.wire", "wire-runtime", "3.6.0")
}
