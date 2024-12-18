plugins {
    kotlin("jvm") version "1.8.20"
    id("com.android.application") version "8.1.1" apply false
}

tasks.named<Delete>("clean") {
    delete(rootProject.buildDir)
}
