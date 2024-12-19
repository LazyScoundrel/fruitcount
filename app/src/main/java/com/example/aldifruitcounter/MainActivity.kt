package com.example.aldifruitcounter

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.Manifest
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var captureButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: Interpreter
    private val cameraRequestCode = 100
    private val storageRequestCode = 200

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        captureButton = findViewById(R.id.captureButton)
        resultTextView = findViewById(R.id.resultTextView)

        // Load TensorFlow Lite model
        tflite = Interpreter(loadModelFile())

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                storageRequestCode
            )
        }

        // Initialize Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), cameraRequestCode)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = androidx.camera.core.Preview.Builder().build()
                .also { it.setSurfaceProvider(viewFinder.surfaceProvider) }

            imageCapture = ImageCapture.Builder().build()

            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
        }, ContextCompat.getMainExecutor(this))

        captureButton.setOnClickListener { captureAndAnalyzeImage() }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("new_set.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun captureAndAnalyzeImage() {
        val outputFileOptions = ImageCapture.OutputFileOptions.Builder(createTempFile()).build()
        imageCapture.takePicture(outputFileOptions, ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val bitmap = outputFileResults.savedUri?.let { loadBitmapFromFile(it) } // Convert to Bitmap
                    if (bitmap != null) {
                        analyzeImage(bitmap)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("FruitCounter", "Image capture failed: ${exception.message}", exception)
                }
            })
    }

    private fun loadBitmapFromFile(uri: Uri): Bitmap {
        val source = ImageDecoder.createSource(contentResolver, uri)
        return ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            // Force the bitmap to be decoded with software rendering and ARGB_8888 format
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE)
        }
    }

    private fun analyzeImage(bitmap: Bitmap) {
        // Resize bitmap to match model input size (640x640)
        val inputImage = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        // Convert to ByteBuffer
        val inputBuffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4) // 640x640x3, 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder())  // Set correct byte order
        inputBuffer.rewind()

        for (y in 0 until 640) {
            for (x in 0 until 640) {
                val pixel = inputImage.getPixel(x, y)
                inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f) // Red
                inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255.0f)  // Green
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)       // Blue
            }
        }

        // Run inference
        val outputBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, 67, 8400), DataType.FLOAT32) // Output shape [1, 67, 8400]
        tflite.run(inputBuffer, outputBuffer.buffer.rewind())  // Running inference

        // Get the result from the output buffer
        val result = outputBuffer.floatArray

        // Debugging: Print all class outputs
        Log.d("FruitCounter", "Model Output (First 10 values):")
        for (i in 0 until 10) {  // Print the first 10 class values for debugging
            Log.d("FruitCounter", "Class $i: ${result[i]}")
        }

        // Store the count of recognized items
        val recognizedItems = mutableMapOf<String, Int>()

        // Identify the most recognized item and its count
        var itemName = ""
        var itemCount = 0

        for (i in 0 until 67) {
            // Extract the 8400 values for each class (item)
            val classValues = result.slice(i * 8400 until (i + 1) * 8400)
            val maxValue = classValues.maxOrNull() ?: 0f  // Find the highest value
            val maxIndex = classValues.indexOf(maxValue)  // Get the index of the max value

            // Get the class name using the map instead of list
            val className = getClassName(i)  // Get the class name based on the class index

            // If the value is high enough, we consider it recognized (use a threshold for confidence)
            if (maxValue > 0.5) {  // Threshold for recognition
                // Check if this item was already recognized and increment count
                recognizedItems[className] = recognizedItems.getOrDefault(className, 0) + 1
            }
        }

        // Find the most recognized fruit and its count
        if (recognizedItems.isNotEmpty()) {
            val mostRecognizedItem = recognizedItems.maxByOrNull { it.value }
            itemName = mostRecognizedItem?.key ?: "Unknown"
            itemCount = mostRecognizedItem?.value ?: 0
        }

        // Build result string in the desired format
        val itemRecognitionResult = StringBuilder()
        itemRecognitionResult.append("Item Recognized: $itemName\n")
        itemRecognitionResult.append("Count of Items: $itemCount")

        // Show the result in a TextView
        resultTextView.text = itemRecognitionResult.toString()
    }


    // Helper function to get class names, now using a map
    private fun getClassName(classIndex: Int): String {
        val classNames = mapOf(
            0 to "almond",
            1 to "apple",
            2 to "apricot",
            3 to "artichoke",
            4 to "asparagus",
            5 to "avocado",
            6 to "banana",
            7 to "bean curd/tofu",
            8 to "bell pepper/capsicum",
            9 to "blackberry",
            10 to "blueberry",
            11 to "broccoli",
            12 to "brussels sprouts",
            13 to "cantaloup/cantaloupe",
            14 to "carrot",
            15 to "cauliflower",
            16 to "cayenne/cayenne spice/cayenne pepper/cayenne pepper spice/red pepper/red pepper",
            17 to "celery",
            18 to "cherry",
            19 to "chickpea/garbanzo",
            20 to "chili/chili vegetable/chili pepper/chili pepper vegetable/chilli/chilli vegetable/chilly/chilly",
            21 to "clementine",
            22 to "coconut/cocoanut",
            23 to "edible corn/corn/maize",
            24 to "cucumber/cuke",
            25 to "date/date fruit",
            26 to "eggplant/aubergine",
            27 to "fig/fig fruit",
            28 to "garlic/ail",
            29 to "ginger/gingerroot",
            30 to "Strawberry",
            31 to "gourd",
            32 to "grape",
            33 to "green bean",
            34 to "green onion/spring onion/scallion",
            35 to "Tomato",
            36 to "kiwi fruit",
            37 to "lemon",
            38 to "lettuce",
            39 to "lime",
            40 to "mandarin orange",
            41 to "melon",
            42 to "mushroom",
            43 to "onion",
            44 to "orange/orange fruit",
            45 to "papaya",
            46 to "pea/pea food",
            47 to "peach",
            48 to "pear",
            49 to "persimmon",
            50 to "pickle",
            51 to "pineapple",
            52 to "potato",
            53 to "prune",
            54 to "pumpkin",
            55 to "radish/daikon",
            56 to "raspberry",
            57 to "strawberry",
            58 to "sweet potato",
            59 to "tomato",
            60 to "turnip",
            61 to "watermelon",
            62 to "zucchini/courgette"
        )

        return classNames[classIndex] ?: "Unknown"
    }
}
