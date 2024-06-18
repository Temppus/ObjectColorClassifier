using OpenCvSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace ColorClassifierLib
{
    public class ObjectColorClassifier : IDisposable
    {
        private readonly InferenceSession _inferenceSession;

        public ObjectColorClassifier(string classifierModelPath)
        {
            if (classifierModelPath == null) throw new ArgumentNullException(nameof(classifierModelPath));
            _inferenceSession = new InferenceSession(classifierModelPath);
        }

        public Dictionary<ObjectColor, float> Classify(Mat objectCropMat,
            out Mat maskedGrabCutMat, out Mat segmentedMat, out Mat dominateColorMat)
        {
            // These are basically hyperparameters for pedestrian
            // We could specialize expected crop area for other object types
            // E.g. Pedestrians can be improved by also getting info about side position because they are more narrow ...
            var cropStartX = (int)(objectCropMat.Width * 0.14f);
            var cropStartY = (int)(objectCropMat.Height * 0.16f);
            var widthCropOffset = cropStartX * 2;
            var heightCropOffset = (int)(objectCropMat.Height / 1.8);

            var cropRectangle = new Rect(cropStartX, cropStartY,
                objectCropMat.Width - widthCropOffset,
                objectCropMat.Height - heightCropOffset);

            const int iterationCount = 3;

            var bgMat = new Mat();
            var fgMat = new Mat();
            var result = new Mat(objectCropMat.Size(), MatType.CV_8UC1);

            Cv2.GrabCut(objectCropMat, result, cropRectangle,
                bgMat, fgMat,
                iterationCount, GrabCutModes.InitWithRect);

            maskedGrabCutMat = ((result & 1)) * 255;

            // Create an output image initialized to white
            var backgroundColorMat = new Mat(objectCropMat.Size(), objectCropMat.Type(), new Scalar(255, 0, 255));

            // Copy the source image to the output image using the mask
            objectCropMat.CopyTo(backgroundColorMat, maskedGrabCutMat);

            //TODO: create clone ?
            segmentedMat = backgroundColorMat;

            // render rectangle to source mat
            objectCropMat.Rectangle(cropRectangle, Scalar.DeepSkyBlue, thickness: 4);

            using var img = MatToImageSharp(backgroundColorMat);

            var colorThief = new ColorThief.ImageSharp.ColorThief();
            var paletteColors = colorThief.GetPalette(img, 2, 10, false);
            var colorDescriptor = paletteColors
                // Skip first one which is assumed to be background mask color
                .Skip(1)
                .First();

            var dominateColor = colorDescriptor.Color;

            var bgrColor = new Scalar(dominateColor.B, dominateColor.G, dominateColor.R);

            // Create a new Mat object with the specified dimensions and color type
            dominateColorMat = new Mat(objectCropMat.Size(), MatType.CV_8UC3, bgrColor);

            // RUN INFERENCE
            float[] inputVector = { dominateColor.R, dominateColor.G, dominateColor.B };

            var inputTensor = new DenseTensor<float>(inputVector,
                // one row, 3 values
                new[] { 1, 3 });

            // Create input container
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("dense_12_input", inputTensor)
            };

            // Run inference
            using var results = _inferenceSession.Run(inputs);

            var outputTensor = results[0].AsTensor<float>();

            // Assuming output tensor shape is [unk__32, 11]
            int numSamples = outputTensor.Dimensions[0];
            int numClasses = outputTensor.Dimensions[1];

            var colorConfidenceMap = new Dictionary<ObjectColor, float>();

            for (int i = 0; i < numSamples; i++)
            {
                // Find the index of the highest confidence score
                float maxConfidence = float.MinValue;
                int colorIndex = -1;
                for (int j = 0; j < numClasses; j++)
                {
                    var confidence = outputTensor[i, j];
                    var objColor = (ObjectColor)j;

                    if (confidence > maxConfidence)
                    {
                        maxConfidence = outputTensor[i, j];
                        colorIndex = j;
                    }

                    colorConfidenceMap.Add(objColor, confidence);
                }

                return colorConfidenceMap;
            }

            throw new InvalidOperationException("");
        }

        private static Image<Rgba32> MatToImageSharp(Mat mat)
        {
            // Ensure the mat is in the format of 8-bit, 3-channel (BGR) or 4-channel (BGRA)
            if (mat.Type() != MatType.CV_8UC3 && mat.Type() != MatType.CV_8UC4)
            {
                throw new ArgumentException("The Mat type is not supported.");
            }

            // Convert Mat to a byte array
            int width = mat.Width;
            int height = mat.Height;
            int channels = mat.Channels();
            int stride = width * channels;
            byte[] data = new byte[height * stride];
            Marshal.Copy(mat.Data, data, 0, data.Length);

            // Create ImageSharp Image from byte array
            Image<Rgba32> imageSharpImage;
            if (channels == 3)
            {
                // BGR to RGBA
                imageSharpImage = Image.LoadPixelData<Bgr24>(data, width, height).CloneAs<Rgba32>();
            }
            else
            {
                // BGRA to RGBA
                imageSharpImage = Image.LoadPixelData<Bgra32>(data, width, height).CloneAs<Rgba32>();
            }

            return imageSharpImage;
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}
