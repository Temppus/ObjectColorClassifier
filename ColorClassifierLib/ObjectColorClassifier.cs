using System.Diagnostics;
using OpenCvSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System.Runtime.InteropServices;

namespace ColorClassifierLib
{
    public class ObjectColorClassifier
    {
        private readonly IBaseColorClassifier _baseColorClassifier;

        public bool ReduceFocusArea { get; init; } = false;

        public ObjectColorClassifier(IBaseColorClassifier baseColorClassifier)
        {
            _baseColorClassifier = baseColorClassifier ?? throw new ArgumentNullException(nameof(baseColorClassifier));
        }

        public IDictionary<BaseColor, float> Classify(Mat objectCropMat, out Mat maskedGrabCutMat, out Mat segmentedMat, out Mat dominateColorMat)
        {
            Rect cropRectangle;

            if (ReduceFocusArea)
            {
                // We could specialize expected crop area for other object types
                // E.g. Pedestrians can be improved by also getting info about side position because they are more narrow ...
                var cropStartX = (int)(objectCropMat.Width * 0.05f);
                var cropStartY = (int)(objectCropMat.Height * 0.15f);
                var widthCropOffset = cropStartX * 2;
                var heightCropOffset = (int)(objectCropMat.Height / 1.8);

                cropRectangle = new Rect(cropStartX, cropStartY,
                    objectCropMat.Width - widthCropOffset,
                    objectCropMat.Height - heightCropOffset);
            }
            else
            {
                cropRectangle = new Rect(0, 0, objectCropMat.Width - 1, objectCropMat.Height - 1);
            }

            const int iterationCount = 1;

            var bgMat = new Mat();
            var fgMat = new Mat();
            var result = new Mat(objectCropMat.Size(), MatType.CV_8UC1);

            var sw = Stopwatch.StartNew();
            Cv2.GrabCut(objectCropMat, result, cropRectangle,
                bgMat, fgMat,
                iterationCount, GrabCutModes.InitWithRect);
            sw.Stop();

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
            var paletteColors = colorThief.GetPalette(img, colorCount: 2, quality: 10, ignoreWhite: false);
            var colorDescriptor = paletteColors
                // Skip first one which is assumed to be background mask color
                .Skip(ReduceFocusArea ? 1 : 0)
                .First();

            var dominateColor = colorDescriptor.Color;

            var bgrColor = new Scalar(dominateColor.B, dominateColor.G, dominateColor.R);

            // Create a new Mat object with the specified dimensions and color type
            dominateColorMat = new Mat(objectCropMat.Size(), MatType.CV_8UC3, bgrColor);

            return _baseColorClassifier.GetBaseColorConfidences(dominateColor.R, dominateColor.G, dominateColor.B);
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
    }
}
