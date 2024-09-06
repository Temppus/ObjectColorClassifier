using OpenCvSharp;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;

namespace ObjectColorClassifier
{
    internal class CropColorAnnotator
    {
        private readonly ColorClassifierLib.ObjectColorClassifier _classifier;

        public CropColorAnnotator(ColorClassifierLib.ObjectColorClassifier classifier)
        {
            _classifier = classifier ?? throw new ArgumentNullException(nameof(classifier));
        }

        public void ClassifyCropsInDirectory(string sourceDir, int? downscaleToMaxSize = null)
        {
            var fullPathSourceDir = Path.GetFullPath(sourceDir);

            if (!Directory.Exists(fullPathSourceDir))
            {
                throw new DirectoryNotFoundException($"Directory {fullPathSourceDir} not found");
            }

            var sourceDirName = Path.GetFileName(fullPathSourceDir);
            var parentDir = Path.GetDirectoryName(fullPathSourceDir);
            var resultDir = Path.Combine(parentDir, $"{sourceDirName}_Annotated");

            Directory.CreateDirectory(resultDir);

            foreach (var sourceFile in Directory.GetFiles(fullPathSourceDir).OrderBy(File.GetCreationTime))
            {
                var origSourceMat = Cv2.ImRead(sourceFile);
                var sourceMatResized = ResizeToMax(origSourceMat, downscaleToMaxSize);

                sourceMatResized = ToEnrichColorsMat(sourceMatResized);

                var sw = Stopwatch.StartNew();
                var baseColor = _classifier.Classify(sourceMatResized, out var grabCut, out var x, out var y);
                sw.Stop();

                Console.WriteLine($"Color is {baseColor}. Took {sw.ElapsedMilliseconds}ms");

                var id = Guid.NewGuid().ToString()[..8];

                {
                    string prefix = $"{id}-{baseColor}-";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    sourceMatResized.SaveImage(dstFilePath);
                }

                {
                    string prefix = $"{id}-{baseColor}-GrabCut-";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    grabCut.SaveImage(dstFilePath);
                }

                {
                    string prefix = $"{id}-{baseColor}-X";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    x.SaveImage(dstFilePath);
                }

                {
                    string prefix = $"{id}-{baseColor}-Y";
                    var dstFilePath = GetResultFilePath(resultDir, prefix, sourceFile);
                    y.SaveImage(dstFilePath);
                }
            }
        }

        private static string GetResultFilePath(string resultDir, string prefix, string sourceFile)
        {
            var sourceFileName = Path.GetFileName(sourceFile);
            var resultFileName = $"{prefix}_{sourceFileName}";
            return Path.Combine(resultDir, resultFileName);
        }


        private static Mat ToEnrichColorsMat(Mat src)
        {
            // Convert the image to HSV color space
            Mat hsv = new Mat();
            Cv2.CvtColor(src, hsv, ColorConversionCodes.BGR2HSV);

            // Split the HSV image into separate channels
            Mat[] hsvChannels = Cv2.Split(hsv);

            // Adjust the saturation (second channel)
            Mat saturation = hsvChannels[1];
            saturation.ConvertTo(saturation, MatType.CV_8UC1, 1.28, 0); // Increase saturation by X%
            saturation.ConvertTo(saturation, MatType.CV_8UC1); // Ensure values are in the range 0-255

            // Adjust the value (brightness) (third channel)
            Mat value = hsvChannels[2];
            value.ConvertTo(value, MatType.CV_8UC1, 1.2, 0); // Increase brightness by X%
            value.ConvertTo(value, MatType.CV_8UC1); // Ensure values are in the range 0-255

            // Merge the adjusted channels back
            hsvChannels[1] = saturation;
            hsvChannels[2] = value;
            Mat adjustedHsv = new Mat();
            Cv2.Merge(hsvChannels, adjustedHsv);

            // Convert the adjusted HSV image back to BGR color space
            Mat result = new Mat();
            Cv2.CvtColor(adjustedHsv, result, ColorConversionCodes.HSV2BGR);
            return result;
        }

        private static Mat ResizeToMax(Mat sourceMat, int? maxSize)
        {
            if (maxSize == null)
            {
                return sourceMat.Clone();
            }

            int widthRatio = sourceMat.Width / maxSize.Value;
            int heightRatio = sourceMat.Height / maxSize.Value;

            var sourceMatResized = new Mat();

            if (widthRatio > 1 || heightRatio > 1)
            {
                var ratio = widthRatio > heightRatio ? widthRatio : heightRatio;
                var newSize = new Size(sourceMat.Width / ratio, sourceMat.Height / ratio);
                Cv2.Resize(sourceMat, sourceMatResized, newSize);
                return sourceMatResized;
            }

            return sourceMat.Clone();
        }
    }
}
